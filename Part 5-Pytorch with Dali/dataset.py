import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import pandas as pd
import yaml
from addict import Dict as Adict
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from pytorch_lightning import LightningDataModule

from utils import get_args


def random_augmentation(probability, augmented, original):
    condition = fn.cast(fn.random.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
    neg_condition = condition ^ True
    return condition * augmented + neg_condition * original


class GenericPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super().__init__(batch_size, num_threads, device_id)
        self.kwargs = kwargs
        self.dim = kwargs["dim"]
        self.device = device_id
        # self.patch_size = [384,384]
        self.load_to_gpu = kwargs["load_to_gpu"]

        self.input_x = self.get_reader(kwargs["imgs"])
        self.input_y = self.get_reader(kwargs["lbls"])
        self.cast = ops.Cast(device="gpu", dtype=types.DALIDataType.FLOAT)

    def get_reader(self, data):
        return ops.readers.Numpy(
            files=data,
            device="cpu",
            read_ahead=True,
            dont_use_mmap=True,
            pad_last_batch=True,
            shard_id=self.device,
            seed=self.kwargs["seed"],
            num_shards=self.kwargs["gpus"],
            shuffle_after_epoch=self.kwargs["shuffle"],
        )

    def load_data(self):
        img = self.input_x(name="ReaderX")  # read X
        img = img.gpu()
        img = self.cast(img)
        if self.input_y is not None:
            lbl = self.input_y(name="ReaderY")  # read Y
            lbl = lbl.gpu()
            lbl = self.cast(lbl)

            return img, lbl
        return img

    def crop_fn(self, img, lbl):
        img, lbl = self.crop(img), self.crop(lbl)
        return img, lbl

    def resize_fn(self, img, lbl):
        img, lbl = fn.resize(img, size=[384, 384]), fn.resize(lbl, size=[384, 384])
        return img, lbl

    def transpose_fn(self, img, lbl):
        img, lbl = fn.transpose(img, perm=[2, 0, 1]), fn.transpose(lbl, perm=[2, 0, 1])
        return img, lbl


class TrainPipeline(GenericPipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super().__init__(batch_size, num_threads, device_id, **kwargs)
        self.oversampling = kwargs["oversampling"]

    """
    define some augumentations, for more augumentation, please read \
        https://github.com/NVIDIA/DALI/tree/main/docs/examples/image_processing
    """

    @staticmethod
    def slice_fn(img):
        return fn.slice(img, 1, 3, axes=[0])

    def resize(self, data, interp_type):
        return fn.resize(data, interp_type=interp_type, size=self.crop_shape_float)

    def noise_fn(self, img):
        img_noised = img + fn.random.normal(img, stddev=fn.random.uniform(range=(0.0, 0.33)))
        return random_augmentation(0.15, img_noised, img)

    def blur_fn(self, img):
        img_blurred = fn.gaussian_blur(img, sigma=fn.random.uniform(range=(0.5, 1.5)))
        return random_augmentation(0.15, img_blurred, img)

    def brightness_fn(self, img):
        brightness_scale = random_augmentation(0.15, fn.random.uniform(range=(0.7, 1.3)), 1.0)
        return img * brightness_scale

    def contrast_fn(self, img):
        scale = random_augmentation(0.13, fn.random.uniform(range=(0.9, 1.1)), 1.0)
        return math.clamp(img * scale, fn.reductions.min(img), fn.reductions.max(img))

    def flips_fn(self, img, lbl):
        kwargs = {
            "horizontal": fn.random.coin_flip(probability=0.5),
            "vertical": fn.random.coin_flip(probability=0.5),
        }
        return fn.flip(img, **kwargs), fn.flip(lbl, **kwargs)

    def define_graph(self):
        img, lbl = self.load_data()
        img, lbl = self.resize_fn(img, lbl)  # reszie to inpput size (384)

        img, lbl = self.flips_fn(img, lbl)
        img = self.noise_fn(img)
        img = self.blur_fn(img)
        img = self.brightness_fn(img)
        img = self.contrast_fn(img)

        img, lbl = self.transpose_fn(img, lbl)
        return img, lbl


class ValidPipeline(GenericPipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super().__init__(batch_size, num_threads, device_id, **kwargs)
        # self.invert_resampled_y = kwargs["invert_resampled_y"]
        # if self.invert_resampled_y:
        #     self.input_meta = self.get_reader(kwargs["meta"])
        #     self.input_orig_y = self.get_reader(kwargs["orig_lbl"])
        print(len(kwargs["imgs"]))

    def define_graph(self):
        img, lbl = self.load_data()
        img, lbl = self.resize_fn(img, lbl)  # reszie to inpput size (384)
        img, lbl = self.transpose_fn(img, lbl)

        return img, lbl


PIPELINES = {"train": TrainPipeline, "eval": ValidPipeline}


class LightningWrapper(DALIGenericIterator):
    def __init__(self, pipe, **kwargs):
        super().__init__(pipe, **kwargs)

    def __next__(self):
        out = super().__next__()[0]
        return out


def fetch_dali_loader(imgs, lbls, batch_size, mode, **kwargs):
    assert len(imgs) > 0, "Empty list of images!"
    if lbls is not None:
        assert len(imgs) == len(lbls), f"Number of images ({len(imgs)}) not matching number of labels ({len(lbls)})"
    pipeline = PIPELINES[mode]
    shuffle = True if mode == "train" else False
    load_to_gpu = True if mode in ["eval", "test"] else False
    pipe_kwargs = {"imgs": imgs, "lbls": lbls, "load_to_gpu": load_to_gpu, "shuffle": shuffle, **kwargs}

    rank = int(os.getenv("LOCAL_RANK", "0"))
    pipe = pipeline(batch_size, kwargs["num_workers"], rank, **pipe_kwargs)
    return pipe


class NailSegmentationDaliDali(LightningDataModule):
    def __init__(self, data_root_npy: str, batch_size: int, csv_folder: str):
        super().__init__()

        self.data_root_npy = str(data_root_npy)
        self.csv_folder = csv_folder
        self.batch_size = batch_size
        self.train_csv = pd.read_csv(os.path.join(self.csv_folder, "train.csv"))
        self.valid_csv = pd.read_csv(os.path.join(self.csv_folder, "valid.csv"))

        self.kwargs = {
            "dim": 2,
            "seed": 42,
            "gpus": 1,
            "overlap": 0.5,
            "benchmark": False,
            "num_workers": 4,
            "oversampling": 0.4,
            "test_batches": 0,
            "train_batches": 0,
            "load_to_gpu": True,
        }

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        self.train_imgs = [os.path.join(self.data_root_npy, path) for path in self.train_csv["images"]]
        self.train_lbls = [os.path.join(self.data_root_npy, path) for path in self.train_csv["masks"]]

        self.val_imgs = [os.path.join(self.data_root_npy, path) for path in self.valid_csv["images"]]
        self.val_lbls = [os.path.join(self.data_root_npy, path) for path in self.valid_csv["masks"]]

        self.train_dataset = fetch_dali_loader(
            imgs=self.train_imgs, lbls=self.train_lbls, batch_size=self.batch_size, mode="train", **self.kwargs
        )
        self.valid_dataset = fetch_dali_loader(
            imgs=self.val_imgs, lbls=self.val_lbls, batch_size=self.batch_size, mode="eval", **self.kwargs
        )

    def train_dataloader(self):
        output_map = ["image", "label"]
        return LightningWrapper(
            self.train_dataset,
            auto_reset=True,
            reader_name="ReaderX",
            output_map=output_map,
            dynamic_shape=False,
        )

    def val_dataloader(self):
        output_map = ["image", "label"]
        return LightningWrapper(
            self.valid_dataset,
            auto_reset=True,
            reader_name="ReaderX",
            output_map=output_map,
            dynamic_shape=True,
        )


def transpose_fn(img, lbl):
    img, lbl = fn.transpose(img, perm=[1, 2, 0]), fn.transpose(lbl, perm=[1, 2, 0])
    return img, lbl


def show_images(image_batch, mask_batch):
    # image_batch, mask_batch = transpose_fn(image_batch, mask_batch)
    columns = 4
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize=(32, (32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows * columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch.as_cpu().at(j))
        plt.imshow(mask_batch.as_cpu().at(j), alpha=0.5)
        plt.axis("off")
        plt.subplots_adjust(wspace=None, hspace=None)

    plt.show()


if __name__ == "__main__":

    args = get_args()
    with open(args.config_path) as file:
        cfg = Adict(yaml.load(file, Loader=yaml.SafeLoader))
    data_root_npy = args.data_root_npy
    batch_size = 4
    csv_folder = f"{data_root_npy}/csv_file"
    datamodule = NailSegmentationDaliDali(data_root_npy=data_root_npy, batch_size=batch_size, csv_folder=csv_folder)

    datamodule.setup()
    valid_dataset = datamodule.train_dataset
    valid_dataset.build()
    (images, masks) = valid_dataset.run()
    len(masks)
