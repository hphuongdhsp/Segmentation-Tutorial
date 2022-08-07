import multiprocessing as mproc
import os
from typing import Any

import albumentations as A
import cv2
import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from addict import Dict as Adict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transform import resize
from utils import get_all_items, get_args, has_file_allowed_extension, img2tensor, read_image, read_mask


class NailDataset(Dataset):
    def __init__(self, data_root: str, csv_folder: str, train: str, tfms: A.Compose):
        self.data_root = data_root
        self.csv_folder = csv_folder
        self.train = train
        self.tfms = tfms
        if self.train == "train":
            self.ids = pd.read_csv(os.path.join(self.csv_folder, "train.csv"))["images"]
        else:
            self.ids = pd.read_csv(os.path.join(self.csv_folder, "valid.csv"))["images"]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Any:
        fname = self.ids[idx]

        image = read_image(self.data_root + f"/{self.train}/images" + fname)
        mask = read_mask(self.data_root + f"/{self.train}/masks" + fname)

        mask = (mask > 0).astype(np.uint8)
        if self.tfms is not None:
            augmented = self.tfms(image=image, mask=mask)
            img, mask = augmented["image"], augmented["mask"]
        return {
            "image": K.image_to_tensor((img / 255).astype(np.float32, copy=False)),
            "label": K.image_to_tensor(mask.astype(np.float32, copy=False)),
        }


class TestDataset(Dataset):
    def __init__(self, folder: str, traintfms=None):
        self.folder = folder
        self.extensions = [".jpg", ".jpeg", ".png"]
        self.allfile = get_all_items(folder)
        self.ids = [i for i in self.allfile if has_file_allowed_extension(i, self.extensions)]
        self.tfms = traintfms

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Any:
        fname = self.folder + self.ids[idx]
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        width = img.shape[0]
        hight = img.shape[1]

        if self.tfms is not None:
            augmented = self.tfms(image=img)
            img = augmented["image"]
        return img2tensor(img), width, hight


class NailSegmentation(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        csv_path: str,
        test_path: str,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        assert os.path.isdir(csv_path), f"missing folder: {csv_path}"
        assert os.path.isdir(data_root), f"missing folder: {data_root}"
        self.data_root = str(data_root)
        self.csv_path = str(csv_path)
        self.test_path = str(test_path)

        self.valid_transform = resize()
        self.train_transform = resize()

        # other configs
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else mproc.cpu_count()

    def prepare_data(self) -> None:
        pass

    def setup(self, *_, **__) -> None:

        self.train_dataset = NailDataset(
            self.data_root,
            self.csv_path,
            train="train",
            tfms=self.train_transform,
        )
        self.valid_dataset = NailDataset(
            self.data_root,
            self.csv_path,
            train="valid",
            tfms=self.valid_transform,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


def data_visualization(dataset: Dataset, masks: bool = True) -> None:
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    dict_image_label = next(iter(dataloader))
    imgs = dict_image_label["image"]
    masks = dict_image_label["label"]
    plt.figure(figsize=(64, 64))
    for i, (img, mask) in enumerate(zip(imgs, masks)):
        img = ((img.permute(1, 2, 0) / 2 + 0.5) * 255.0).numpy().astype(np.uint8)
        plt.subplot(4, 4, i + 1)
        plt.imshow(img, vmin=0, vmax=255)
        plt.imshow(mask[0].squeeze().numpy(), alpha=0.5)
        plt.axis("off")
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.show()
    del dataloader, imgs, masks


if __name__ == "__main__":

    args = get_args()
    with open(args.config_path) as file:
        cfg = Adict(yaml.load(file, Loader=yaml.SafeLoader))
    data_root = args.data_root
    datamodule = NailSegmentation(
        data_root=data_root,
        csv_path=f"{data_root}/csv_file",
        test_path="",
        batch_size=16,
        num_workers=4,
    )

    datamodule.setup()
    train_dataset = datamodule.train_dataset

    data_visualization(train_dataset)
