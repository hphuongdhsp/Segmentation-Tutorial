import glob
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
import nvidia.dali.types as DTypes
from utils import get_args

seed = 42
batch_size = 4


def show_images(image_batch, mask_batch):
    # image_batch, mask_batch = transpose_fn(image_batch, mask_batch)
    print(image_batch[0])
    columns = 4
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize=(32, (32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows * columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch.at(j))
        plt.imshow(mask_batch.at(j), alpha=0.5)
        plt.axis("off")
        plt.subplots_adjust(wspace=None, hspace=None)

    plt.show()


def random_augmentation(probability, augmented, original):
    condition = fn.cast(fn.random.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
    neg_condition = condition ^ True
    return condition * augmented + neg_condition * original


def noise_fn(image, mask):
    img_noised = image + fn.random.normal(image, stddev=fn.random.uniform(range=(0.0, 0.1)))
    return random_augmentation(0.5, img_noised, image), mask


def resize_fn(image, mask):
    image, mask = fn.resize(image, size=[384, 384]), fn.resize(mask, size=[384, 384])
    return image, mask


def brightness_fn(image, mask):
    brightness_scale = random_augmentation(0.5, fn.random.uniform(range=(0.7, 1.3)), 1.0)
    return image * brightness_scale, mask


def contrast_fn(image, mask):
    scale = random_augmentation(0.13, fn.random.uniform(range=(0.9, 1.1)), 1.0)
    return math.clamp(image * scale, fn.reductions.min(image), fn.reductions.max(image)), mask


def flips_fn(image, mask):
    kwargs = {
        "horizontal": fn.random.coin_flip(probability=0.5),
        "vertical": fn.random.coin_flip(probability=0.5),
    }
    return fn.flip(image, **kwargs), fn.flip(mask, **kwargs)


def normalize_fn(image, mask):
    image = fn.normalize(image, scale=64, shift=128, axis_names="HW")
    return image, mask


def cast(image, mask):
    image = fn.cast(image, dtype=DTypes.DALIDataType.FLOAT)
    mask = fn.cast(mask, dtype=DTypes.DALIDataType.FLOAT)
    return image, mask


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=0)
def image_decoder_pipeline(data, label):
    images = fn.readers.numpy(files=data, device="cpu")
    masks = fn.readers.numpy(files=label, device="cpu")
    images, masks = cast(image=images, mask=masks)
    images, masks = resize_fn(image=images, mask=masks)
    images, masks = flips_fn(image=images, mask=masks)
    # images, masks = normalize_fn(image=images, mask=masks)

    return (images, masks)


def main():
    args = get_args()
    data_root_npy = os.path.join(args.data_root_npy)

    image_dir = os.path.join(data_root_npy, "valid/images")
    mask_dir = os.path.join(data_root_npy, "valid/masks")

    data = sorted(glob.glob(os.path.join(image_dir, "*npy")))
    label = sorted(glob.glob(os.path.join(mask_dir, "*npy")))

    pipe = image_decoder_pipeline(data, label)
    pipe.build()
    (images, masks) = pipe.run()

    show_images(images, masks)


if __name__ == "__main__":
    main()
