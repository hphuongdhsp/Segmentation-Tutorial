import multiprocessing
from functools import partial
from pathlib import Path
from typing import Any, List, Tuple, Union

import albumentations as A
import pandas as pd
import tensorflow as tf

from transform import train_transform, valid_transform
from utils import get_args

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_data_path(data_root: Union[str, Path], csv_dir: Union[str, Path], train: str) -> Tuple:
    r"""This function is to make the the data path for images and masks.
    You can customize this function to suit with your data

    Args:
        base_path (Union[str, Path]): the base_path that stored all of data
        csv_dir (Union[str, Path]): the csv file path, that store the image path of training or validation set
        train (str): the pharse, train or valid

    Returns:
        Tuple: tuple of 2 lists, which stores the path of images and masks.
    """
    csv_file = pd.read_csv(csv_dir)
    ids = sorted(csv_file["images"])
    #
    images = [data_root + f"/{train}/images" + fname for fname in ids]
    masks = [data_root + f"/{train}/masks" + fname for fname in ids]
    return (images, masks)



def set_shapes(image:Any, mask:Any, img_shape:int=384):
    image.set_shape((img_shape, img_shape, 3))
    mask.set_shape((img_shape, img_shape, 1))
    return image, mask


def load_image_and_mask_from_path(image_path: tf.string, mask_path: tf.string) -> Any:
    """this function is to load image and mask

    Args:
        image_path (tf.string): the tensorflow string of image
        mask_path (tf.string): the tensorflow string of mask

    Returns:
        [type]: image and mask
    """
    # read image by tensorflow function
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    # read mask by tensorflow function
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1)
    return img, mask


def tf_dataset(
    dataset: Tuple[List[str], List[str]],
    shuffle: bool,
    batch_size: Any,
    transforms: A.Compose,
    dtype: Any,
    device: List[int],
):
    r"""This function is to create dataloader for tensorflow training

    Args:
        dataset Tuple[List[str], List[str]]: Tuple of List data path that have same size
        shuffle (bool): True if you want shuffle dataset when do training
        batch_size [Any]: None if you dont want spit dataset by batch
        transforms (A.Compose): the augumentation that you want to apple for the data

    Returns:
        datast : the prepare dataset for the training step
    """

    # do augumentation by albumentations, remark that in the the end, we use tf.cast to normalize
    # image and mask and also make sure that the out put of this function be in form of tensorflow (tf)
    def aug_fn(image, mask):
        # do augumentation by albumentations library
        data = {"image": image, "mask": mask}
        aug_data = transforms(**data)
        aug_img = aug_data["image"]
        aug_mask = aug_data["mask"]
        # do normalize by using the tensorflow.cast function
        aug_img = tf.cast(aug_img / 255.0, dtype)
        aug_mask = tf.cast(aug_mask / 255.0, dtype)
        return aug_img, aug_mask

    def process_data(image, mask):
        # using tf.numpy_function to apply the aug_img to image and mask
        aug_img, aug_mask = tf.numpy_function(aug_fn, [image, mask], [dtype, dtype])
        return aug_img, aug_mask

    # convert the tuple of list (images, masks) into the tensorflow.data form
    dataset = tf.data.Dataset.from_tensor_slices(dataset)

    # apply the map reading image and mask (make sure that the input and output are in the tensorflow form (tf.))
    dataset = dataset.map(load_image_and_mask_from_path, num_parallel_calls=multiprocessing.cpu_count() // len(device))
    # shuffle data
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100000)
    # do the process_data map (augumentation and normalization)
    dataset = dataset.map(
        partial(process_data), num_parallel_calls=multiprocessing.cpu_count() // len(device)
    ).prefetch(AUTOTUNE)
    # make batchsize, here we use batch_size as a parameter, in some case we dont split dataset by batchsize
    # for example, if we want to mix multi-dataset, then we skip this step and split dataset by batch_size later
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset


if __name__ == "__main__":
    # test dataset
    args = get_args()

    data_root = str(args.data_root)
    train_csv_dir = f"{data_root}/csv_file/train.csv"
    valid_csv_dir = f"{data_root}/csv_file/valid.csv"
    dtype = tf.float16
    train_dataset = load_data_path(data_root, train_csv_dir, "train")
    train_loader = tf_dataset(train_dataset, True, 16, train_transform(), dtype, [0])
    valid_dataset = load_data_path(data_root, valid_csv_dir, "valid")
    valid_loader = tf_dataset(valid_dataset, True, 16, valid_transform(), dtype, [0])

    for image, mask in train_loader.take(1):
        print(image.shape, mask.shape)
        break
