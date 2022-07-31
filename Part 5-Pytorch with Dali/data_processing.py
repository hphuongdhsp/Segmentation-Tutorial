import os

import numpy as np
import pandas as pd
from typing import Any
from transform import normalize
from utils import get_all_items, get_args, mkdir, read_image, read_mask


def png2numpy(args):
    data_root = args.data_root

    train_images_dir = os.path.join(data_root, "train", "images")
    train_masks_dir = os.path.join(data_root, "train", "masks")
    valid_images_dir = os.path.join(data_root, "valid", "images")
    valid_masks_dir = os.path.join(data_root, "valid", "masks")

    data_root_npy = os.path.join(data_root, "numpy_format")
    mkdir(data_root_npy)
    mkdir(os.path.join(data_root_npy, "train", "images"))
    mkdir(os.path.join(data_root_npy, "train", "masks"))
    mkdir(os.path.join(data_root_npy, "valid", "images"))
    mkdir(os.path.join(data_root_npy, "valid", "masks"))

    do_normalize = normalize(p=1)

    for item in get_all_items(train_images_dir):

        try:
            image = read_image(os.path.join(train_images_dir, item[1:]))
            image = do_normalize(image=image)["image"]

            mkdir(os.path.join(data_root_npy, "train", "images", item[1:].rpartition("/")[0]))
            des = os.path.join(data_root_npy, "train", "images", item[1:].split(".")[0] + ".npy")
            np.save(des, image)
        except Exception as e:
            print(e)
            print(item)

    for item in get_all_items(valid_images_dir):

        try:
            mkdir(os.path.join(data_root_npy, "valid", "images", item[1:].rpartition("/")[0]))
            image = read_image(os.path.join(valid_images_dir, item[1:]))
            image = do_normalize(image=image)["image"]
            des = os.path.join(data_root_npy, "valid", "images", item[1:].split(".")[0] + ".npy")
            np.save(des, image)
        except Exception as e:
            print(e)
            print(item)

    for item in get_all_items(train_masks_dir):

        try:
            mkdir(os.path.join(data_root_npy, "train", "masks", item[1:].rpartition("/")[0]))
            mask = read_mask(os.path.join(train_masks_dir, item[1:])) / 255
            mask = np.expand_dims(mask, 2)
            # mask =  do_resize(image = mask)["image"]
            des = os.path.join(data_root_npy, "train", "masks", item[1:].split(".")[0] + ".npy")
            np.save(des, mask)
        except Exception as e:
            print(e)
            print(item)

    for item in get_all_items(valid_masks_dir):

        try:
            mkdir(os.path.join(data_root_npy, "valid", "masks", item[1:].rpartition("/")[0]))
            mask = read_mask(os.path.join(valid_masks_dir, item[1:])) / 255
            mask = np.expand_dims(mask, 2)
            # mask =  do_resize(image= mask)["image"]
            des = os.path.join(data_root_npy, "valid", "masks", item[1:].split(".")[0] + ".npy")
            np.save(des, mask)
        except Exception as e:
            print(e)
            print(item)
    print(mask.shape)


def make_csv_file_npy(args) -> None:
    data_root_npy = args.data_root_npy

    list_images_train_masks = get_all_items(os.path.join(data_root_npy, "train", "masks"))

    list_images_train_images = get_all_items(os.path.join(data_root_npy, "train", "images"))

    list_images_train = [
        os.path.join("train", "images", i[1:]) for i in list_images_train_images if i in list_images_train_masks
    ]
    list_masks_train = [
        os.path.join("train", "masks", i[1:]) for i in list_images_train_images if i in list_images_train_masks
    ]

    dict_train = {"images": list_images_train, "masks": list_masks_train}

    list_images_valid_masks = get_all_items(os.path.join(data_root_npy, "valid", "images"))
    list_images_valid_images = get_all_items(os.path.join(data_root_npy, "valid", "masks"))

    list_images_valid = [
        os.path.join("valid", "images", i[1:]) for i in list_images_valid_images if i in list_images_valid_masks
    ]

    list_masks_valid = [
        os.path.join("valid", "masks", i[1:]) for i in list_images_valid_images if i in list_images_valid_masks
    ]

    dict_valid = {"images": list_images_valid, "masks": list_masks_valid}

    train_frame = pd.DataFrame(dict_train)

    valid_frame = pd.DataFrame(dict_valid)

    mkdir(f"{data_root_npy}/csv_file")

    train_frame.to_csv(f"{data_root_npy}/csv_file/train.csv", index=False)
    valid_frame.to_csv(f"{data_root_npy}/csv_file/valid.csv", index=False)


if __name__ == "__main__":

    args = get_args()
    print("Make the numpy format dataset")
    png2numpy(args)
    make_csv_file_npy(args)
    print("Done")
