import os

import numpy as np
import pandas as pd
from typing import Any
from transform import normalize, resize
from utils import get_all_items, get_args, mkdir, read_image, read_mask


def make_csv_file(args: Any) -> None:
    """This function is to make csv files (train and valid) containing all the images and their masks paths

    Args:
        args (Any): args
    """

    data_root = args.data_root

    list_images_train_masks = get_all_items(os.path.join(data_root, "train", "masks"))

    list_images_train_images = get_all_items(os.path.join(data_root, "train", "images"))

    list_images_train = [i for i in list_images_train_images if i in list_images_train_masks]

    print(len(list_images_train))
    list_images_valid = get_all_items(os.path.join(data_root, "valid", "masks"))

    train_frame = pd.DataFrame(list_images_train, columns=["images"])

    train_frame["train"] = 1
    valid_frame = pd.DataFrame(list_images_valid, columns=["images"])

    valid_frame["train"] = 0
    mkdir(f"{data_root}/csv_file")
    train_frame.to_csv(f"{data_root}/csv_file/train.csv", index=False)
    valid_frame.to_csv(f"{data_root}/csv_file/valid.csv", index=False)


def png2numpy(args):
    data_root = args.data_root

    train_images_dir = os.path.join(data_root, "train", "images")
    train_masks_dir = os.path.join(data_root, "train", "masks")
    valid_images_dir = os.path.join(data_root, "valid", "images")
    valid_masks_dir = os.path.join(data_root, "valid", "masks")

    new_dir = os.path.join(data_root, "numpy_format")
    mkdir(new_dir)
    mkdir(os.path.join(new_dir, "train", "images"))
    mkdir(os.path.join(new_dir, "train", "masks"))
    mkdir(os.path.join(new_dir, "valid", "images"))
    mkdir(os.path.join(new_dir, "valid", "masks"))

    do_resize = resize(p=1)
    do_normalize = normalize(p=1)

    for item in get_all_items(train_images_dir):

        try:
            image = read_image(os.path.join(train_images_dir, item[1:]))
            image = do_normalize(image=image)["image"]

            mkdir(os.path.join(new_dir, "train", "images", item[1:].rpartition("/")[0]))
            des = os.path.join(new_dir, "train", "images", item[1:].split(".")[0] + ".npy")
            np.save(des, image)
        except Exception as e:
            print(e)
            print(item)

    for item in get_all_items(valid_images_dir):

        try:
            mkdir(os.path.join(new_dir, "valid", "images", item[1:].rpartition("/")[0]))
            image = read_image(os.path.join(valid_images_dir, item[1:]))
            image = do_normalize(image=image)["image"]
            des = os.path.join(new_dir, "valid", "images", item[1:].split(".")[0] + ".npy")
            np.save(des, image)
        except Exception as e:
            print(e)
            print(item)

    for item in get_all_items(train_masks_dir):

        try:
            mkdir(os.path.join(new_dir, "train", "masks", item[1:].rpartition("/")[0]))
            mask = read_mask(os.path.join(train_masks_dir, item[1:])) / 255
            mask = np.expand_dims(mask, 2)
            # mask =  do_resize(image = mask)["image"]
            des = os.path.join(new_dir, "train", "masks", item[1:].split(".")[0] + ".npy")
            np.save(des, mask)
        except Exception as e:
            print(e)
            print(item)

    for item in get_all_items(valid_masks_dir):

        try:
            mkdir(os.path.join(new_dir, "valid", "masks", item[1:].rpartition("/")[0]))
            mask = read_mask(os.path.join(valid_masks_dir, item[1:])) / 255
            mask = np.expand_dims(mask, 2)
            # mask =  do_resize(image= mask)["image"]
            des = os.path.join(new_dir, "valid", "masks", item[1:].split(".")[0] + ".npy")
            np.save(des, mask)
        except Exception as e:
            print(e)
            print(item)
    print(mask.shape)


def make_csv_file_npy(base_path: str) -> None:

    list_images_train_masks = get_all_items(os.path.join(base_path, "train", "masks"))

    list_images_train_images = get_all_items(os.path.join(base_path, "train", "images"))

    list_images_train = [
        os.path.join("train", "images", i[1:]) for i in list_images_train_images if i in list_images_train_masks
    ]
    list_masks_train = [
        os.path.join("train", "masks", i[1:]) for i in list_images_train_images if i in list_images_train_masks
    ]

    print(len(list_images_train))
    print(len(list_masks_train))

    dict_train = {"images": list_images_train, "masks": list_masks_train}

    list_images_valid_masks = get_all_items(os.path.join(base_path, "valid", "images"))
    list_images_valid_images = get_all_items(os.path.join(base_path, "valid", "masks"))

    list_images_valid = [
        os.path.join("valid", "images", i[1:]) for i in list_images_valid_images if i in list_images_valid_masks
    ]

    list_masks_valid = [
        os.path.join("valid", "masks", i[1:]) for i in list_images_valid_images if i in list_images_valid_masks
    ]

    print(len(list_images_valid))
    print(len(list_masks_valid))
    dict_valid = {"images": list_images_valid, "masks": list_masks_valid}

    train_frame = pd.DataFrame(dict_train)

    valid_frame = pd.DataFrame(dict_valid)

    mkdir("./csv_file")
    train_frame.to_csv("./csv_file/train_npy.csv", index=False)
    valid_frame.to_csv("./csv_file/valid_npy.csv", index=False)


if __name__ == "__main__":

    args = get_args()
    print("Make the training dataframe and valid dataframe")
    make_csv_file(args)
    # base_path = args.base_path
    # tails = ["jpeg", "jpg", "PNG", "JPEG", "JPG"]
    # for tail in tails:
    #     rename2png(str(base_path), tail)
    # make_csv_file(base_path)
