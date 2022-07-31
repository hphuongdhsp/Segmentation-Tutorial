import os
from typing import Any
import pandas as pd

from utils import get_all_items, get_args, mkdir

def make_csv_file(args: Any) -> None:
    """This function is to make csv files (train and valid) containing all the images and their masks paths

    Args:
        args (Any): args
    """

    data_root = args.data_root


    list_images_train_masks = get_all_items(os.path.join(data_root, "train", "masks"))

    list_images_train_images = get_all_items(os.path.join(data_root, "train", "images"))

    list_images_train = [
        i for i in list_images_train_images if i in list_images_train_masks
    ]

    print(len(list_images_train))
    list_images_valid = get_all_items(os.path.join(data_root, "valid", "masks"))

    train_frame = pd.DataFrame(list_images_train, columns=["images"])

    train_frame["train"] = 1
    valid_frame = pd.DataFrame(list_images_valid, columns=["images"])

    valid_frame["train"] = 0
    mkdir(f"{data_root}/csv_file")
    train_frame.to_csv(f"{data_root}/csv_file/train.csv", index=False)
    valid_frame.to_csv(f"{data_root}/csv_file/valid.csv", index=False)


def split_data_train(args: Any) -> None:
    r"""
    This function is to split the train into some subsets. The purpose of this step is to make the balanced dataset.
    """
    data_root = args.data_root
    path_csv = f"{data_root}/csv_file/train.csv"
    train = pd.read_csv(path_csv)
    train["type"] = train["images"].apply(lambda x: x.split("/")[1])
    for i in train["type"].unique().tolist():
        df = train.loc[train["type"] == i]
        df.to_csv(f"{data_root}/csv_file/train{i}.csv", index=False)


if __name__ == "__main__":

    args = get_args()
    print("Make the training dataframe and valid dataframe")
    make_csv_file(args)
    split_data_train(args)
