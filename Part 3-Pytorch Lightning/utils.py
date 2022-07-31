import argparse
import errno
import glob
import os
import pydoc
from pathlib import Path
from typing import Any, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def get_args() -> Any:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "-c",
        "--config_path",
        default="./configs/config.yaml",
        type=Path,
        help="Path to the config.",
        required=False,
    )

    arg(
        "-d",
        "--data_root",
        default="/home/aime3/Documents/Nail_Segmentation/dataset",
        type=str,
        help="Path to the dataset",
        required=False,
    )

    arg(
        "-w",
        "--work_dir",
        default="/home/aime3/Documents/Nail_Segmentation/work_dir",
        type=str,
        help="Path to the workdir",
        required=False,
    )

    arg(
        "-r",
        "--resume",
        default=False,
        type=bool,
        help="continue to training",
        required=False,
    )
    arg(
        "-b",
        "--debug",
        default=False,
        type=bool,
        help="debug code or not",
        required=False,
    )

    return parser.parse_args()


def read_image(image_path: str):
    image = cv2.imread(image_path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_mask(mask_path: str):
    mask = cv2.imread(mask_path, 0)
    return mask


def img2tensor(img: np.ndarray, dtype: np.dtype = np.float32):
    r"""
    Convert numpy image (RGB, BGR, Grayscale, SAR, Mask image, etc.) to tensor
    Args:
        image: A numpy array of [H,W,C] shape
        dtype: dtype of iamge that we want to obtain, defaut = np.float32
    Returns:
        Torch tensor of [C,H,W] or [H,W].
    """
    if len(img.shape) not in {2, 3}:
        raise ValueError(f"Image must have shape [H,W] or [H,W,C]. Got image with shape {img.shape}")
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False)).float()


def has_file_allowed_extension(filename: str, extensions: List[str]):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def getListOfFiles(dirName: str):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def object_from_dict(dict: DictConfig, parent: bool = None, **default_kwargs):
    """create an object from DictConfig

    Args:
        dict (DictConfig): a dict
        parent (bool, optional):  Defaults to None.

    Returns:
        _type_: object obtain from the DictConfig
    """
    kwargs = {}
    object_type = dict["type"]

    for name, value in dict.items():
        if name != "type":
            kwargs.setdefault(name, value)
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)
    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034

    return pydoc.locate(object_type)(**kwargs)


def get_all_items(path: str):
    images = []  # list of images
    path = str(path)
    folder = glob.glob(path + "*")
    while folder:
        f = folder[0]
        if os.path.isfile(f):
            images.append(f.replace(path, ""))
            folder.remove(f)
        else:
            folder += glob.glob(f + "/*")
            folder.remove(f)
    return images


def mask_to_inner_contour(mask: np.ndarray):
    mask = mask > 0.5
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), "reflect")
    contour = mask & (
        (pad[1:-1, 1:-1] != pad[:-2, 1:-1])
        | (pad[1:-1, 1:-1] != pad[2:, 1:-1])
        | (pad[1:-1, 1:-1] != pad[1:-1, :-2])
        | (pad[1:-1, 1:-1] != pad[1:-1, 2:])
    )
    return contour


def draw_contour_overlay(
    image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 1
):
    contour = mask_to_inner_contour(mask)
    if thickness == 1:
        image[contour] = color
    else:
        r = max(1, thickness // 2)
        for y, x in np.stack(np.where(contour)).T:
            cv2.circle(image, (x, y), r, color, lineType=cv2.LINE_4)
    return image


def mkdir(path: str):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_smooth_mask(mask: np.ndarray, open_kernel: int = 5, closing_kernel: int = 10):

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), np.uint8))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, np.ones((closing_kernel, closing_kernel), np.uint8))
    return closing


def get_contours(mask):

    mask = np.uint8(mask)
    # im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    try:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except Exception as e:
        imgs, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def plot_prediction(dict_prediction) -> None:

    for fname, pred in dict_prediction.items():

        image = cv2.imread(fname)
        mask = (pred[0][0]) * 255
        mask = get_smooth_mask(mask, 1, 2)

        plt.title("Nail Prediction")
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5)
        plt.pause(0.2)


def parallel(n_jobs: int, function: Any, list_items: List):
    Parallel(n_jobs=n_jobs)(delayed(function)(numpy_path) for numpy_path in tqdm(list_items, total=len(list_items)))
