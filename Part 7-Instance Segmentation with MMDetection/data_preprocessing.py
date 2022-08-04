import json
import numpy as np
from pycocotools import mask
from skimage import measure
from typing import List, Dict, Union, Tuple, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import mmcv
import json, itertools
from typing import Any
from joblib import Parallel, delayed
import json, itertools
from utils import read_image, read_mask, get_args, get_all_items, mkdir
import pandas as pd
import skimage.io as io
import os
import cv2
from skimage import measure
from pycocotools.coco import COCO
from utils import get_args


args = get_args()

data_root = args.data_root


def make_csv_file(args: Any) -> None:
    """The function to create a csv file that contains the information of the images and masks.

    Args:
        base_path (str): the folder path of the images and masks.
    """

    data_root = args.data_root
    list_images_train_masks = get_all_items(os.path.join(data_root, "train", "masks"))
    list_images_train_images = get_all_items(os.path.join(data_root, "train", "images"))

    list_images_train = [i for i in list_images_train_images if i in list_images_train_masks]

    print(len(list_images_train))

    list_images = []
    list_masks = []
    widths = []
    heights = []
    for path in list_images_train:
        binary_mask = read_mask(str(data_root) + "/train/masks" + path)
        size = binary_mask.shape
        widths.append(size[1])
        heights.append(size[0])
        list_images.append("train/images" + path)
        list_masks.append("train/masks" + path)

    dictionary = {
        "images": list_images,
        "masks": list_masks,
        "width": widths,
        "height": heights,
    }

    train_frame = pd.DataFrame(dictionary)

    list_images_valid_masks = get_all_items(os.path.join(data_root, "valid", "masks"))

    list_images = []
    list_masks = []
    widths = []
    heights = []
    for path in list_images_valid_masks:
        binary_mask = read_mask(str(data_root) + "/valid/masks" + path)
        size = binary_mask.shape
        widths.append(size[0])
        heights.append(size[1])
        list_images.append("valid/images" + path)
        list_masks.append("valid/masks" + path)

    dictionary = {
        "images": list_images,
        "masks": list_masks,
        "width": widths,
        "height": heights,
    }

    valid_frame = pd.DataFrame(dictionary)

    valid_frame["train"] = 0

    mkdir(f"{data_root}/csv_file")
    train_frame.to_csv(f"{data_root}/csv_file/train_info.csv", index=False)
    valid_frame.to_csv(f"{data_root}/csv_file/valid_info.csv", index=False)


def find_contours(mask: np.ndarray):
    ret, thresh_img = cv2.threshold(mask.astype(np.uint8), 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours((thresh_img).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_annotations(dataframe: pd.DataFrame):
    """get_annotations is to convert a dataframe into the coco format

    Args:
        train_df (pd.DataFrame): the dataframe that stored the infomation
        of the dataset. the form of the dataframe is
        images | width | height |

    Returns:
        [type]: the coco format data of the dataset
    """

    cats = [{"id": 0, "name": "nail"}]

    annotations = []
    images = []
    obj_count = 0

    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        filename = row.images

        images.append(
            {
                "id": idx,
                "file_name": filename,
                "width": row.width,
                "height": row.height,
            }
        )

        binary_mask = read_mask(os.path.join(str(data_root), row.masks))

        contours = find_contours(binary_mask)

        for contour in contours:
            xmin = int(np.min(contour[:, :, 0]))
            xmax = int(np.max(contour[:, :, 0]))
            ymin = int(np.min(contour[:, :, 1]))
            ymax = int(np.max(contour[:, :, 1]))

            poly = contour.flatten().tolist()
            poly = [x + 0.5 for x in poly]

            data_anno = {
                "image_id": idx,
                "id": obj_count,
                "category_id": 0,
                "bbox": [xmin, ymin, (xmax - xmin), (ymax - ymin)],
                "area": (xmax - xmin) * (ymax - ymin),
                "segmentation": [poly],
                "iscrowd": 0,
            }
            if (xmax - xmin) * (ymax - ymin) < 20:
                continue

            else:
                annotations.append(data_anno)

                obj_count += 1

    return {"categories": cats, "images": images, "annotations": annotations}


## run it on first three images for demonstration:
def get_json_coco(args) -> None:
    train_df = pd.read_csv(f"{data_root}/csv_file/train_info.csv")
    valid_df = pd.read_csv(f"{data_root}/csv_file/valid_info.csv")

    coco_json = os.path.join(data_root, "annotations")
    mkdir(coco_json)
    train_json = get_annotations(train_df)
    valid_json = get_annotations(valid_df)

    with open(f"{coco_json}/train.json", "w+", encoding="utf-8") as f:
        json.dump(train_json, f, ensure_ascii=True, indent=4)
    with open(f"{coco_json}/valid.json", "w+", encoding="utf-8") as f:
        json.dump(valid_json, f, ensure_ascii=True, indent=4)


def visualize_annotations(images_info: List[Dict], coco: Any):
    """visualize_annotations is to visualize the annotations of the dataset."""
    _, axs = plt.subplots(len(images_info), 2, figsize=(30, 15 * len(images_info)))
    for image_info, ax in zip(images_info, axs):
        image = io.imread(os.path.join(data_root, image_info["file_name"]))
        # print(image_info["file_name"])
        annIds = coco.getAnnIds(imgIds=[image_info["id"]])
        anns = coco.loadAnns(annIds)
        ax[0].imshow(image)
        ax[1].imshow(image)
        plt.sca(ax[1])
        coco.showAnns(anns, draw_bbox=True)
    plt.show()


if __name__ == "__main__":
    args = get_args()
    import json

    coco_json = os.path.join(data_root, "annotations")

    make_csv_file(args)
    get_json_coco(args)

    with open(f"{coco_json}/valid.json") as f:
        train_coco = json.load(f)
    images_info = train_coco["images"][87:89]
    annFile = f"{coco_json}/valid.json"
    coco = COCO(annFile)
    visualize_annotations(images_info, coco)
