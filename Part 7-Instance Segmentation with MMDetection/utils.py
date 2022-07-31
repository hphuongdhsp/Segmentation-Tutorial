import argparse
import errno
import glob
import os
import pydoc
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from mmcv import Config, DictAction
import warnings


IMG_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def get_args() -> Any:
    parser = argparse.ArgumentParser(description="Train Nail detector")
    parser.add_argument(
        "--config",
        default="./configs/details_config.py",
        help="train config file path",
        required=False,
    )

    parser.add_argument(
        "-d",
        "--data_root",
        default="/home/aime3/Documents/Nail_Segmentation/dataset",
        type=str,
        help="Path to the dataset",
        required=False,
    )

    parser.add_argument(
        "-w",
        "--work_dir",
        default="/home/aime3/Documents/Nail_Segmentation/work_dir",
        type=str,
        help="the dir to save logs and models",
        required=False,
    )
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument("--auto-resume", action="store_true", help="resume from the latest checkpoint automatically")
    parser.add_argument(
        "--no-validate", action="store_true", help="whether not to evaluate the checkpoint during training"
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="(Deprecated, please use --gpu-id) number of gpus to use "
        "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="(Deprecated, please use --gpu-id) ids of gpus to use " "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-id", type=int, default=0, help="id of gpu to use " "(only applicable to non-distributed training)"
    )

    parser.add_argument(
        "-r",
        "--resume",
        default=False,
        type=bool,
        help="continue to training",
        required=False,
    )
    parser.add_argument(
        "-b",
        "--debug",
        default=False,
        type=bool,
        help="debug code or not",
        required=False,
    )

    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--diff-seed", action="store_true", help="Whether or not set different seeds for different ranks"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="whether to set deterministic options for CUDNN backend."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--auto-scale-lr", action="store_true", help="enable automatically scaling LR.")
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both "
            "specified, --options is deprecated in favor of --cfg-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --cfg-options")
        args.cfg_options = args.options

    return parser.parse_args()


def get_args_valid() -> Any:
    parser = argparse.ArgumentParser(description="Train Nail detector")
    parser.add_argument("config", help="train config file path")

    parser.add_argument(
        "-d",
        "--data_root",
        default="/home/aime3/Documents/Nail_Segmentation/dataset",
        type=str,
        help="Path to the dataset",
        required=False,
    )

    parser.add_argument(
        "-w",
        "--work_dir",
        default="/home/aime3/Documents/Nail_Segmentation/work_dir",
        type=str,
        help="the dir to save logs and models",
        required=False,
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="/home/aime3/Documents/Nail_Segmentation/work_dir/Intances Sementation/",
        type=str,
        help="checkpoint file",
        required=False,
    )
    parser.add_argument("--work-dir", help="the directory to save the file containing evaluation metrics")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase" "the inference speed",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="(Deprecated, please use --gpu-id) ids of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="id of gpu to use " "(only applicable to non-distributed testing)"
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where painted images will be saved")
    parser.add_argument("--show-score-thr", type=float, default=0.3, help="score threshold (default: 0.3)")
    parser.add_argument("--gpu-collect", action="store_true", help="whether to use gpu to collect results.")
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both "
            "specified, --options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


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


def read_image(image_path: str):
    image = cv2.imread(image_path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_mask(mask_path: str):
    mask = cv2.imread(mask_path, 0)
    return mask


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
