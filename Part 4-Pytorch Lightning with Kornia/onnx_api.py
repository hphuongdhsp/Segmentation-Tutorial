import os
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.onnx

from typing import Union, List
from torch.utils.data import Dataset
import yaml
from addict import Dict as Adict
from pathlib import Path
from transform import valid_transform
from utils import getListOfFiles, plot_prediction, get_args

args = get_args()
data_root = args.data_root
work_dir = args.work_dir
with open(args.config_path) as file:
    config = Adict(yaml.load(file, Loader=yaml.SafeLoader))

providers = [
    (
        "CUDAExecutionProvider",
        {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        },
    )
]


def sigmoid_function(z: np.ndarray):
    """this function implements the sigmoid function, and
    expects a numpy array as argument"""

    if isinstance(z, np.ndarray):
        return 1.0 / (1.0 + np.exp(-z))
    else:
        return None


def get_image(image_or_path: Union[np.ndarray, Path, str]):
    """Reads an image from file or array/tensor and converts it to RGB (H,W,3).
    Arguments:
        tensor {Sstring, numpy.array or torch.tensor} -- [the input image or path to it]
    """
    if isinstance(image_or_path, str):
        try:
            image = cv2.cvtColor(cv2.imread(image_or_path), cv2.COLOR_BGR2RGB)
        except OSError:
            print("error opening file :: ", image_or_path)
            return None
    elif isinstance(image_or_path, torch.Tensor):
        image = image_or_path.detach().cpu().numpy()
    else:
        image = image_or_path

    return image


class InferenceDataset(Dataset):
    def __init__(self, list_image: List[Union[str, np.ndarray]], transform=valid_transform()):
        self.dataset = list_image
        self.transform = transform

    def __getitem__(self, index):
        image_or_path = self.dataset[index]
        if isinstance(image_or_path, str):
            img = get_image(image_or_path)
            info = {"img_size": img.shape, "img_path": image_or_path}

        else:
            img = cv2.cvtColor(image_or_path, cv2.COLOR_BGR2RGB)
            info = {"img_size": img.shape, "img_path": "None"}
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, info

    @staticmethod
    def collate(batch):
        images, infos = [], []
        for i, (img, info) in enumerate(batch):
            images.append(np.transpose(img, (2, 0, 1)))
            infos.append(info)
        return np.stack(images), infos

    def __len__(self):
        return len(self.dataset)


class NailSegmentationOnnx:
    def __init__(self, onnx_path: str, TTA: bool = False, batch_size: int = 16, threshold: float = 0.5):

        """
        args:
            + device: "cuda:0", "cuda", "cpu", defaut = 'cuda:0'

            + Using test time augument or not, defaut = True
            + half: Using fn16 or not, defaut = half
        """
        self.threshold = threshold
        self.batch_size = batch_size
        self.ort_session = ort.InferenceSession(os.path.join(onnx_path), providers=providers)
        print(self.ort_session.get_providers())
        onnx.checker.check_model(onnx.load(os.path.join(onnx_path)))
        self.TTA = TTA
        self.transform = valid_transform(p=1)

    def preprocess_batch_data(self, list_images):
        """The function to preprocess the batch data

        Args:
            list_images List[Union[str, np.ndarray]]: List of images to preprocess

        Returns:
            dataset: dataset of preprocessed images
        """

        dataset = InferenceDataset(list_images, transform=self.transform)
        return dataset

    def predict_by_batch(self, list_image):
        loader = self.preprocess_batch_data(list_image)
        pred_masks = []
        images, infos = loader.collate([loader[i] for i in range(len(loader))])

        ort_inputs = {"image": images}
        ort_inputs = self.ort_session.run(None, ort_inputs)
        logits = ort_inputs[0]
        preds = sigmoid_function(logits)
        if self.TTA:
            flips = [[-1]]
            for aug in flips:

                ort_inputs = {"image": np.flip(images, aug)}
                ort_inputs = self.ort_session.run(None, ort_inputs)
                logits = ort_inputs[0]
                logits = np.flip(logits, aug)
                preds += sigmoid_function(logits)

            preds /= 1 + len(flips)

        ohe_labels = (np.transpose(preds, (0, 2, 3, 1)) > self.threshold) * 1

        for (pred, info) in zip(ohe_labels, infos):
            fname = info["img_path"]
            shape = info["img_size"]

            pred = cv2.resize(pred.astype("float32"), (shape[1], shape[0]))

            pred_masks.append(pred.astype(np.uint8))

        return pred_masks


def test_api_onnx(images_path: str) -> None:
    r"""This function is to test the Human Part Segmentatation api"""

    args = get_args()
    work_dir = args.work_dir
    with open(args.config_path) as file:
        config = Adict(yaml.load(file, Loader=yaml.SafeLoader))
    weight_dir = os.path.join(work_dir, config.experiment_name, "weight")
    onnx_path = os.path.join(weight_dir, config.checkpoint_callback.filename + ".onnx")

    BDP = NailSegmentationOnnx(onnx_path=onnx_path, TTA=False, batch_size=16)
    images_list = getListOfFiles(images_path)
    print(images_list)

    list_preds = BDP.predict_by_batch(images_list)
    plot_prediction(list_preds, images_list)


if __name__ == "__main__":
    test_api_onnx("./imgs")
