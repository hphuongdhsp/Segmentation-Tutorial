import os
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.onnx
import tqdm
from albumentations import *
from torch.utils.data import DataLoader, Dataset

from transform import valid_transform
from utils import getListOfFiles, plot_prediction

FILE_PATH = str(Path(__file__).parent.resolve())


def sigmoid_function(z):
    """this function implements the sigmoid function, and
    expects a numpy array as argument"""

    if isinstance(z, np.ndarray):
        return 1.0 / (1.0 + np.exp(-z))
    else:
        return None


def get_image(image_or_path):
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
    def __init__(self, list_image, transform=valid_transform()):
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


class NailSegmentationOnnx:
    def __init__(self, TTA=False, batch_size=16, threshold=0.5):

        """
        args:
            + device: "cuda:0", "cuda", "cpu", defaut = 'cuda:0'

            + Using test time augument or not, defaut = True
            + half: Using fn16 or not, defaut = half
        """
        self.threshold = threshold
        self.batch_size = batch_size
        self.ort_session = ort.InferenceSession(
            os.path.join(FILE_PATH, "models/timm-efficientnet-b4.onnx"), providers=providers
        )
        print(self.ort_session.get_providers())
        onnx.checker.check_model(onnx.load(os.path.join(FILE_PATH, "models/timm-efficientnet-b4.onnx")))
        self.TTA = TTA
        self.transform = valid_transform(p=1)

    def preprocess_batch_data(self, list_images):
        dataset = InferenceDataset(list_images, transform=self.transform)
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=dataset.collate,
        )
        return loader

    def predict_by_batch(self, list_image):
        loader = self.preprocess_batch_data(list_image)
        pred_masks = []
        for images, infos in tqdm.tqdm(loader):
            # print(images)
            ort_inputs = {"images": images}
            ort_inputs = self.ort_session.run(None, ort_inputs)
            logits = ort_inputs[0]
            preds = sigmoid_function(logits)
            if self.TTA:
                flips = [[-1]]
                for aug in flips:

                    ort_inputs = {"images": np.flip(images, aug)}
                    ort_inputs = self.ort_session.run(None, ort_inputs)
                    logits = ort_inputs[0]
                    logits = np.flip(logits, aug)
                    preds += sigmoid_function(logits)

                preds /= 1 + len(flips)

            preds = (np.transpose(preds, (0, 2, 3, 1)) > self.threshold) * 1

            for (pred, info) in zip(preds, infos):
                fname = info["img_path"]
                shape = info["img_size"]

                pred = cv2.resize(pred.astype("float32"), (shape[1], shape[0]))

                pred_masks[fname].append([pred])

        return pred_masks


def test_api_onnx(images_path: str) -> None:
    r"""This function is to test the Human Part Segmentatation api"""

    BDP = NailSegmentationOnnx(TTA=False, batch_size=16)
    images_list = getListOfFiles(images_path)
    print(images_list)

    dict_prediction = BDP.predict_by_batch(images_list)
    plot_prediction(dict_prediction)


if __name__ == "__main__":
    test_api_onnx("./imgs")
