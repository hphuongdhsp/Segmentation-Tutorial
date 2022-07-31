from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from mmcv import Config, DictAction
from utils import get_args, getListOfFiles
import numpy as np
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
import torch
from mmcv.ops import RoIPool
from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from typing import List


img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]


class NailSegmentationInference:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda:0"):
        self.config_path = config_path
        self.config = mmcv.Config.fromfile(config_path)
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = init_detector(self.config_path, self.checkpoint_path, device=self.device)

        self.test_pipeline = Compose(replace_ImageToTensor(test_pipeline))

    def inference(self, imgs):
        """
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
               Either image files or loaded images.

        Returns:
            If imgs is a list or tuple
        """
        if isinstance(imgs, (list, tuple)):
            is_batch = True
        else:
            imgs = [imgs]
            is_batch = False
        test_pipeline = self.test_pipeline
        if isinstance(imgs[0], np.ndarray):
            # set loading pipeline type
            test_pipeline.type = "LoadImageFromWebcam"
        datas = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                # directly add img
                data = dict(img=img)
            else:
                # add information into dict
                data = dict(img_info=dict(filename=img), img_prefix=None)
            # build the data pipeline
            data = test_pipeline(data)
            datas.append(data)

        data = collate(datas, samples_per_gpu=len(imgs))
        # just get the actual data from DataContainer
        data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
        data["img"] = [img.data[0] for img in data["img"]]
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [self.device])[0]
        else:
            for m in self.model.modules():
                assert not isinstance(m, RoIPool), "CPU inference with RoIPool is not supported currently."

        # forward the model
        with torch.no_grad():
            results = self.model(return_loss=False, rescale=True, **data)

        if not is_batch:
            return results[0]
        else:
            return results

    def show_result_pyplot(
        self, imgs, result, score_thr=0.5, title="result", wait_time=0.5, palette=None, out_file=None
    ):
        """Visualize the detection results on the image.
        Args:
            model (nn.Module): The loaded detector.
            img (str or np.ndarray): Image filename or loaded image.
            result (tuple[list] or list): The detection result, can be either
                (bbox, segm) or just bbox.
            score_thr (float): The threshold to visualize the bboxes and masks.
            title (str): Title of the pyplot figure.
            wait_time (float): Value of waitKey param. Default: 0.
            palette (str or tuple(int) or :obj:`Color`): Color.
                The tuple of color should be in BGR order.
            out_file (str or None): The path to write the image.
                Default: None.
        """
        if hasattr(self.model, "module"):
            self.model = self.model.module
        self.model.show_result(
            imgs,
            result,
            score_thr=score_thr,
            show=True,
            wait_time=wait_time,
            win_name=title,
            bbox_color=palette,
            text_color=(200, 200, 200),
            mask_color=palette,
            out_file=out_file,
        )


if __name__ == "__main__":
    config_path = "./configs/details_config.py"
    checkpoint_path = "/home/aime3/Documents/Nail_Segmentation/work_dir/Intances Sementation/latest.pth"
    device = "cuda:0"

    API = NailSegmentationInference(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
    # images = "/home/aime3/Segmentation-Tutorial/Instance Segmentation with MMDetection/imgs/images/170517_1_R_1.png"

    images = getListOfFiles(
        "/home/aime3/Segmentation-Tutorial/Instance Segmentation with MMDetection/imgs/results/images"
    )
    results = API.inference(images)
    for (i, image) in enumerate(images):
        API.show_result_pyplot(image, results[i])
