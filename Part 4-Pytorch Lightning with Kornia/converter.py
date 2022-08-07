from collections import defaultdict
from pathlib import Path
import os
import torch
import torch.onnx
import yaml
from addict import Dict
from addict import Dict as Addict
from albumentations import *
import yaml
from addict import Dict as Adict
from typing import Any
from models import LitNailSegmentation, get_model
from utils import get_args

# Function to Convert to ONNX


def Convert_ONNX(args: Any):
    """The function is to convert the model to ONNX format.

    Args:
        cfg (Addict): the configuration of the model.
    """
    work_dir = args.work_dir
    with open(args.config_path) as file:
        config = Adict(yaml.load(file, Loader=yaml.SafeLoader))
    model = get_model(config).to(config.testing.device)
    model_lighning = LitNailSegmentation(model=model, learning_rate=config.training.learning_rate)

    weight_dir = os.path.join(work_dir, config.experiment_name, "weight")
    ckpt_path = os.path.join(weight_dir, config.checkpoint_callback.filename + ".ckpt")
    onnx_path = os.path.join(weight_dir, config.checkpoint_callback.filename + ".onnx")

    model_lighning = model_lighning.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        map_location=config.testing.device,
    )

    model_lighning.eval()
    # Let's create a dummy input tensor
    dummy_input = torch.randn(
        config.testing.batch_size,
        3,
        config.data.input_size,
        config.data.input_size,
        requires_grad=True,
        device=config.testing.device,
    )
    # Export the model
    torch.onnx.export(
        model_lighning,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        onnx_path,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["image"],  # the model's input names
        output_names=["mask"],  # the model's output names
        dynamic_axes={
            "image": {0: "batch_size"},  # variable length axes
            "label": {0: "batch_size"},
        },
    )
    print(" ")
    print("Model has been converted to ONNX")


if __name__ == "__main__":
    args = get_args()
    Convert_ONNX(args)
