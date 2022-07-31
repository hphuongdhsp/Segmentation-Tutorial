from collections import defaultdict
from pathlib import Path

import torch
import torch.onnx
import yaml
from addict import Dict
from addict import Dict as Addict
from albumentations import *

from models import LitNailSegmentation, get_model
from utils import get_args, mkdir

# Function to Convert to ONNX


def Convert_ONNX(cfg: Addict):

    model = get_model(cfg).to(cfg.testing.device)

    model_lighning = LitNailSegmentation(model=model, learning_rate=cfg.training.learning_rate)

    model_lighning = model_lighning.load_from_checkpoint(
        checkpoint_path="models/best_checkpoint.ckpt",
        map_location=cfg.testing.device,
    )

    model_lighning.eval()
    # Let's create a dummy input tensor
    dummy_input = torch.randn(
        cfg.testing.batch_size,
        3,
        cfg.data.input_size,
        cfg.data.input_size,
        requires_grad=True,
        device=cfg.testing.device,
    )
    # Export the model
    torch.onnx.export(
        model_lighning,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        "models/" + cfg.model.encoder_name + ".onnx",  # where to save the model
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
    with open(args.config_path) as file:
        cfg = Dict(yaml.load(file, Loader=yaml.SafeLoader))
    Convert_ONNX(cfg)
