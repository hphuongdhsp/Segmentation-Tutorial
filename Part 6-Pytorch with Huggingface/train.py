import os
from pytorch_lightning.loggers import WandbLogger
from typing import Any
from dataset import NailSegmentation
from models import FineTuneCB, LitNailSegmentation, get_model, SegFormer
from utils import get_args, mkdir, object_from_dict
import yaml
from addict import Dict as Adict


def main(args: Any) -> None:

    # GET MODEL + DATALOADER
    data_root = args.data_root
    work_dir = args.work_dir
    with open(args.config_path) as file:
        config = Adict(yaml.load(file, Loader=yaml.SafeLoader))
    batch_size = config.training.batch_size
    workers = config.training.workers

    csv_path = f"{data_root}/csv_file"

    # GET MODEL + DATALOADER
    if config.model.type == "transformers":

        model = SegFormer(config.model.encoder_name, config.model.size, config.model.classes)
    else:
        model = get_model(config)

    datamodule = NailSegmentation(
        data_root=data_root,
        csv_path=csv_path,
        test_path="",
        batch_size=batch_size,
        num_workers=workers,
    )

    model_lighning = LitNailSegmentation(model=model, learning_rate=config.training.learning_rate)
    #  CALLBACK
    log_dir = os.path.join(work_dir, config.experiment_name, "logs")
    weight_dir = os.path.join(work_dir, config.experiment_name, "weight")
    mkdir(log_dir)
    mkdir(weight_dir)
    logger = WandbLogger(project="Nail Segmentation", save_dir=log_dir)
    finetune = FineTuneCB(unfreeze_epoch=config.training.freeze_epochs)
    # CKPT
    ckpt = object_from_dict(config.checkpoint_callback, dirpath=weight_dir)

    ckpt_path = os.path.join(weight_dir, config.checkpoint_callback.filename + ".ckpt")
    # DEBUG
    if config.debug is True:
        limit_train_batches = 0.01
    else:
        limit_train_batches = 1.0
    # TRAINER
    trainer = object_from_dict(
        config.trainer, logger=logger, callbacks=[ckpt, finetune], limit_train_batches=limit_train_batches
    )
    #  RESUME
    if config.resume is False:
        trainer.fit(model=model_lighning, datamodule=datamodule)
    else:
        trainer.fit(
            model=model_lighning,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )


if __name__ == "__main__":
    args = get_args()
    main(args)
