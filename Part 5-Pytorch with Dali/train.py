import os
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from dataset import NailSegmentationDaliDali
from models import FineTuneCB, LitNailSegmentation, get_model
from utils import get_args, mkdir, object_from_dict
from addict import Dict as Adict
import yaml


def main(args: Any) -> None:

    # GET MODEL + DATALOADER
    data_root_npy = args.data_root_npy
    work_dir = args.work_dir
    with open(args.config_path) as file:
        config = Adict(yaml.load(file, Loader=yaml.SafeLoader))
    dataloader_type = config.dataloader_type
    batch_size = config.training.batch_size
    csv_folder = f"{data_root_npy}/csv_file"
    print(f"Using dataloader:  {dataloader_type}")
    model = get_model(config)
    datamodule = datamodule = NailSegmentationDaliDali(
        data_root_npy=data_root_npy, batch_size=batch_size, csv_folder=csv_folder
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

if __name__ == "__main__":
    main()
