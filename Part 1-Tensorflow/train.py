import gc
import os
from datetime import datetime

import segmentation_models as sm
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint

from dataset import load_data_path, tf_dataset
from transform import train_transform, valid_transform
from utils import get_args, mkdir

sm.set_framework("tf.keras")
sm.framework()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


def run(args):
    # setting gpu (device)
    if len(args.device) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[0])
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # set mixed_precision
    if args.mixed_precision:
        tf.keras.mixed_precision.experimental.set_policy("mixed_float16")
        dtype = tf.float16
    else:
        dtype = tf.float32

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.keras.backend.clear_session()
    gc.collect()

    # Define callback
    work_dir = args.work_dir
    file_path = f"{work_dir}/tensorflow/weights/best-weight_effnetb4.h5"

    callbacks = [
        ModelCheckpoint(filepath=file_path, save_best_only=True, monitor="val_loss"),
    ]

    if args.log == "wandb":
        import wandb
        from wandb.keras import WandbCallback
        logdir = f"{work_dir}/tensorflow/logs/wandb"
        mkdir(logdir)
        wandb.init(project="Segmentation by Tensorflow", dir=logdir)

        wandb.config = {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        }
        callbacks.append(WandbCallback())
    if args.log == "tensorboard":
        logdir = f"{work_dir}/tensorflow/logs/tensorboard/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        callbacks.append(tensorboard_callback)
    else:
        pass

    data_root = str(args.data_root)
    train_csv_dir = f"{data_root}/csv_file/train.csv"
    valid_csv_dir = f"{data_root}/csv_file/valid.csv"
    # set batch_size
    batch_size = args.batch_size
    epochs = args.epochs

    # get training and validation set
    train_dataset = load_data_path(data_root, train_csv_dir, "train")
    train_loader = tf_dataset(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        transforms=train_transform(),
        dtype=dtype,
        device=args.device,
    )
    valid_dataset = load_data_path(data_root, valid_csv_dir, "valid")
    valid_loader = tf_dataset(
        dataset=valid_dataset,
        shuffle=False,
        batch_size=batch_size,
        transforms=valid_transform(),
        dtype=dtype,
        device=args.device,
    )

    # get model
    def create_model():

        model = sm.Unet(
            "efficientnetb4",
            input_shape=(384, 384, 3),
            encoder_weights="imagenet",
            classes=1,
        )
        # TO USE mixed_precision, HERE WE USE SMALL TRICK, REMOVE THE LAST LAYER AND ADD
        # THE ACTIVATION SIGMOID WITH THE DTYPE  TF.FLOAT32
        last_layer = tf.keras.layers.Activation(activation="sigmoid", dtype=tf.float32)(model.layers[-2].output)
        model = tf.keras.Model(model.input, last_layer)

        # define optimization, here we use the tensorflow addon, but use can also use some normal \
        # optimazation that is defined in tensorflow.optimizers
        optimizer = tfa.optimizers.RectifiedAdam()

        if args.mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
        # define a loss fucntion
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss()
        total_loss = dice_loss + focal_loss
        # define metric
        metrics = [
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),
        ]
        # compile model with optimizer, losses and metrics
        model.compile(optimizer, total_loss, metrics)
        return model

    if len(args.device) == 1:
        model = create_model()
    else:
        devices = [f"/gpu:{args.device[i]}" for i in args.device]
        print(f"Devices:{devices}")
        strategy = tf.distribute.MirroredStrategy(devices=devices)
        with strategy.scope():
            model = create_model()

    total_steps = int(len(train_dataset[0]) / batch_size) + 1

    history = model.fit(
        train_loader,
        steps_per_epoch=total_steps,
        epochs=epochs,
        validation_data=valid_loader,
        callbacks=callbacks,
    )
    return history


if __name__ == "__main__":
    args = get_args()
    history = run(args)
