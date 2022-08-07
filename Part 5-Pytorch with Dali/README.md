The source code of post [Segmentation Model-Part V - Data augmentation on the GPU with DALI](https://hphuongdhsp.github.io/ml-blog/pytorchlightning/semanticsegmentation/deeplearning/dali/2022/08/06/segmentation-model-part5.html)
## Installation 
## First: Install the Python environment using Conda



To install `conda`, follow instructions at https://conda.io/projects/conda/en/latest/user-guide/install/linux.html

Note that you will likely need to close and re-open your terminal.
Afterward, you should have ability to run the `conda` command in your terminal.

Run 
```sh
make conda-update
``` 
to create an environment called `nail`, as defined in `environment.yml`.
This environment will provide us with the right Python version as well as the CUDA and CUDNN libraries.

If you edit `environment.yml`, just run  `make conda-update`
again to get the latest changes.

Next, activate the conda environment.

```sh
conda activate segmentation-pt
```

> Important: every time you work in this directory, make sure to start your session with `conda activate segmentation-pt`.
#### Next: install Python packages

Next, install all necessary Python packages by running `make pip-tools`

Using `pip-tools` lets us do three nice things:

1. Separate out dev from production dependencies (`requirements-dev.in` vs `requirements.in`).
2. Have a lockfile of exact versions for all dependencies (the auto-generated `requirements-dev.txt` and `requirements.txt`).
3. Allow us to easily deploy to targets that may not support the `conda` environment.

If you add, remove, or need to update versions of some requirements, edit the `.in` files, and simply run `make pip-tools` again.


## Dataset 
Download the data from [link](https://drive.google.com/file/d/1qBLwdQeu9nvTw70E46XNXMciB0aKsM7r/view) and put it in `data_root`. 

The dataset with be stored at the "data_root", and were organized as the following

```bash
├── data_root 
│   ├── train
│   │   ├── images
│   │   ├── masks
│   ├── valid 
│   │   ├── images
│   │   ├── masks

```

## Configuration 


The configuration is stored at "./configs/config.yaml"

The `config` file is organized as the following

```

data:
  dataset: Human Part Segmentation
  input_size: 384
  csv_path: "./csv_file"

model:
  type: segmentation_models_pytorch.Unet
  encoder_name: timm-efficientnet-b4
  encoder_weights: "noisy-student"
  in_channels: 3
  classes: 1

training:
  seed: 42
  freeze_epochs: 1
  learning_rate: 0.001
  batch_size: 32
  workers: 4


checkpoint_callback:
  type: pytorch_lightning.callbacks.ModelCheckpoint
  filename: "timm-efficientnet-b4"
  monitor: valid_loss
  verbose: True
  save_top_k: 1
  save_last: True
  mode: min

trainer:
  type: pytorch_lightning.Trainer
  gpus: [0]
  max_epochs: 300
  benchmark: True
  precision: 16
  progress_bar_refresh_rate: 1
  gradient_clip_val: 5.0
  auto_lr_find: True
  weights_summary: "top"
  limit_train_batches: 1.0

```

Here: 
- model: the model parameters
- training: dataset and model module parameters
- checkpoint callback parameters
- trainer: trainer parameters


## Data preparation 

We use

```sh
python data_preparation.py --data_root data_root --work_dir work_dir --config_path configs/comfig.yaml
```

In this step we do: 
- Change all image to png to make sure that all images in images folder and masks folder having the same tail

- Make the `train.csv` and `valid.csv` files and stores them in the `.csvfile` folder.

## Training model 

We do 
```
python train.py --data_root data_root --work_dir work_dir --config_path configs/comfig.yaml --resume False
```

where 
+ the data_root is the folder that you stores the data 
+ work_dir is the path of the logs and weights
+ config_path is the path of config file, in this repository: config_path = `./configs/config.yaml`
