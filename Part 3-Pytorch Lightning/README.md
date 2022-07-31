# Nail Segmentation



## Installation 


## First: Install the Python environment using Conda



To install `conda`, follow instructions at https://conda.io/projects/conda/en/latest/user-guide/install/linux.html

Note that you will likely need to close and re-open your terminal.
Afterward, you should have ability to run the `conda` command in your terminal.

Run 
```
make conda-update
``` 
to create an environment called `nail`, as defined in `environment.yml`.
This environment will provide us with the right Python version as well as the CUDA and CUDNN libraries.

If you edit `environment.yml`, just run  `make conda-update`
again to get the latest changes.

Next, activate the conda environment.

```sh
conda activate nail
```

**IMPORTANT**: every time you work in this directory, make sure to start your session with `conda activate nail`.

#### Next: install Python packages

Next, install all necessary Python packages by running `make pip-tools`

Using `pip-tools` lets us do three nice things:

1. Separate out dev from production dependencies (`requirements-dev.in` vs `requirements.in`).
2. Have a lockfile of exact versions for all dependencies (the auto-generated `requirements-dev.txt` and `requirements.txt`).
3. Allow us to easily deploy to targets that may not support the `conda` environment.

If you add, remove, or need to update versions of some requirements, edit the `.in` files, and simply run `make pip-tools` again.


## Dataset 
Download the data from [link](https://storage.torus.lan/sharing/O4eZXKrxJ) and put it in `basepath`. 

The dataset with be stored at the "basepath", and were organized as the following

```
├── base_path 
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
  fold: 2
  max_epochs: 300
  mixed_precision: True
  device: "cuda:0"
  freeze_epochs: 1
  learning_rate: 0.001
  batch_size: 16
  workers: 4

testing:
  fold: 2
  phase: "test"
  batch_size: 16
  tta: True
  mixed_precision: False
  threshold: 0.39
  device: "cuda:0"

```
To continue, we first change the configuration that is suitable for the local machine. For configuration of local machine, we recommend to change 

+ batch_size
+ device

## Data preparation 

We will

```
python data_preparation.py --base_path base_path
```

In this step we do: 
- Change all image to png to make sure that all images in images folder and masks folder having the same tail

- Make the `train.csv` and `valid.csv` files and stores them in the `.csvfile` folder.

## Training model 

We do 

```
python train.py --base_path base_path --weight_path weight_path --config_path config_path --resume False
```

where 
+ the base_path is the folder that you stores the data 
+ weight_path is the path of folder that you want to save weights
+ config_path is the path of config file, in this repository: config_path = `./configs/config.yaml`


These parameters is saved as the function `get_args` in `utils`

To simplicity, we use command:

```
python train.py --base_path base_path --weight_path ./weights --config_path ./configs/config.yaml --resume False
```

## CI manually 

To do CI manually, we use command

```
make lint
```

Remark to use onnx tensor rtx
```
import onnx
import onnx_tensorrt.backend as backend
import numpy as np

model = onnx.load("/path/to/model.onnx")
engine = backend.prepare(model, device='CUDA:1')
input_data = np.random.random(size=(32, 3, 224, 224)).astype(np.float32)
output_data = engine.run(input_data)[0]
print(output_data)
print(output_data.shape)
```