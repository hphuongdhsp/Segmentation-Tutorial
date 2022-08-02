The goal of the reposiroty is to introduce some tools, libraries to train the deep learning models. We will go with following topics:

- Training deep learning models using Tensorflow platform
- How to use data balance in data pipeline loader
- Traing deep learning models using Pytorch Lightining
- Data augmentation on the GPU
- Speed up Dataloader by Using DALI
- Pytorch Lightning with Hugging Face
- Train an Intance Segmentation Model with MMDetection

Throgh all of topics, we will use the [Nail Segmentaion Dataset](https://www.kaggle.com/datasets/vpapenko/nails-segmentation). For the Nail Segmentation Dataset we will detail the semantic segmentation, we also tackle the instance segmentation problem in the last part with the awesome library `mmdetection`.


Let us summerize the main idea of each part.

------------------------------------------------
## Training deep learning models using Tensorflow platform

In this part we will introduce how to train a segmentation model by using the tensorflow platform. We will:

- Using `segmentation_models` library to get the segmentation model.

**Segmentation models** is python library with Neural Networks for Image Segmentation based on Keras (Tensorflow) framework. This is the high level API, you need only some lines of code to create a Segmentation Neural Network, for example 

```
import segmentation_models as sm

model = sm.Unet(
    "efficientnetb4",
    input_shape=(384, 384, 3),
    encoder_weights="imagenet",
    classes=1,
)
```

- Using `albumentation` library to do augmentation. 
- 
**Albumentations** is a Python library for fast and flexible image augmentations. Albumentations efficiently implements a rich variety of image transform operations that are optimized for performance, and does so while providing a concise, yet powerful image augmentation interface for different computer vision tasks, including object classification, segmentation, and detection.

- How to combine `albumentation` with the tf.data loader

The tf.data API enables you to build complex input pipelines from simple, reusable pieces. `Tf.data` API also supports the augumentation task with family of function `tf.image`: 
  -  tf.image.stateless_random_brightness
  -  tf.image.stateless_random_flip_left_right
  -  ...

This part we figure out how to combite the `tf.data` api and `albumentation`. 

- Using mixed-precision to improve the training speed
We introduce a simple tecnique that permit import the speed of training: `mixed precision`. Today, most models use the float32 dtype, which takes 32 bits of memory. However, there are two lower-precision dtypes, float16 and bfloat16, each which take 16 bits of memory instead. Modern accelerators can run operations faster in the 16-bit dtypes. We will introduce how to use  `mixed-precision` (mix float32 dtype and float16 dtype)
- Using `tensorboard` or `wandb` to visualize training in a ML project


------------------------------------------------
## How to use data balance in data pipeline loader

In the second part, we will introduce a simple way to deal with the imbalanced data. The tutorial deals with the segmentation problem, but you can apply this technique to the classification problem too. 

------------------------------------------------
## Traing deep learning models using Pytorch Lightining

In this part we will introduce how to use the Pytorch platform to train the segmentation model. In particular, we will use the Pytorch Lightning to speed up the training. 

We also introduce a segmentation library: `segmentation_models_pytorch` to build the segmentation model for the Pytorch platform.

------------------------------------------------
## Data augmentation on the GPU

This part will show how to perform augumentation on the GPU with `kornia` library. With this technique, we get a slight speed improvement.

------------------------------------------------
## Speed up Dataloader by Using DALI

To do image processing, we can also use the `DALI` library. The NVIDIA Data Loading Library (DALI) is a library for data loading and pre-processing to accelerate deep learning applications. It provides a collection of highly optimized building blocks for loading and processing image, video and audio data. It can be used as a portable drop-in replacement for built in data loaders and data iterators in popular deep learning frameworks. Data processing pipelines implemented using DALI are portable because they can easily be retargeted to TensorFlow, PyTorch, MXNet and PaddlePaddle.

In this part will detail how to combine DALI with Pytorch Lightning. 

------------------------------------------------
## Pytorch Lightning with Hugging Face

In the previous parts we used  the `segmentation_models_pytorch` to get the `segmentation model` for Pytorch platform. In this part, we will introduce very powerful library `transformers` that helps us easy to approach the state of the art of several tasks of the deep learning: NLP, Classification, ... 

We will introduce the SegFormer model use it to the semantic segmentation models.

------------------------------------------------
## Train an Intance Segmentation Model with MMDetection

The last part of the segmentation series, we will cover the intances segmentation model. We will recall some approaches for the instance segmentation problem. We also introduce a efficient library for the detection task `MMDetection`. For that, we will detail how to build the configuration for a training. 


