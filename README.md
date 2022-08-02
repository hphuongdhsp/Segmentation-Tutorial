The goal of the reposiroty is to introduce some tools, libraries to train the deep learning models. We will go with following topics:

- Training deep learning models using Tensorflow platform
- How to use data balance in data pipeline loader
- Traing deep learning models using Pytorch Lightining
- Data augmentation on the GPU
- Speed up Dataloader by Using DALI
- Pytorch Lightning with Hugging Face
- Train an Intance Segmentation Model with MMDetection

Throgh all of topics, we will use the [Nail Segmentaion Dataset](https://www.kaggle.com/datasets/vpapenko/nails-segmentation). For the Nail Segmentation Dataset we will detail the semantic segmentation, we also tackle the instance segmentation problem in the last part with the awesome library `mmdetection`.


---------------------------------

Let us summerize the main idea of each part.

------------------------------------------------
## Training deep learning models using Tensorflow platform

In this part we will introduce how to train a segmentation model by using the tensorflow platform. We will through:

- Using `segmentation_models` library to get the segmentation model
- Using `Albumentation` library to do augmentation. 
- How to use the `albumentation` in the tf.data loader
- Using mixed-precision to improve the training speed
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


