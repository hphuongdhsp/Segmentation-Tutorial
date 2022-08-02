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
# 1. Training deep learning models using Tensorflow platform

In this part we will introduce how to train a segmentation model by using the tensorflow platform. We will:

### Using `segmentation_models` library to get the segmentation model.

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

### Using `albumentation` library to do augmentation. 
**Albumentations** is a Python library for fast and flexible image augmentations. Albumentations efficiently implements a rich variety of image transform operations that are optimized for performance, and does so while providing a concise, yet powerful image augmentation interface for different computer vision tasks, including object classification, segmentation, and detection.

- How to combine `albumentation` with the tf.data loader

The tf.data API enables you to build complex input pipelines from simple, reusable pieces. `Tf.data` API also supports the augumentation task with family of function `tf.image`: 
  -  tf.image.stateless_random_brightness
  -  tf.image.stateless_random_flip_left_right
  -  ...

This part we figure out how to combite the `tf.data` api and `albumentation`. 

### Using mixed-precision to improve the training speed
We introduce a simple tecnique that permit import the speed of training: `mixed precision`. Today, most models use the float32 dtype, which takes 32 bits of memory. However, there are two lower-precision dtypes, float16 and bfloat16, each which take 16 bits of memory instead. Modern accelerators can run operations faster in the 16-bit dtypes. We will introduce how to use  `mixed-precision` (mix float32 dtype and float16 dtype) to accelerate the training task. 
### Using `tensorboard` or `wandb` to visualize training in a ML project
We also introduce some tools to monitori the evaluation of training task. 

------------------------------------------------
# 2. How to use data balance in data pipeline loader

Base on the pipeline of the previous part, this part will introduce how to deal with a imbalanced data. 

A classification data set with skewed class proportions is called imbalanced. Classes that make up a large proportion of the data set are called majority classes. Those that make up a smaller proportion are minority classes. 

<!-- ![](./imgs/imbalanced data.png) -->

![](https://habrastorage.org/webt/zg/xl/wb/zgxlwbdpbxtq57gzdqscq3tubws.png)

For the nail segmentation data, we assume that the data is collected from various sources (folders). Those sources have a diffence of quality (skewed) and a difference quantity (maybe), or a difference types. Then we can also do balancing when loading the data, this may boost the performance of the model. 

The Tutorial deals with the segmentation problem, but with a slight modifying, it also works with the classification problem. 


------------------------------------------------
# 3. Traing deep learning models using Pytorch Lightining

In this part we will introduce how to use the Pytorch platform to train the segmentation model. In particular, we will use the Pytorch Lightning to train the model. 

We also introduce a segmentation library: `segmentation_models_pytorch` to build the segmentation model for the Pytorch platform.

Similar to the `segmentation_models`, `segmentation_models_pytorch` is a high level API, it helps us build a sementic segmentation model with only some lines of code. 

```
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)
```

By using Pytorch Lightning, we can easy implement new SOTA methods of training with some lines of code. 

------------------------------------------------
# 4. Data augmentation on the GPU
**GPUs Are Fast! Datasets Are Your Bottleneck.** The fact is that  today these transforms are applied one input at a time on `CPUs`. This means that they are super slow. If your models allows you to, you could apply the same transforms to a batch of data at once on GPUs.

`Kornia` is a library that helps you do the augumentations in GPU memory. 

![](https://habrastorage.org/webt/ew/tc/u7/ewtcu7o4gjurdfii2tx8ghmrkbq.jpeg)

This part will show how to perform augumentation on the GPU with `kornia` library.
------------------------------------------------
# 5. Speed up Dataloader by Using DALI

In this part we will `DALI` library. The NVIDIA Data Loading Library (DALI) is a library for data loading and pre-processing to accelerate deep learning applications. It provides a collection of highly optimized building blocks for loading and processing image, video and audio data. It can be used as a portable drop-in replacement for built in data loaders and data iterators in popular deep learning frameworks. Data processing pipelines implemented using DALI are portable because they can easily be retargeted to TensorFlow, PyTorch, MXNet and PaddlePaddle.


![](https://habrastorage.org/webt/7g/9t/dr/7g9tdr0yuwvtsssxi5l-jmup80q.png)


------------------------------------------------
# 6. Pytorch Lightning with Hugging Face

In the previous parts we used  the `segmentation_models_pytorch` to get the `segmentation model` for Pytorch platform. In this part, we will introduce very powerful library `transformers` that helps us easy to approach the state of the art of several tasks of the deep learning: NLP, Classification, ... 

We will introduce the SegFormer model use it to the semantic segmentation models. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. 

![](https://habrastorage.org/webt/ya/al/tf/yaaltf5hfe8duuxihmrfmum6aea.png)

------------------------------------------------
# 7. Train an Intance Segmentation Model with MMDetection

The last part of the segmentation series, we will cover the intances segmentation model. We will recall some approaches for the instance segmentation problem. We also introduce a efficient library for the detection task: `MMDetection`. MMDetection is an open source object detection toolbox based on PyTorch. 


![](https://habrastorage.org/webt/o9/wk/ae/o9wkaedfqkbfmf1q7csc61mmtkk.png)

Major features of mmdetection: 

- Modular Design: One decomposes the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- Support of multiple frameworks out of box: The toolbox directly supports popular and contemporary detection frameworks, e.g. Faster RCNN, Mask RCNN, RetinaNet, etc.

- High efficiency: All basic bbox and mask operations run on GPUs. The training speed is faster than or comparable to other codebases, including Detectron2, maskrcnn-benchmark and SimpleDet.

- State of the art: The toolbox stems from the codebase developed by the MMDet team, who won COCO Detection Challenge in 2018, and we keep pushing it forward.





