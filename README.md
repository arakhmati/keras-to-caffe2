# Keras to Caffe2 Converter
Caffe2 is currently one of the fastest Deep Learning libraries on Android and iOS. 
However, a lot of researches and engineers use keras to train their models.
This tool combines the best of both worlds, as it allows the models to be trained in Keras and consequently deployed on Android and iOS by converting them to Caffe2.

### Prerequisites
[Numpy](https://github.com/numpy/numpy)
[Keras](https://github.com/fchollet/keras) (Tested with Tensroflow backend only)
[Caffe2](https://github.com/caffe2/caffe2)
### Installing
Clone the repository and install it as a python module:
```
git clone https://github.com/arakhmat/keras-to-caffe2
cd keras-to-caffe2
pip install -e .
```
While in the same directory, test the installation by running:
```
python test_conv.py
```

## Supported Layers
* Conv2D
* MaxPool2D
* Flatten
* Dense
* BatchNormalization (for Conv2D layer only)

## Supported Activations
* Relu
* Softmax

## Known Limitations
* Only convolutional 2D neural networks are supported
* There is no BatchNormalization layer for dense networks
* Conv2D and MaxPool2D layers must have 'channel_first' data format
* Conv2D layers cannot be padded
* It is assumed that a network always has Conv2D as its first layer

