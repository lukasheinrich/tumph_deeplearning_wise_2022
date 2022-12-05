# Training Convolutional Nets

In this exercise we will look more into ConvNets

As discussed in the lecture, the key ideas to convnets are local connectivity and weight sharing which lead us to convolutions.

Before trying to *learn* convolutional filters, let's manually construct them

## Step 1 - Image Processing

Write a function that downloads an image at a given URL and converts it into greyscale and returns a 2D array.

Tip: use the `requests` and the `PIL` python libraries

## Step 2 - the `torch.nn.Conv2D` object

Research how to access the parameters of the convolution kernel module `torch.nn.Conv2D` in PyTorch and make sure you understand where 
kernel size, input channels and output channels come into definint the shape of the parameter tensors.


## Step 3 - Manually creating Convolutions

Create a `Conv2D` module with 1 input channel and 2 output channels with kernel size 3

* set the weights of of first output channel is a vertical edge detector
* set the weights of the second output channel is a horizontal edge detector
* create a size 1 input batch with a greyscale image of your choice and visualize the outputss

## Step 4 - Make a CNN

Create a ConvNet with the following structure

* Conv 5 x 5 ( 1 -> 16 channels) -> ReLU -> MaxPool 2 x 2
* Conv 3 x 3 ( 16 -> 16 channels) -> ReLU -> MaxPool 2 x 2
* Conv 2 x 2 ( 16 -> 32 channels) 


Find out what the output on a random MNIST-like torch tensor is, i.e. `x = torch.randn(123,1,28,28)`


* Use `torch.nn.Unflatten` to make the model work with "flattened" MNIST tensors as we had them in the prior exercise, i.e. the model 
should work with a tensor `x = torch.randn(123,784)`
* Use `torch.nn.Flatten` to flatten all the remaining dimensions after the three convolutions into

How big is this intermediate representation of an image?

## Step 5 - Train a CNN on MNIST

* Take a note of the number of outputs after `Flatten` and add a final lineary projection layer (i.e. a perceptron) to the network to 
predict 10 logits
* Adapt your solution from Exercise 6 to train a CNN on mnist images


## Step 6 - Use a pretrained network

* Check out the following link, and try using the famour `inception_v3` model to process an image of your choice
* https://pytorch.org/hub/pytorch_vision_inception_v3/
