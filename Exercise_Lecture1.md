# Deep Learning Lecture 1 - Tutorial


## Step 1

Draw a 200 samples each from these two distributions

* A sample `s1` from a 2-D Gaussian with mean = `[1,1]` and covariance matrix cov = `[[1,0],[0,1]]`
* A sample `s2` from a 2-D 1 Gaussian with mean = `[0,0]` and covariance matrix cov = `[[1,0.8],[0.8,1.0]]`

## Step 2

Prepare a scatter plot of both these sets of samples on the 2-D plane, with the markers having two different colors

## Step 3

As discussed in the lecture, the simples neural network is the "Perceptron" 

$$
f(\vec{x}; \vec{w},b) = \Phi(\sum w_i x_i - b) = \Phi(\vec{w}\cdot\vec{x} - b)
$$

where $\Phi$ is the heaviside step function.

Write a function that implements the perceptron model and is able to evaluate a input sample for $w_1 = 0.2$, $w_2 = 0.4$ and $b = 0.7$

## Step 4

Evaluate the function on a 2-D grid with 100 x 100 points and visualize what the function value looks like in the 2-D plane

## Step 5

As above, we often will want to evaluate a function on many points at once. Therefore it's useful to write the function in a way such that we can quickly do so. 

One way to to this is via "matrix multiplication" 

$$
r = \Phi(X w^T - b)
$$

where $X$ is a (N,2) matrix and $w^T$ is the transpose of the weight vector $\vec{w}$ (i.e. column vector or (2,1) matrix)

Write a function that can evaluate $N$ points in one go, by making use of this relationship

## Step 6

Sample 10,000 points in the 2-D input space uniformly between (-5,5) and evaluate those points using the perceptron model

Additionally, re-add the scatter plot of the two Gaussian samples

## Step 7

* Evaluate both Gaussian samples with the perceptron model.
* Create a histogram for the output of the histogram of the two samples.

## Step 8 

Out of the 200 samples in `s1`, for how many events does the perceptron return 1 or zero respectively? 

How about for the sample `s2`?

## Step 9 

Let's say the samples of s1 should of of type `1` and the samples of s2 are of type `0`. What's the accuracy (as a percentage) of the perceptron model in predicting the right type?

## Step 10

This time we have you values for $w_1$, $w_2$, and $b$.. can you find values that are better at this prediction tasks?
