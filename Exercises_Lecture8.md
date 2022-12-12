# Training Permutation-Invariant Networks

This week, we'll focus on permutation invariant neural networks. In general the answer here is graph neural networks but for this exercise we will focus on datasets that are just bags of objects, without any edges, i.e.
`DeepSets` as discussed in the lecture.


## Step 1: Create the Data

First, let's create a data-generating function `make_data(N)` that produces the `N` instances following data 

* each instance has 10 objects
* each object has 1 feature that either has value $x=0$ or a nonzero value $x!=0$
* for data from class 0, a random subset of 3 objects has value 1/3
* for data from class 1, a random subset of 5 objects has value 1/5

The function `make_data(N)` should return a balanced (i.e. equal numbers of class 1 and class 0) dataset through a PyTorch feature matrix of shape `(N,10,1)` and a PyTorch label array of shape `(N,1)`

* Visualize the a batch of data through `plt.imshow(X)`

## Step 2: Create the Models

Now we create two models that we want to compare

* a DeepSet model
* a standard MLP

For the DeepSet create a class `DeepSet` that subclasses `torch.nn.Module`. Recall the definition of a DeepSet. It has two components

* an element wise network that just transforms the per-element features into a new representation
* a global set-wide network that takes the pooled representations of the objects and performs the final classification

For the element-wise network create a small MLP that maps `\mathbb{R} \to \mathbb{R}` with two hidden layers and a `Tanh` activation and a small number of hidden units (e.g. 5) per layer

For the global network use a single hidden layer with also a small number of units and prepare the network for binary classification (i.e. use a sigmoid as the last operation)

When implementing the `forward()` of the `DeepSet` class go through the right order of operations
* calculate the per-element values for all objects
* implement the pooling (use sum pooling) 
* pass the pooled representation into the set-wide network

For the MLP, just use a standard network architecture with `torch.nn.Sequential`

## Step 3: Writing the Training Loop

Write a training loop that takes an untrained model and trains it for some number of steps (e.g. 4000)
using the Adam optimizer. At each loop iteration, draw random data from `make_data` (this is effectively an infinitely large training dataset)


* Train on binary cross entropy as a loss function.
* As you are training, track the loss value
* The `train(model)` function should return both the trained model as well as the loss history

## Step 4: Train the Models

Train both the Deep Set as well as the MLP!

## Step 5: Comparing MLP and DeepSet

* Write a function `count_params(model)` that returns the number of parameters for the neural network
* Compute the number of parameters for both trained models
* Plot the loss curve for both the MLP as well as the DeepSet 

## Step 6: Verify the Permutation Invariance

The advantage of encoding known symmetries into the network is not only that the learning is faster but of course also that the symmetries hold exactly. 

* Take an example instance of the data and produce 3 random permutations of it via `np.random.permutation`
* Evaluate the model on each of these permuted inputs
* Compare / Pring the model outputs and verify that the deep set is indeed permutation invariant, while the MLP is not

## Step 7: Extra Question

* How many parameters do you need in order to match the Deep Set performance with the MLP?
* In the Deep Set try removing the non-Linearity in the element wise network
* can you explain what is happening and why?
