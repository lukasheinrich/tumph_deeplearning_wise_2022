# Training Recurrent Neural Nets

This week we'll train recurrent neural networks. Here it's all about processing an stream of data in a particular order, which happens for example for time-series data.

RNNs process the data by introducing "memory" which they modify based on each time-steps data.

To see how this works, we'll work on classifying sequences into binary categories:

* sequences with just positive random integers vs.
* sequence with a -1 somewhere in the sequence

The RNN will need to track the elements in the sequence and store into memory whether it has seen a "-1" or not



## Step 1 Write a sequence generator

To start let's write a data generating function 

* Write a function `make_datum(size)` that creates a array of size `size` filled with random integers between 0 and 9

* With a probability of 50% overwrite a random position with a -1

* The function should return `x` as an `(size,1)` array and `y` as a label (0 for unmodified, 1 for modified)

## Step 2 Write a mini-batch generator

To generate minibatches of data create a function `make_data(N)` that randomly samples a data size between 10 and 25 and produces `N` samples of data with that chosen size. The output should look as follows:

* The data array should be of shape `(Nbatch, size, 1)`
* The label array should be of shape `(Nbatch, 1)`

## Step 3: Create a RNN class

Create a `torch.nn.Module` subclass that holds two sub-modules

* a RNN that consumes the sequence of N elements and produces a sequence of N outputs. Make the class configurable, such that one could use either `torch.nn.RNN` or `torch.nn.LSTM` as a possibile RNN flavor


* a MLP that implements the "high-level" task of doing binary classification based on an output vector. Here a simple perceptron-style network that is just a linear layer followed by a sigmoid is sufficient

In the `forward` of your module, pass the input sequence through the RNN. This returns to you the output sequence (and additionally some data about the hidden memory states)

Pass the output sequence (all time steps) through your MLP "head" network. This will give you information how the decision evolves as the sequence is processed. The final answer is the output of the MLP at the last time-step

The `forward` should return both the final answer as well as the full sequence of decisions leading up to it

## Step 4: Write a training loop and train the networks

Write a training loop for our RNN with signature `train(rnn_cls)`, where `rnn_cls` is either `torch.nn.LSTM` or `torch.nn.RNN` that trains your new module on the data we get from the data-generating process you wrote above. Add code that tracks the evolution of the loss during training

* Use your training loop to train both a `LSTM`- and a `RNN`-based network
* Plot the loss curves for both

## Step 5: Testing the results

To see how the decision is being reached as the sequence in being processed, try out the following thing

* take the array `[1,2,3,4,5,6,7,8,9]`
* Write code that visualizes the output for all time steps of the trained models
* Visualize the output
* go back to your test array and place a -1 at a position of your choosing, re-visualize and see whether the RNN changes it's decision at the right moment
