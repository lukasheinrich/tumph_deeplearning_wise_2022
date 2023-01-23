# Training an Autoencoder 

As discussed in the lecture, one of the standard self-supervised learning techniques is "auto-encoding", that is 
taking a datum $x$ to a "code" $c = f_\phi(x)$ and reconstructing it again $x' = g_\theta(c)$

The training goal is to make the reconstruction as close as possible to the original: $L(x,x')$

# Step 1

Write an Autoencoder PyTorch module that has a latent space of dimension 2

* a `encoder` submodule that is a MLP that can take `(N,28,28)` tensors and return "encoded" tensors of shape `(N,2)`
* a `decoder` submodule that is also a MLP which 

* a `forward()` method that takes a batch of MNIST images and returns a 2-tuple with 1) the encoded  values and 2)  the reconstructed images

# Step 2

Use `pip install mnist` to get access to the MNIST dataset and write a data-generating function
`sample_train(N, return_labels = False)` that samples N instances from the training set

optionally it should also return the labels (this is just for later, since we are doing unsupervised learning, we don't need the labels for training)

# Step 3

Write a training loop with your Autoencoder model that trains using mini-batches of size 200 and using a standard mean-squared-error (MSE) loss. Train for 50k steps with learning rate 1e-3 (with Adam)

**Optionally**: it's nice to track how the autoencoder is doing as you train.

Write a plotting function `plot(model,samples)` that takes a mini batch of size 1 and plots both the original as well as the reconstructed images side by side

# Step 4: Exploring the Latent Space

Sample a mini batch of size 10,000 **with** labels and encode it into the latent space with the
trained encoder. Since it's a 2-D latent space we can easily visualize it. Plot the distribution
of codes as a scatter plot in the $(c_1,c_2)$-plane and color the markers according to the true label

Observe how the individual clusters correspond among other things to the label

# Step 5: Generating new images

Given the distribution in the latent space you should have a good feel, which type of code, 
corresponds to which digit.

* Try to take a code that you think would generate a 4

As you see, the distribution in the latetn space is a bit unruly, if you would

* Generate a digit based on a random R^2 value

you would have a hard time recognizing it as a digit. Try it!

To fix this, we discussed in the lecture the idea of a "Variational Autoencoder" that tries
to control the latent distribution
