## Building a Multilayer Perceptron

Let's follow the lecture and build a multilayer perceptron to approximate the function $\theta(x)$ in our variational inference model $$\mathrm{Bernoulli}(z|\theta(x))$$ with the following structure

* 2 input nodes
* 2 internal features computed using ReLu activation
* 1 output feature with sigmoid activation that 

From Ex3, we had already the following code ready

```python
import numpy as np
import matplotlib.pyplot as plt

def linear(X,pars):
    out = X @ pars[:2].T + pars[2]
    grad_pars = np.column_stack([X,np.ones(len(X))])
    return out,grad_pars

def sigmoid(x):
    out = 1/(1+np.exp(-x))
    grad = out*(1-out)
    return out,grad.reshape(-1,1)

def theta_perceptron(X,pars):
    feature1,grad_linear1 = linear(X,pars[0:3])
    activation1,grad_sigmoid1 = sigmoid(feature1)
    return activation1, grad_sigmoid1*grad_linear1
```

### Step 1

* Write a function `relu(x)` that produces the relu activation function and its gradient.
* make sure the gradient is returned as a shape (N,1)
* plot it for the range: `x = np.linspace(-5,5,1001)`

### Step 2

Write a function `multilayer_perceptron(X,pars)` that takes 9 parameters (3 for each artificial neurons: 2 weights and 1 bias) that can compute the $\theta(x)$ prediction for many input points at once (batched computation)


### Step 3

Use this function to plot the contour of the multilayer perceptron

```python
def plot_contour(func,pars):
    grid = np.mgrid[-5:5:101j,-5:5:101j]
    X = np.swapaxes(grid,0,-1).reshape(-1,2)
    out = func(X,pars)
    out = np.swapaxes(out.reshape(101,101),0,-1)
    plt.contour(grid[0],grid[1],out)
```

Try for example parameter vectors: `np.array([1,0,0,0,1,0,1,1,0])` or what every you like.


### Step 4

Now comes the hard part!

We want to compute gradients for this function

$$
\vec{a} = [\;\mathrm{ReLU}(\mathrm{Lin}(\vec{x},\phi_1))\;,\;\mathrm{ReLU}(\mathrm{Lin}(\vec{x},\phi_2))\;]\\
a_2 = \sigma(\mathrm{Lin}(\vec{a},\phi_3))\\
$$

To compute the gradient we need to also have the gradients $\frac{\partial \mathrm{Lin}}{\partial \vec{x}}$

* Write a new function that also outputs the partial derivatives with respect to $\vec{x}$
* Hint: the output shape of `grad_x` should be (1,2)!

```python
def linear(X,pars):
   ...
   return out,grad_pars,grad_x
```

### Step 5

With this in hadn you can now carefully piece back the gradients together.

We can start with the function like this

```python
def multilayer_perceptron(X,pars):
    feature1,grad_f1,_ = linear(X,pars[0:3])
    feature2,grad_f2,_ = linear(X,pars[3:6])
    activation1,grad_r1 = relu(feature1)
    activation2,grad_r2 = relu(feature2)
    hidden = np.column_stack([activation1,activation2])
    output_feature,grad_f3,grad_x = linear(hidden,pars[6:9]) #here is the new gradient!
    theta,grad_sig = sigmoid(output_feature)
    return theta
```

* Try to work out what the 9-dimensional gradient vector looks like for $\nabla_\phi \theta(x,\phi)$
* Hint 1: the gradient should have the shape `(N,9)`
* Hint 2: The following would be a correct result

```python
Xtest = np.array([[1.0,2.0]])
pars = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
multilayer_perceptron(Xtest,pars)

value: [0.9552123]
gradient: [[0.02994724 0.05989447 0.02994724 0.03422541 0.06845082 0.03422541
  0.03422541 0.08556353 0.04278176]]
``` 

### Step 6

## Training the Multi-Layer Perceptron

We can adapt the learning code from Exercise 3 to learn the multi-layer perceptron.

We can get some interesting datasets from the `scikit-learn` package

* Install scikit-learn via `pip install scikit-learn`
* Use the following data-generating function and plot the data

```python
def generate_data(N):
    import sklearn.datasets as skld
    X,z = skld.make_circles(N, noise = 0.1, factor=0.5)
    filt =  (X[:,1] > 0)
    return X[filt],z[filt]
```

We can now adapt our training code from Exercise 3 (take some time to go through it, but nothing fundamental changed) and train out multilayer perceptron!

```python
def loss(z,theta):
    out = np.where(z==1,-np.log(theta),-np.log(1-theta))
    grad = np.where(z==1,-1/theta,-1/(1-theta)*(-1))
    return out,grad.reshape(-1,1)

def empirical_risk(X,z,theta_func,pars):
    theta,grad_theta = theta_func(X,pars)
    loss_val,grad_loss = loss(z,theta)
    grad1 = grad_loss*grad_theta
    grad = np.concatenate([grad1], axis=-1)
    return loss_val.mean(axis=0),grad.mean(axis=0),theta

def plot(X,z,theta_func,pars):
    grid = np.mgrid[-5:5:101j,-5:5:101j]
    Xi = np.swapaxes(grid,0,-1).reshape(-1,2)   
    _,_,zi = empirical_risk(Xi,np.zeros(len(Xi)),theta_func,pars)
    zi = zi.reshape(101,101).T
    plt.contour(grid[0],grid[1],zi, levels = np.linspace(0,1,21))
    plt.scatter(X[:,0],X[:,1],c = z)
    plt.xlim(-2,2)
    plt.ylim(-2,2)

def learn(data,pars,theta_func, nsteps = 15000):
    X,z = data
    for i in range(nsteps):
        val,grad,_ = empirical_risk(X,z,theta_func,pars)
        pars = pars - 0.01*grad
        if i % (nsteps//4) == 0:
            print(val,pars)
            plot(X,z,theta_func,pars)
            plt.gcf().set_size_inches(3,3)
            plt.show()
    return pars
```

# Try Learning

Try learning on a dataset of 1000 samples and initialize the parameters with 

`pars = np.array([-.1,1,0,.1,1,0,-.1,-.1,0])`

* The learning itself, depending on the initialization might or might not be super-convincing
* Try executing this multiple times to try out different initializations
* Ultimately, we will need to add more & more neurons, but as you see the gradient calculation is painful!
* Try other initializations and see what happens

### Going Beyond

* Can you extend this to N hidden neurons?
* If yes, try the following learning problem with e.g. 15 neurons

```python
def generate_data(N):
    import sklearn.datasets as skld
    X,z = skld.make_circles(N, noise = 0.1, factor=0.5)
    return X,z
```


