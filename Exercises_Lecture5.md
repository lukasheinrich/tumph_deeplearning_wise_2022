# Automatic Differentiation

Matrix Multiplication is defined as follows

$$R_{ij} = \sum_k A_{ik}B_{kj}$$


## Step 1

Derive a formula for how many basic operations are required for a matrix multiplication


## Step 2
Implement an non-fancy algorithm (just use python loops) that multiplies to matrices and returns
  * The resulting matrix R
  * A count of the number of additions and multiplications (i.e. floating point operations) required for the computation

## Step 3
* Check that your code works by comparing the result with `np.matmul` using the function `np.allclose` and using random matrices (from a standard normal)
* Check your formula for the full number of floating point operations (FLOPS)

## Step 4

Consider this decreasing set of matrices
* A = Matrix with shape (100,200)
* B = Matrix with shape (50,100)
* C = Matrix with shape (2,50)

We are interested in computing $M = ABC$ and we could do it 2 ways

* The "forward" way $C(BA)$
* The "backward" way $(CB)A$

* Use your matrix multiply to compute the computational cost of these two options
* cross-check with your formula that this matches expectations
* Which one is more advantagous for a linear map from 200 → 2 dimensions? What's the ratio of the reuqired FLOPs?

## Step 5

Repeat the same Exercise for an increasing set of matrices


* A = Matrix with shape (300,2)
* B = Matrix with shape (500,300)
* C = Matrix with shape (1000,500)

Which direction is more advantages now?

## Step 6

Take the function $$\mathbb{R}^3 \to \mathbb{R}^2$$

* Install the library JAX `pip install jax jaxlib`

and consider the following function

```python
import jax
import jax.numpy as jnp

def func(x):
    x1,x2,x3 = x
    z1 = 2*x1*x2
    z2 = x3**2
    return jnp.stack([z1,z2])
```

* What is the Jacobian of this function $\frac{\partial z_i}{\partial x_i}$?
* What does the Jacobian look like at the input `x = [2.,3.,2.]` ?

## Step 7

`jax` allows us to extract the Jacobian via Jacobian Vector Products of Vector Jacobian products

To compute $Jv$ at a point $x$ we can use `value, jac_column = jax.jvp(function, (x,), (v,))` 

This will give us `value` corresponding to $f(x)$ and `jac_column` corresponding to $Jv$

* Use this API to extract the 3 columns of the Jacobian via 3 special choices of $v$

## Step 8

`jax` also allows us to to VJPs. As we know these are the ones that are most important for Deep Learning

Here the API is slighly different. We can call `value, vjp_func = jax.vjp(function,x)`
The variable `value` will correspond as usual to $f(x)$, while `jvp_func` is a function that we can call
with `vjp_func(v)` to compute $v^T J$

Congratulations - you've now used your first automatic differentiation system!

To celebrate let's calculate the gradient of some fun functions

In JAX we can do this like this

```python
def myfunction(x):
    return x**2


value_and_gradient_func = jax.vmap(jax.value_and_grad(myfunction))
```

`value_and_gradient_func` now is a function in which we an pass an array of x values and it will return an array
of y values and the gradient value $\frac{\partial f}{\partial x}$ at that point.

```python
values, gradients = value_and_gradient_func(xarray)
```

Use this to compute the function value and gradients for `x^2`, `\sin(x^2)\cos(x)`, or whatever function you can thing of on the interval (-5,5)
