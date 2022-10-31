# Exploring the Bias Variance Tradeoff


## Step 1: Write a function `generate_data(N)` 

Write a function `generate_data(N)` that produces `N` samples from the following model

$$
p(s) = p(x,y) = p(y|x)p(x)
$$

with the following true" underlying polynomial noisy model

$$p(x) = \mathrm{Uniform}(-1,1)$$
$$p(y|x) = \mathrm{Normal}(\mu = f(x),\sigma = 0.2)$$
$$f(x) = \sum_i p_i x^i$$,

with $p_0 = -0.7, p_1 = 2.2, p_2 = 0.5, p_3 = 1.0$


Hint: you can use `np.polyval` to evaluate a polynomial with a fixed set of coefficients (but watch out for the order)

The function should return a array of `x` values and an array of `y` values

## Step 2: Plot Samples and Functions

Write a function `plot(ax,train_x,train_y,p_trained,p_true)` that
takes a matplotlib axis object and plots

* plot the true function 
* plot a second (trained or random) function 
* plot the samples 

In the end you should be able to call it like this:

```
f = plt.figure()
x,y = generate_data(10)
plot(f.gca(),x,y,np.random.normal(size = (4,)), p_true)
```

## Step 3

One can show that given a Hypothesis Set of Polynomial functions

$$f(x) = \sum_i w_i x^i$$

and a risk function of the following form

$$l(s) = l(x,y) = (y - f(x))^2$$

there is a closed form solution for finding the empirical risk minimization, where the best fit coefficients $\vec{w}$ is given by

$$
w = (X^T X)^{-1} X^T y
$$

where $X$ is the matrix with rows $(x^0,x^1,x^2,x^3,\dots,x^d)$ and one row for each sample

$$
X = \left(
\begin{array}{}
x_1^0,\dots,x_1^d  \\
x_2^0,\dots,x_2^d  \\
\dots \\
x_n^0,\dots,x_n^d  \\
\end{array}
\right)
$$

* Write a function `learn(train_x, train_y, degree)` to return the $(d+1)$ optimal coefficients for a polynomial fit of degree $d$.
* Fit a sampled of 5 data points with degree 4
* Plot the Trained function together with the true function using the plotting method from the last step
* Try this multiple time to get a feel for how much the data affects the fit
* Try degree 1 and observe how the trained function is much less sensitive to the data

## Step 4

Write a function to evaluate the risk or loss of a sample. Use our loss function for which we have the training procedure above

$$
l(s) = l(x,y) = (y - f(x))^2
$$

and right a function `risk(x,y_true, trained_coeffs)` to cmpute

$$
\hat{L} = \frac{1}{N}\sum_i l(s_i) = \frac{1}{N}\sum_i l(x,y) = \frac{1}{N}\sum_i (y - f(x))^2
$$

* Draw a size 100 data sample and fit the result to obtain trained coefficients
* Draw 10000 samples of size 10 and compute their empirical risk under the trained coefficients
* Repeat the same but use the true coefficients of the underlying data-generating process
* Histogram the two sets of 10,000 risk evaluations. Which one has lower average risk?

## Step 5

Explore how the fit improves when adding more data. Plot the best fit model for data set sizes of 

$$N = 5,10,100,200,1000$$

## Step 6

Explore how the fit changes when using more and more complex models. Plot the best fit model for degrees

$$d = 1,2,5,10$$

## Step 7 Bias-Variance Tradeoff

Draw two datasets:

* A train dataset with $N=10$
* A test dataset with $N=1000$

Perform trainings on the train dataset for degrees $1\dots8$ and store the training coefficients

* Evaluate the risk under the various trainings for the train and the test dataset
* Plot the train and test risk as a function of the polynomial degree
