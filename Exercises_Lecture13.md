## Step 1:

Consider a N=1000 sample from `mean = [1,2]` and `cov = [[0.5,0.3],[0.3,0.3]]`. 

Write code that generates such a sample and visualize it on the 2-D plane

## Step 2: - The Bijection and its Inverse

Write two PyTorch functions that applies a affine transformation y = Ax + b and its inverse. Hint: `torch.einsum(..)` makes it easy to do.
**Note**: he're we're assuming `A` will be invertible at initialization and then also during training. We'll ignore the possibility that non-invertable matrices are possible and we will not try to enforce intertibility this.

## Step 3 - The Jacobian

Derive an expression for the Jacobian $\frac{\partial y}{\partial x}$ and write a function that computes the log abs determinant of the Jacobian $|\mathrm{det} J|$ for a given affine transform

## Step 4 - Evaluating $p(y)$

Using the Change of Variables Formula and the `torch.distributions` module write a function that evaluates the log-probability of a sample under the distribution

$$p(y = f_{A,b}(x)) $$ 

where $$p(x) = \mathrm{Normal}(0,1) $$

## Step 5

Create a PyTorch module that has Affine Transformation parameters (A,b) and provides two methods: 1) "log_prob" for a batch of samples and "generate" to generate transformed samples starting from a base distribution

