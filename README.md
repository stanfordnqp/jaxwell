# Jaxwell: GPU-accelerated, differentiable 3D iterative FDFD electromagnetic solver

Jaxwell is [JAX](https://github.com/google/jax) +
[Maxwell](https://github.com/stanfordnqp/maxwell-b):
an iterative solver for solving the finite-difference frequency-domain
Maxwell equations on NVIDIA GPUs.
Jaxwell is differentiable and fits seamlessly in the JAX ecosystem,
enabling nanophotonic inverse design problems to be cast as ML training jobs
and take advantage of the tsunami of innovations
in ML-specific hardware, software, and algorithms.

Jaxwell is a finite-difference frequency-domain solver
that finds solutions to the time-harmonic Maxwell's equations, specifically:

```
(∇ x ∇ x - ω²ε) E = -iωJ
```

for the electric field `E` via the API

```python
x, err = jaxwell.solve(params, z, b)
```

where `E → x`, `ω²ε → z`, `-iωJ → b`, 
`params` controls how the solve proceeds iteratively, and
`err` is the error in the solution.

Jaxwell uses
[dimensionless units](https://meep.readthedocs.io/en/latest/Introduction/#units-in-meep),
assumes `μ = 1` everywhere,
and implements stretched-coordinate perfectly matched layers (SC-PML)
for absorbing boundary conditions.

You can install Jaxwell with `pip install git+https://github.com/jan-david-fischbach/jaxwell.git` 
but the easiest way to get started is to go straight to the example 
[colaboratory notebook](https://colab.research.google.com/github/jan-david-fischbach/jaxwell/blob/main/example/colab.ipynb).

References:

- PMLs and diagonalization: [Shin2012] W. Shin and S. Fan. “Choice of the perfectly matched layer boundary condition for frequency-domain Maxwell's equations solvers.” Journal of Computational Physics 231 (2012): 3406–31
- COCG algorithm: [Gu2014] X. Gu, T. Huang, L. Li, H. Li, T. Sogabe and M. Clemens, "Quasi-Minimal Residual Variants of the COCG and COCR Methods for Complex Symmetric Linear Systems in Electromagnetic Simulations," in IEEE Transactions on Microwave Theory and Techniques, vol. 62, no. 12, pp. 2859-2867, Dec. 2014


