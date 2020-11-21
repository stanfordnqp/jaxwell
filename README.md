# Jaxwell: GPU-accelerated 3D iterative FDFD electromagnetic solver

Jaxwell is [JAX](https://github.com/google/jax) +
[Maxwell](https://github.com/stanfordnqp/maxwell-b):
an iterative solver for solving the the finite-difference frequency-domain
Maxwell equations on NVIDIA GPUs. 

Jaxwell solves the equation `(∇ x ∇ x - ω²ε) E = -iωJ`
for the electric field `E`.
Jaxwell uses
[dimensionless units](https://meep.readthedocs.io/en/latest/Introduction/#units-in-meep),
assumes `μ = 1` everywhere,
and implements stretched-coordinate perfectly matched layers (SC-PML)
for absorbing boundary conditions.

You can install Jaxwell with `pip install ...`,
but the easiest way to get started is to go straight to the example 
[colaboratory notebook](link here).

References:

- PMLs and diagonalization: [Shin2012] W. Shin and S. Fan. “Choice of the perfectly matched layer boundary condition for frequency-domain Maxwell's equations solvers.” Journal of Computational Physics 231 (2012): 3406–31
- COCG algorithm: [Gu2014] X. Gu, T. Huang, L. Li, H. Li, T. Sogabe and M. Clemens, "Quasi-Minimal Residual Variants of the COCG and COCR Methods for Complex Symmetric Linear Systems in Electromagnetic Simulations," in IEEE Transactions on Microwave Theory and Techniques, vol. 62, no. 12, pp. 2859-2867, Dec. 2014


