'''Solves `(∇ x ∇ x - ω²ε) E = -iωJ` for `E`.'''

from jaxwell import operators, cocg, vecfield

import dataclasses
from functools import partial
from jax import custom_vjp
import jax.numpy as np
from typing import Callable, Tuple


@dataclasses.dataclass
class Params:
  '''Parameters for FDFD solves.

  Attributes:
    ths: `((-x, +x), (-y, +y), (-z, +z))` PML thicknesses.
    pml_params: `operators.PmlParams` controlling PML parameters.
    eps: Error threshold stopping condition.
    max_iters: Iteration number stopping condition.
  '''
  pml_ths: Tuple[Tuple[int, int], Tuple[int, int],
                 Tuple[int, int]] = ((10, 10), (10, 10), (10, 10))
  pml_params: operators.PmlParams = operators.PmlParams()
  eps: float = 1e-6
  max_iters: int = 1000000


@partial(custom_vjp, nondiff_argnums=(0,))
def solve(params, z, b):
  '''Solves `(∇ x ∇ x - ω²ε) E = -iωJ` for `E`.

  Note that this solver requires JAX's 64-bit (double-precision) mode which can
  be enabled via

    ```
    from jax.config import config
    config.update("jax_enable_x64", True)
    ```

  #Double-(64bit)-precision
  see https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html

  Args:
    params: `Params` options structure.
    z: 3-tuple of `(xx, yy, zz)` arrays of type `jax.numpy.complex128`
       corresponding to the x-, y-, and z-components of the `ω²ε` term.
    b: Same as `z` but for the `-iωJ` term.
  '''
  x, err = solve_impl(z, b, params=params)
  return x, err[-1]


def solve_fwd(params, z, b):
  x, err = solve(params, z, b)
  return (x, err), (x, z)


def solve_bwd(params, res, grad):
  x, z = res
  x_grad, _ = grad
  x_adj, _ = solve_impl(z, x_grad, adjoint=True, params=params)
  z_grad = tuple(np.real(np.conj(a) * b) for a, b in zip(x_adj, x))
  return z_grad, x_adj


solve.defvjp(solve_fwd, solve_bwd)


def _default_monitor_fn(x, errs):
  pass


def solve_impl(z,
               b,
               adjoint=False,
               params=Params(),
               monitor_fn=_default_monitor_fn,
               monitor_every_n=1000,
               ):
  '''Implementation of a FDFD solve.

  Args:
    z: 3-tuple of `(xx, yy, zz)` arrays of type `jax.numpy.complex128`
       corresponding to the x-, y-, and z-components of the `ω²ε` term.
    b: Same as `z` but for the `-iωJ` term.
    adjoint: Solve the adjoint problem instead, default `False`.
    params: `Params` options structure.

  Returns:
    `(x, errs)` where `x` is the `vecfield.VecField` of `jax.numpy.complex128`
    corresponding to the electric field `E` and `errs` is a list of errors.
  '''
  shape = z[0].shape
  z, b = vecfield.from_tuple(z), vecfield.from_tuple(b)

  pre, inv_pre = operators.preconditioners(
      shape, params.pml_ths, params.pml_params)
  def A(x, z): return operators.operator(
      x, z, pre, inv_pre, params.pml_ths, params.pml_params)

  # Adjoint solve uses the fact that we already know how to symmetrize the
  # operator, `inv_pre * A * pre == pre * AT * inv_pre`. Note that the resulting
  # operator is symmetric, but not Hermitian!
  b = b * pre if adjoint else b * inv_pre
  def unpre(x): return vecfield.conj(x * inv_pre) if adjoint else x * pre

  init, iter = cocg.solver(A, b, params.eps)

  p, r, x, term_err = init(z, b)
  errs = []
  for i in range(params.max_iters):
    p, r, x, err = iter(p, r, x, z)
    errs.append(err)
    if i % monitor_every_n == 0:
      monitor_fn(unpre(x), errs)
    if err <= term_err:
      break

  monitor_fn(unpre(x), errs)

  return vecfield.to_tuple(unpre(x)), errs
