'''Solves `(∇ x ∇ x - ω²ε) E = -iωJ` for `E`.'''

from jaxwell import operators, cocg, vecfield

import dataclasses
from functools import partial
from jax import custom_vjp
from typing import Callable, Tuple


def _default_monitor_fn(x, errs):
  pass


@dataclasses.dataclass
class Params:
  '''Parameters for FDFD solves.

  Attributes:
    ths: `((-x, +x), (-y, +y), (-z, +z))` PML thicknesses.
    pml_params: `operators.PmlParams` controlling PML parameters.
    eps: Error threshold stopping condition.
    max_iters: Iteration number stopping condition.
    monitor_fn: Function of the form `monitor(x, errs)` used to show progress.
    monitor_every_n: Cadence for which to call `monitor_fn`.
  '''
  pml_ths: Tuple[Tuple[int, int], Tuple[int, int],
                 Tuple[int, int]] = ((10, 10), (10, 10), (10, 10))
  pml_params: operators.PmlParams = operators.PmlParams()
  eps: float = 1e-6
  max_iters: int = 1000000
  monitor_fn: Callable[[], None] = _default_monitor_fn
  monitor_every_n: int = 1000


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
  x, _ = solve_impl(z, b, params=params)
  return x


def solve_fwd(params, z, b):
  x, _ = solve_impl(z, b, params=params)
  return x, (x, z)


def solve_bwd(params, res, grad):
  x, z = res
  x_adj, _ = solve_impl(z, grad, adjoint=True, params=params)
  z_grad = vecfield.real(vecfield.conj(x_adj) * x)
  return z_grad, x_adj


solve.defvjp(solve_fwd, solve_bwd)


def solve_impl(z,
               b,
               adjoint=False,
               params=Params()):
  '''Implementation of a FDFD solve.

  Args:
    z: `vecfield.VecField` of `jax.numpy.complex128` corresponding to `ω²ε`.
    b: `vecfield.VecField` of `jax.numpy.complex128` corresponding to `-iωJ`.
    adjoint: Solve the adjoint problem instead, default `False`.
    params: `Params` options structure.

  Returns:
    `(x, errs)` where `x` is the `vecfield.VecField` of `jax.numpy.complex128`
    corresponding to the electric field `E` and `errs` is a list of errors.
  '''
  shape = b.shape

  pre, inv_pre = operators.preconditioners(
      shape[2:], params.pml_ths, params.pml_params)
  def A(x, z): return operators.operator(
      x, z, pre, inv_pre, params.pml_ths, params.pml_params)

  # Adjoint solve uses the fact that we already know how to symmetrize the
  # operator, `inv_pre * A * pre == pre * AT * inv_pre`. Note that the resulting
  # operator is symmetric, but not Hermitian!
  b = b * pre if adjoint else b * inv_pre
  def unpre(x): return vecfield.conj(x * inv_pre) if adjoint else x * pre

  init, iter = cocg.cocg(A, b, params.eps)

  p, r, x, term_err = init(z, b)
  errs = []
  for i in range(params.max_iters):
    p, r, x, err = iter(p, r, x, z)
    errs.append(err)
    if i % params.monitor_every_n == 0:
      params.monitor_fn(unpre(x), errs)
    if err <= term_err:
      break

  params.monitor_fn(unpre(x), errs)

  return unpre(x), errs
