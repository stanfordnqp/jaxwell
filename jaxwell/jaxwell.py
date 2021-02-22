'''Solves `(∇ x ∇ x - ω²ε) E = -iωJ` for `E`.'''

from jaxwell import operators, cocg, vecfield

import dataclasses
from typing import Callable, Tuple


def _default_monitor_fn(x, errs):
  pass


@dataclasses.dataclass
class JaxwellParams:
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


def solve(z,
          b,
          adjoint=False,
          params=JaxwellParams()):
  '''Implementation of a FDFD solve.

  Args:
    z: `vecfield.VecField` of `jax.numpy.complex128` corresponding to `ω²ε`.
    b: `vecfield.VecField` of `jax.numpy.complex128` corresponding to `-iωJ`.
    adjoint: Solve the adjoint problem instead, default `False`.
    params: `JaxwellParams` options structure.

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
