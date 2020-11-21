'''Solves `(∇ x ∇ x - ω²ε) E = -iωJ` for `E`.'''

from jaxwell import solver
from jaxwell import vecfield


def _default_monitor_fn(x, errs):
  pass


def solve(z,
          b,
          ths,
          pml_params,
          eps=1e-6,
          max_iters=1000,
          monitor_fn=_default_monitor_fn,
          monitor_every_n=100):
  '''Solves `(∇ x ∇ x - ω²ε) E = -iωJ` for `E`.

  Note that this solver requires JAX's 64-bit (double-precision) mode.
  Specifically, the `z` and `b` inputs, as well as the `x` outputs are all
  `vecfield.VecField` objects of type `jax.numpy.complex128` and of shape
  `(1, 1, xx, yy, zz)` (the extra singular dimensions are used for convenience
  when performing the spatial differencing via convolution operators).

  JAX's 64-bit can be enabled via

    ```
    from jax.config import config
    config.update("jax_enable_x64", True)
    ```

  or equivalent method, see https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#Double-(64bit)-precision

  Args:
    z: `vecfield.VecField` of `jax.numpy.complex128` corresponding to `ω²ε`.
    b: `vecfield.VecField` of `jax.numpy.complex128` corresponding to `-iωJ`.
    ths: `((-x, +x), (-y, +y), (-z, +z))` PML thicknesses.
    pml_params: `operators.PmlParams` controlling PML parameters.
    eps: Error threshold stopping condition.
    max_iters: Iteration number stopping condition.
    monitor_fn: Function of the form `monitor(x, errs)` used to show progress. 
    monitor_every_n: Cadence for which to call `monitor_fn`.

  Returns:
    `(x, errs)` where `x` is the `vecfield.VecField` of `jax.numpy.complex128`
    corresponding to the electric field `E` and `errs` is a list of errors.
  '''
  init_fn, iter_fn = solver.loop_fns(b.shape, ths, pml_params, eps)

  p, r, x, term_err = init_fn(z, b)
  errs = []
  for i in range(max_iters):
    p, r, x, err = iter_fn(p, r, x, z)
    errs.append(err)
    if i % monitor_every_n == 0:
      monitor_fn(x, errs)
    if err <= term_err:
      break

  monitor_fn(x, errs)
  return x, errs
