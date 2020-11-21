'''Loop initialization and iteration for COCG solver, see [Gu2014].'''

import jax
from jaxwell import operators
from jaxwell import vecfield


def loop_fns(shape, ths, pml_params, eps):
  '''Returns the loop initialization and iteration functions.'''
  pre, inv_pre = operators.preconditioners(shape[2:], ths, pml_params)
  A = lambda x, z: operators.operator(x, z, pre, inv_pre, ths, pml_params)

  def loop_init(z, b):
    '''Forms the args that will be used to update stuff.'''
    b = b * inv_pre
    term_err = eps * vecfield.norm(b)

    x = vecfield.zeros(shape)
    r = b - A(x, z)
    p = r

    return p, r, x, term_err

  @jax.jit
  def loop_iter(p, r, x, z):
    '''Run the iteration loop `n` times.'''
    rho = vecfield.dot(r, r)
    v = A(p, z)
    alpha = rho / vecfield.dot(p, v)
    x += alpha * p
    r -= alpha * v
    beta = vecfield.dot(r, r) / rho
    p = r + beta * p
    err = vecfield.norm(r)
    return p, r, x, err

  return loop_init, loop_iter
