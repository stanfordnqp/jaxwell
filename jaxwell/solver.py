'''Loop initialization and iteration for COCG solver, see [Gu2014].'''

import jax
from jaxwell import vecfield


def loop_fns(A, b, eps):
  '''Returns the loop initialization and iteration functions.'''

  def loop_init(z, b):
    '''Forms the args that will be used to update stuff.'''
    term_err = eps * vecfield.norm(b)

    x = vecfield.zeros(b.shape)
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
