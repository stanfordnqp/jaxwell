import jax.numpy as np


def double_precision_enabled():
  '''Returns `True` iff double-precision is enabled.'''
  return np.zeros((1, ), np.float64).dtype == np.float64
