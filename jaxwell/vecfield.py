import jax.numpy as np
from typing import Any, NamedTuple


class VecField(NamedTuple):
  '''Represents a 3-tuple of arrays.'''
  x: Any
  y: Any
  z: Any

  @property
  def shape(self):
    assert self.x.shape == self.y.shape == self.z.shape
    return self.x.shape

  @property
  def dtype(self):
    assert self.x.dtype == self.y.dtype == self.z.dtype
    return self.x.dtype

  def as_array(self):
    return VecField(*(np.array(a) for a in self))

  def __add__(x, y):
    return VecField(*(a + b for a, b in zip(x, y)))

  def __sub__(x, y):
    return VecField(*(a - b for a, b in zip(x, y)))

  def __mul__(x, y):
    return VecField(*(a * b for a, b in zip(x, y)))

  def __rmul__(y, x):
    return VecField(*(x * b for b in y))


def zeros(shape):
  return VecField(*(np.zeros(shape, np.complex128) for _ in range(3)))


# TODO: Check if this hack is still necessary to obtain good performance.
def dot(x, y):
  z = VecField(*(a * b for a, b in zip(x, y)))
  return sum(np.sum(np.real(c)) + 1j * np.sum(np.imag(c)) for c in z)


def norm(x):
  return np.sqrt(sum(np.square(np.linalg.norm(a)) for a in x))

def conj(x):
  return VecField(*(np.conj(a) for a in x))

def real(x):
  return VecField(*(np.real(a) for a in x))
