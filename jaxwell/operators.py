'''Defines the matrix-free FDFD operator.'''

import dataclasses
import functools
import itertools
import jax
import jax.numpy as np
import numpy as onp
import jaxwell.vecfield


@dataclasses.dataclass
class PmlParams:
  '''Parameters for the stretched-coordinate perfectly matched layers.'''
  w_eff: float = 1.  # Effective frequency of the PML, see [Shin2012].
  m: float = 4.  # Degree of polynomial grading of the PML, see [Shin2012].
  ln_r: float = -16.  # `ln(R)` where `R` is target reflection , see [Shin2012].


def diff_kernel(axis, transpose=False):
  '''Difference kernel for `axis`, `transpose=False` is forward differencing.'''
  if transpose:
    kernel = [-1, 1]
  else:
    kernel = [-1, 1, 0]
  shape = (1, 1) + tuple(-1 if i == axis else 1 for i in range(3))
  return onp.reshape(kernel, shape)


def spatial_diff(x,
                 axis,
                 transpose=False,
                 precision=jax.lax.Precision.HIGHEST):
  '''Spatial difference of `(1, 1, xx, yy, zz)`-shaped `x` along `axis`.'''
  kernel = np.array(diff_kernel(axis, transpose), dtype=np.complex128)
  return jax.lax.conv_general_dilated(x,
                                      kernel,
                                      window_strides=(1, 1, 1),
                                      padding='SAME',
                                      precision=precision)


def scpml_coeffs(n, th, pml_params, axis, transpose=False):
  '''Returns scpml coefficients for an axis of length `n` and a pml size `p`.'''
  pos = onp.arange(n).astype(float)
  if transpose:
    pos += 0.5

  pml_dist = onp.maximum(
      (th[0] - pos) / th[0] if th[0] > 0 else -pos,
      (pos - (n - th[1] - 0.5)) / th[1] if th[1] > 0 else pos - n)
  pml_dist[pml_dist < 0] = 0.

  s_max = (pml_params.m + 1) * pml_params.ln_r / 2.
  coeffs = onp.reciprocal(1 + 1j * s_max *
                          (pml_dist**pml_params.m) / pml_params.w_eff)
  shape = tuple(n if i == axis else 1 for i in range(3))
  return onp.reshape(coeffs, shape).astype(onp.complex128)


def stretched_spatial_diff(x, axis, th, pml_params, transpose=False):
  '''Stretched spatial difference of `(1, 1, xx, yy, zz)`-shaped `x`.'''
  if x.shape[axis] == 0:
    return np.zeros_like(x)
  else:
    coeffs = np.array(
        scpml_coeffs(x.shape[axis + 2], th, pml_params, axis, transpose),
        np.complex128)
    return coeffs * spatial_diff(x, axis, transpose)


def curl(x, ths, pml_params, transpose=False):
  '''Stretched curl of `vecfield.VecField` of `(1, 1, xx, yy, zz)` arrays.'''
  diff_fn = functools.partial(stretched_spatial_diff,
                              pml_params=pml_params,
                              transpose=transpose)
  y = []
  for i in range(3):
    j, k = (i + 1) % 3, (i + 2) % 3
    y.append(
        diff_fn(x[k], axis=j, th=ths[j]) - diff_fn(x[j], axis=k, th=ths[k]))
  return vecfield.VecField(*y)


def preconditioners(shape, ths, pml_params):
  '''`(pre, inv_pre)` as 3-tuples of `(1, 1, xx, yy, zz)` arrays.'''
  pre = []
  for axis in range(3):
    p = functools.reduce(onp.multiply, (scpml_coeffs(
        shape[i], ths[i], pml_params, axis=i, transpose=(i == axis))
                                        for i in range(3)))
    p = onp.reshape(onp.sqrt(p), (1, 1) + shape)
    pre.append(p)
  pre = vecfield.VecField(*pre)
  inv_pre = vecfield.VecField(*(1 / p for p in pre))
  return pre.as_array(), inv_pre.as_array()


def operator(x, z, pre, inv_pre, ths, pml_params):
  '''Returns symmetrized `curl(curl(x)) - z * x` operation.'''
  curl_fn = functools.partial(curl, ths=ths, pml_params=pml_params)
  x *= pre
  y = curl_fn(curl_fn(x, transpose=True)) - z * x
  return y * inv_pre
