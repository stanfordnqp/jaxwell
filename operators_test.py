# TODO: Remove.
from jax.config import config
config.update("jax_enable_x64", True)

import functools
import jax.numpy as np
import numpy as onp
import operators as ops
import unittest
import vecfield


class TestOperator(unittest.TestCase):
  def test_diff_kernel(self):
    onp.testing.assert_array_equal(ops.diff_kernel(2), [[[[[-1, 1, 0]]]]])
    onp.testing.assert_array_equal(ops.diff_kernel(2, transpose=True),
                                   [[[[[-1, 1]]]]])

  def test_spatial_diff(self):
    onp.testing.assert_array_equal(
        ops.spatial_diff(np.array([[[[[0, 1, 0]]]]], np.complex128), axis=2),
        np.array([[[[[0, 1, -1]]]]], np.complex128))
    onp.testing.assert_array_equal(
        ops.spatial_diff(np.array([[[[[0, 1, 0]]]]], np.complex128),
                         axis=2,
                         transpose=True),
        np.array([[[[[1, -1, 0]]]]], np.complex128))

  def test_scpml_coeffs(self):
    self.assertEqual(
        ops.scpml_coeffs(5, (2, 2), ops.PmlParams(), axis=0).shape, (5, 1, 1))
    self.assertEqual(
        ops.scpml_coeffs(5, (2, 2), ops.PmlParams(), axis=1).shape, (1, 5, 1))
    self.assertEqual(
        ops.scpml_coeffs(5, (2, 2), ops.PmlParams(), axis=2).shape, (1, 1, 5))
    onp.testing.assert_array_almost_equal(
        np.reshape(ops.scpml_coeffs(5, (2, 2), ops.PmlParams(), axis=0),
                   (-1, )),
        [
            6.24609619e-04 + 0.02498438j, 1.37931034e-01 + 0.34482759j, 1.,
            9.76167779e-01 + 0.15252622j, 6.20421814e-03 + 0.07852214j
        ])
    onp.testing.assert_array_almost_equal(
        np.reshape(ops.scpml_coeffs(5, (2, 0), ops.PmlParams(), axis=0),
                   (-1, )), [
                       6.24609619e-04 + 0.02498438j,
                       1.37931034e-01 + 0.34482759j, 1., 1., 1.
                   ])
    onp.testing.assert_array_equal(
        np.flip(
            np.reshape(
                ops.scpml_coeffs(5, (2, 2), ops.PmlParams(w_eff=10.), axis=0),
                (-1, ))),
        np.reshape(
            ops.scpml_coeffs(5, (2, 2),
                             ops.PmlParams(w_eff=10.),
                             axis=0,
                             transpose=True), (-1, )))

  def test_curl(self):
    def _point_field(axis):
      '''All zero-field except for central element for component `axis`.'''
      z = onp.zeros((1, 1, 3, 3, 3), dtype=onp.complex128)
      f = onp.zeros_like(z)
      f[0, 0, 1, 1, 1] = 1
      return tuple(f if i == axis else z for i in range(3))

    curl = functools.partial(ops.curl,
                             ths=((1, 1), (1, 1), (1, 1)),
                             pml_params=ops.PmlParams())
    self.assertEqual(len(curl(_point_field(0))), 3)
    self.assertEqual(curl(_point_field(0))[0].shape, (1, 1, 3, 3, 3))
    self.assertEqual(curl(_point_field(0))[1][0, 0, 1, 1, 1], 1)
    self.assertEqual(curl(_point_field(0))[2][0, 0, 1, 1, 1], -1)
    onp.testing.assert_array_almost_equal(
        curl(_point_field(0))[1][0, 0, 1, 1, 2], (-0.13793103 - 0.3448276j))
    onp.testing.assert_array_almost_equal(
        curl(_point_field(0))[2][0, 0, 1, 2, 1], (0.13793103 + 0.3448276j))
    self.assertEqual(
        curl(_point_field(1), transpose=True)[0][0, 0, 1, 1, 1], 1)
    self.assertEqual(
        curl(_point_field(1), transpose=True)[2][0, 0, 1, 1, 1], -1)
    onp.testing.assert_array_almost_equal(
        curl(_point_field(1, ), transpose=True)[0][0, 0, 1, 1, 0],
        (-0.13793103 - 0.3448276j))
    onp.testing.assert_array_almost_equal(
        curl(_point_field(1, ), transpose=True)[2][0, 0, 0, 1, 1],
        (0.13793103 + 0.3448276j))

  def test_operator(self):
    input = np.eye(375, dtype=np.complex128)
    input = np.reshape(input, (375, 3, 5, 5, 5))
    input = vecfield.VecField(*np.split(input, 3, axis=1))

    z = vecfield.VecField(*(np.ones((1, 1, 5, 5, 5), np.complex128)
                            for _ in range(3)))

    pml_params = ops.PmlParams(w_eff=0.3)
    pre, inv_pre = ops.preconditioners((5, 5, 5),
                                       ths=((2, 2), ) * 3,
                                       pml_params=pml_params)

    A = ops.operator(input, z, pre, inv_pre, ((2, 2), (2, 2), (2, 2)),
                     pml_params)
    A = np.reshape(np.concatenate(A, axis=1), (375, 375))
    onp.testing.assert_array_almost_equal(A, np.transpose(A))


if __name__ == '__main__':
  unittest.main()
