# TODO: Remove.
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np
from jaxwell import vecfield as vf
from jaxwell.vecfield import VecField
import numpy as onp
import unittest


class TestVecField(unittest.TestCase):
  def test_zeros(self):
    self.assertEqual(vf.zeros((10, 20, 30)).shape, (10, 20, 30))
    self.assertEqual(vf.zeros((10, 20, 30)).dtype, np.complex128)

  def test_as_array(self):
    self.assertIsInstance(
        VecField(*(onp.zeros(5) for _ in range(3))).as_array().x, np.ndarray)

  def test_add(self):
    self.assertEqual(VecField(1, 2, 3) + VecField(4, 5, 6), VecField(5, 7, 9))

  def test_sub(self):
    self.assertEqual(
        VecField(1, 2, 3) - VecField(4, 5, 6), VecField(-3, -3, -3))

  def test_mul(self):
    self.assertEqual(
        VecField(1, 2, 3) * VecField(4, 5, 6), VecField(4, 10, 18))

  def test_scalar_mul(self):
    self.assertEqual(2 * VecField(1, 2, 3), VecField(2, 4, 6))

  def test_dot(self):
    self.assertEqual(
        vf.dot(VecField(*((1 + 1j) * np.ones(5) for _ in range(3)), ),
               VecField(*((1 + 1j) * np.ones(5) for _ in range(3)), )), 30j)

  def test_norm(self):
    self.assertEqual(vf.norm(VecField(*(3j * np.ones(3) for _ in range(3)))),
                     9)

  def test_conj(self):
    self.assertEqual(vf.conj(VecField(1 + 1j, 2j, 3)), VecField(1 - 1j, -2j, 3))

  def test_real(self):
    self.assertEqual(vf.real(VecField(1 + 1j, 2j, 3)), VecField(1, 0, 3))

if __name__ == '__main__':
  unittest.main()
