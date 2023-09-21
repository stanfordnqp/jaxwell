# TODO: Remove.
import jax
import jax.numpy as np
import unittest
import numpy as onp
from jaxwell import fdfd, operators, vecfield
from jax.config import config
config.update("jax_enable_x64", True)
#config.update("jax_debug_nans", True)


class TestJaxwell(unittest.TestCase):
  def setUp(self):
    self.b = onp.zeros((10, 10, 10), onp.complex128)
    self.b[5, 5, 5] = 1.
    self.b = (0 * self.b, 0 * self.b, self.b)
    self.z = (onp.zeros((10, 10, 10)),) * 3
    self.params = fdfd.Params(pml_ths=((10, 10), ) * 3,
                              pml_omega=0.3,
                              max_iters=1)

  def test_solve(self):
    x, err = fdfd.solve(self.params, self.z, self.b)
    self.assertAlmostEqual(err, 35.25115523)

  def test_norm_grad(self):
    def foo(a):
      return np.linalg.norm(a)
    b = onp.zeros((10, 10, 10), onp.complex128)
    grad_x = jax.grad(foo)(b)
    print(grad_x)

  def test_vecfield_grad(self):
    def foo(a):
      return vecfield.norm(a)
    
    grad_a = jax.grad(foo)(self.b)
    print(grad_a)

  def test_grad(self):

    def foo(z, b):
      res, err = fdfd.solve(self.params, z, b)
      norm = vecfield.norm(res)
      return norm

    grad_z, grad_b = jax.grad(foo, (0, 1))(self.z, self.b)

    # Used as pseudo-golden tests.
    self.assertAlmostEqual(sum(np.nansum(g) for g in grad_z), -1484.39503965)
    self.assertAlmostEqual(sum(np.nansum(g)
                               for g in grad_b), 34.22197829+8.38256152e-15j)

  def test_loop_fns(self):
    x, errs = fdfd.solve_impl(self.z, self.b, params=self.params)
    self.assertEqual(len(x), 3)
    self.assertEqual(x[0].shape, (10, 10, 10))
    self.assertEqual(len(errs), 1)
    self.assertAlmostEqual(errs[0], 35.25115523)

  def test_adjoint(self):
    x, errs = fdfd.solve_impl(self.z, self.b, adjoint=True, params=self.params)
    self.assertAlmostEqual(errs[0], 0.01358992)


if __name__ == '__main__':
  unittest.main()
