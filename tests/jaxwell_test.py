# TODO: Remove.
import jax
import unittest
import numpy as onp
from jaxwell import cocg, jaxwell, operators, vecfield
from jax.config import config
config.update("jax_enable_x64", True)


class TestJaxwell(unittest.TestCase):
  def test_grad(self):
    b = onp.zeros((1, 1, 10, 10, 10), onp.complex128)
    b[0, 0, 5, 5, 5] = 1.
    b = vecfield.VecField(0 * b, 0 * b, b)
    z = vecfield.zeros((1, 1, 10, 10, 10))
    params = jaxwell.Params(pml_ths=((10, 10), ) * 3,
                            pml_params=operators.PmlParams(w_eff=0.3),
                            max_iters=1)

    def foo(z, b):
      return vecfield.norm(jaxwell.solve(params, z, b))

    jax.grad(foo, (0, 1))(z, b)

  def test_loop_fns(self):
    b = onp.zeros((1, 1, 10, 10, 10), onp.complex128)
    b[0, 0, 5, 5, 5] = 1.
    b = vecfield.VecField(0 * b, 0 * b, b)
    z = vecfield.zeros((1, 1, 10, 10, 10))
    params = jaxwell.Params(pml_ths=((10, 10), ) * 3,
                            pml_params=operators.PmlParams(w_eff=0.3),
                            max_iters=1)
    x, errs = jaxwell.solve_impl(z, b, params=params)
    self.assertIsInstance(x, vecfield.VecField)
    self.assertEqual(x.shape, (1, 1, 10, 10, 10))
    self.assertEqual(len(errs), 1)
    self.assertAlmostEqual(errs[0], 35.25115523)

  def test_adjoint(self):
    b = onp.zeros((1, 1, 10, 10, 10), onp.complex128)
    b[0, 0, 5, 5, 5] = 1.
    b = vecfield.VecField(0 * b, 0 * b, b)
    z = vecfield.zeros((1, 1, 10, 10, 10))
    params = jaxwell.Params(
        pml_ths=((10, 10), ) * 3,
        pml_params=operators.PmlParams(w_eff=0.3),
        max_iters=1)
    x, errs = jaxwell.solve_impl(z, b, adjoint=True, params=params)
    self.assertAlmostEqual(errs[0], 0.01358992)


if __name__ == '__main__':
  unittest.main()
