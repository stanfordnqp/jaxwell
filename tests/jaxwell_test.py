# TODO: Remove.
from jax.config import config
config.update("jax_enable_x64", True)

from jaxwell import cocg, jaxwell, operators, vecfield
import numpy as onp
import unittest


class TestJaxwell(unittest.TestCase):
  def test_loop_fns(self):
    b = onp.zeros((1, 1, 10, 10, 10), onp.complex128)
    b[0, 0, 5, 5, 5] = 1.
    b = vecfield.VecField(0 * b, 0 * b, b)
    z = vecfield.zeros((1, 1, 10, 10, 10))
    x, errs = jaxwell.solve(z,
                            b,
                            ths=((10, 10), ) * 3,
                            pml_params=operators.PmlParams(w_eff=0.3),
                            max_iters=1)
    self.assertIsInstance(x, vecfield.VecField)
    self.assertEqual(x.shape, (1, 1, 10, 10, 10))
    self.assertEqual(len(errs), 1)
    self.assertAlmostEqual(errs[0], 35.25115523)

  def test_adjoint(self):
    b = onp.zeros((1, 1, 10, 10, 10), onp.complex128)
    b[0, 0, 5, 5, 5] = 1.
    b = vecfield.VecField(0 * b, 0 * b, b)
    z = vecfield.zeros((1, 1, 10, 10, 10))
    x, errs = jaxwell.solve(z,
                            b,
                            ths=((10, 10), ) * 3,
                            pml_params=operators.PmlParams(w_eff=0.3),
                            max_iters=1,
                            adjoint=True)
    self.assertAlmostEqual(errs[0], 0.01358992)


if __name__ == '__main__':
  unittest.main()
