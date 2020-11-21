# TODO: Remove.
from jax.config import config
config.update("jax_enable_x64", True)

import numpy as onp
import operators
import solver
import unittest
import vecfield


class TestSolver(unittest.TestCase):
  def test_loop_fns(self):
    loop_init, loop_iter = solver.loop_fns(
        shape=(1, 1, 10, 10, 10),
        ths=((2, 2), ) * 3,
        pml_params=operators.PmlParams(w_eff=0.3),
        eps=1e-6)
    b = onp.zeros((1, 1, 10, 10, 10), onp.complex128)
    b[0, 0, 5, 5, 5] = 1.
    b = vecfield.VecField(0 * b, 0 * b, b)
    z = vecfield.zeros((1, 1, 10, 10, 10))
    p, r, x, term_err = loop_init(z, b)
    onp.testing.assert_array_equal(p, r)
    onp.testing.assert_array_equal(x, onp.zeros_like(x))
    self.assertEqual(term_err, 1e-6)

    p, r, x, err = loop_iter(p, r, x, z)
    self.assertAlmostEqual(err, 0.8660254)


if __name__ == '__main__':
  unittest.main()
