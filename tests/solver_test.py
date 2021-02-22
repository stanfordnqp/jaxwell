# TODO: Remove.
from jax.config import config
config.update("jax_enable_x64", True)

from jaxwell import operators, solver, vecfield
import numpy as onp
import unittest


class TestSolver(unittest.TestCase):
  def test_loop_fns(self):
    shape = (1, 1, 10, 10, 10)
    ths = ((2, 2),) * 3
    pml_params = operators.PmlParams(w_eff=0.3)

    pre, inv_pre = operators.preconditioners(shape[2:], ths, pml_params)
    A = lambda x, z: operators.operator(x, z, pre, inv_pre, ths, pml_params)
    b = onp.zeros(shape, onp.complex128)
    b[0, 0, 5, 5, 5] = 1.
    b = vecfield.VecField(0 * b, 0 * b, b)
    z = vecfield.zeros(shape)

    loop_init, loop_iter = solver.loop_fns(A, b, eps=1e-6)

    p, r, x, term_err = loop_init(z, b)
    onp.testing.assert_array_equal(p, r)
    onp.testing.assert_array_equal(x, onp.zeros_like(x))
    self.assertEqual(term_err, 1e-6)

    p, r, x, err = loop_iter(p, r, x, z)
    self.assertAlmostEqual(err, 0.8660254)


if __name__ == '__main__':
  unittest.main()
