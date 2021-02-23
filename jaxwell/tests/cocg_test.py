# TODO: Remove.
import unittest
import numpy as onp
from jaxwell import operators, cocg, vecfield
from jax.config import config
config.update("jax_enable_x64", True)


class TestSolver(unittest.TestCase):
  def test_fns(self):
    shape = (1, 1, 10, 10, 10)
    ths = ((2, 2),) * 3
    pml_params = operators.PmlParams(w_eff=0.3)

    pre, inv_pre = operators.preconditioners(shape[2:], ths, pml_params)
    def A(x, z): return operators.operator(x, z, pre, inv_pre, ths, pml_params)
    b = onp.zeros(shape, onp.complex128)
    b[0, 0, 5, 5, 5] = 1.
    b = vecfield.VecField(0 * b, 0 * b, b)
    z = vecfield.zeros(shape)

    init, iter = cocg.solver(A, b, eps=1e-6)

    p, r, x, term_err = init(z, b)
    onp.testing.assert_array_equal(p, r)
    onp.testing.assert_array_equal(x, onp.zeros_like(x))
    self.assertEqual(term_err, 1e-6)

    p, r, x, err = iter(p, r, x, z)
    self.assertAlmostEqual(err, 0.8660254)


if __name__ == '__main__':
  unittest.main()
