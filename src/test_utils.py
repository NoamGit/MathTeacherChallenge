from unittest import TestCase

from utils import solve_eq_string


class TestUtils(TestCase):

    def test_solve_eq_string(self):
        eq = ["unkn: x,y", "equ: x + 3 = y", "equ: 2*y + 12 = 5*x"]
        res = solve_eq_string(eq)
        self.assertEqual(res,(6,9))