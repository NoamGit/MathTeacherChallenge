from unittest import TestCase
import pandas as pd
from utils import solve_eq_string, is_number


class TestUtils(TestCase):

    def test_solve_eq_string(self):
        eq = ["unkn: x,y", "equ: x + 3 = y", "equ: 2*y + 12 = 5*x"]
        res = solve_eq_string(eq)
        self.assertEqual(res,[6,9])

    def test_solve_eq_all(self):
        train = pd.read_json('../Data/dolphin-number_word_std/number_word_std.dev.json')
        test = pd.read_json('../Data/dolphin-number_word_std/number_word_std.test.json')

        # train["equations"].apply(lambda x:set(solve_eq_string(x)))
        # train["ans"].apply(.iloc[6]

        # bad ii = 4
        ii = 344
        problem = test.iloc[ii]
        res = solve_eq_string(problem["equations"],integer_flag=is_number(problem["text"]))
        # if isinstance(res,List[list])
        self.assertAlmostEqual(list(set(res)), list(set(problem["ans_simple"])))
