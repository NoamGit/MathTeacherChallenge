from unittest import TestCase
import pandas as pd
from utils import solve_eq_string, is_number, are_close
import numpy as np

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

        # bad ii = 4, 74
        ii = 4
        problem = test.iloc[ii]
        res = solve_eq_string(problem["equations"],integer_flag=is_number(problem["text"]))
        # more than 1 possible solution

        is_in = False
        if isinstance(res,list) and isinstance(res[0],list):
            for ans in res:
                if are_close(ans,problem["ans_simple"]):
                    is_in = True
            self.assertTrue(is_in)
        else:
            self.assertTrue(are_close(res,problem["ans_simple"]))
