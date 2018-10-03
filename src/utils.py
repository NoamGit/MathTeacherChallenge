from typing import List

from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols
from sympy.solvers import solve
from sympy.solvers.solveset import linsolve

# region Utilities
def solve_eq_string(math_eq_format: List[str]):
    """
    >>  ["unkn: x,y",
    >>  "equ: x + 3 = y",
    >>  "equ: 2*y + 12 = 5*x"]
    :param math_eq_format:
    :return:
    """
    var_str = math_eq_format[0].split('unkn: ')[-1]
    eq_list = []
    for eq in math_eq_format[1:]:

        # remove whitespaces and move to onesided equation
        rhs,lhs = eq.split('equ: ')[-1].replace(' ','').split('=')
        if len(lhs) > len(rhs):
            lhs,rhs=rhs,lhs

        # oneside handling with negation
        if (lhs[:2] =='-(' and lhs[-1] == ')') or (lhs[0] == '-'):
            lhs[0] = '+'
        else:
            lhs = '-' + lhs

        eq_str = rhs+lhs
        eq_list += [parse_expr(eq_str, evaluate=True)]

    sol = linsolve(eq_list, tuple(symbols(var_str)))
    return list(sol)[0]

    # extracts symbols from
# endregion
