from typing import List

from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols,var,Symbol
from sympy import solveset, S
from sympy.sets.sets import EmptySet
from sympy.solvers.solveset import linsolve
from sympy.solvers import solve

IS_NUM_DICT = ['integer','consecutive',]
# region Utilities

def switch_sign(txt:str):
    minus_pos = txt.find('-')
    plus_pos = txt.find('+')
    txt_list = list(txt)
    if minus_pos != -1:
        txt_list[minus_pos] = '+'
    if plus_pos != -1:
        txt_list[plus_pos] = '-'
    return ''.join(txt_list)

def solve_eq_string(math_eq_format: List[str], integer_flag=False):
    """
    >>  ["unkn: x,y",
    >>  "equ: x + 3 = y",
    >>  "equ: 2*y + 12 = 5*x"]
    :param math_eq_format:
    :return:
    """
    var_str = math_eq_format[0].split('unkn: ')[-1]
    sym_var = tuple()
    # TODO: fix hack of constraining integer solution
    for v in var_str.split(','):
        var = Symbol(v)
        # var = Symbol(v, integer=integer_flag) if integer_flag else Symbol(v)
        sym_var+=(var,)

    eq_list = []
    parse_eq_list = []
    for eq in math_eq_format[1:]:

        # remove whitespaces and move to onesided equation
        rhs,lhs = eq.split('equ: ')[-1].replace(' ','').split('=')
        parse_eq_list += [parse_expr(lhs, evaluate=True) * -1 + parse_expr(rhs, evaluate=True)]

    from sympy import nsolve
    sol = solve(parse_eq_list, sym_var)
    if isinstance(sol,dict):
        eval_sol = [parse_expr(v).evalf(subs=dict(sol)) for v in var_str.split(',')]
        if sol == []:
            return []
        elif integer_flag and not all([y.is_integer for x, y in sol.items()]):
            return []
        else:
            return eval_sol
    elif isinstance(sol,list):
        eval_sol = []
        for s in sol:
            sol_dict = dict(zip([s.strip() for s in var_str.split(',')],s))
            eval_itr = [parse_expr(v).evalf(subs=sol_dict) for v in var_str.split(',')]
            if sol == []:
                continue
            elif integer_flag and not all([y.is_integer for x,y in sol_dict.items() if y.is_number]):
                continue
            else:
                eval_sol += [eval_itr]
        return eval_sol

    # print(solve(parse_eq_list, sym_var))
    # print(solve(parse_eq_list, (x,y)))
    # print(solve([-x + y - 4, 2*x - 5*y + 11], sym_var))
    # print(solve([-x + y - 4, 2*x - 5*y + 11], (x,y)))


def is_number(query_text:str):
    query_text = query_text.lower()
    if any([t in query_text for t in IS_NUM_DICT]):
        return True
    else:
        return False

def parse_ans_col(ans: str):
    if ans == 'ans_no_result':
        return []
    else:
        ans.replace
        if '|' not in ans:
            pass

# endregion
