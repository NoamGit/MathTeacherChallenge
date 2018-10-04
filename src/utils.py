from typing import List

import wolframalpha
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations, convert_xor,implicit_multiplication
from sympy import symbols,var,Symbol
from sympy import solveset, S
from sympy.sets.sets import EmptySet
from sympy.solvers.solveset import linsolve
from sympy.solvers import solve
from numpy import *
import numpy as np
from pprint import pprint
from Timeout import Timeout, timeout

IS_NUM_DICT = ['integer','consecutive']
TRANSFORMATION = standard_transformations + (implicit_multiplication,convert_xor,)
WOLF_CLIENT = wolframalpha.Client(app_id="23XUAT-H2875HHEEX")
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

# def solve_with_wolfram(input_str:str):
#     sol_wolfram = WOLF_CLIENT.query(input_str.split('equ: ')[-1].replace(' ', ''))
#     # TODO: parse wolfram solution
#     sol_wolf = [cur_result['subpod'] for cur_result in list(sol_wolfram.results) if
#                 cur_result['@id'] in ['IntegerSolution', 'IntegerSolutions', 'Solutions', 'Solution', 'Results', 'Result']][0]
#     sol_wolf = [cur_sol_wolf['plaintext'].replace(' ', '') for cur_sol_wolf in sol_wolf]
#
#     final_results = [result.split('=') for result in sol_wolf]
#     return [{k: float(v)} for k,v in final_results]

def solve_with_wolfram(input_str:str):
    sol_wolfram = WOLF_CLIENT.query(input_str.split('equ: ')[-1].replace(' ', ''))
    # TODO: parse wolfram solution
    if 'Solutions' in sol_wolfram.details:
        sol_wolf = sol_wolfram.details['Solutions']
    elif 'Solution' in sol_wolfram.details:
        sol_wolf = sol_wolfram.details['Solution']
    elif 'Roots' in sol_wolfram.details:
        sol_wolf = sol_wolfram.details['Roots']
    elif 'Root' in sol_wolfram.details:
        sol_wolf = sol_wolfram.details['Root']
    else:
        return []

    results = [result.split('=') for result in sol_wolf.replace(' ', '').split(',')]
    return {k: float(eval(v)) for k,v in results}

def solve_eq_string(math_eq_format: List[str], integer_flag=False):
    """
    >>  ["unkn: x,y",
    >>  "equ: x + 3 = y",
    >>  "equ: 2*y + 12 = 5*x"]
    :param math_eq_format:
    :return:
    """
    kw_parser = dict(evaluate=True, transformations=TRANSFORMATION)
    do_wolfram = use_wolfram(math_eq_format[1:])
    var_str = math_eq_format[0].replace(' ','').split('unkn:')[-1]
    sym_var = tuple()
    # TODO: fix hack of constraining integer solution
    for v in var_str.split(','):
        var = Symbol(v)
        # var = Symbol(v, integer=integer_flag) if integer_flag else Symbol(v)
        sym_var += (var,)
    if not do_wolfram:
        parse_eq_list = []
        for eq in math_eq_format[1:]:
            # remove whitespaces and move to onesided equation
            rhs,lhs = eq.split('equ:')[-1].replace(' ','').split('=')
            parse_eq_list += [parse_expr(lhs, **kw_parser) * -1 + parse_expr(rhs, **kw_parser)]

        if len(sym_var)<=3 and '^' not in ';'.join(math_eq_format[1:]).replace('equ:','').replace(' ',''):
            sol = solve(parse_eq_list)
            if sol == []:
                do_wolfram = True
        else:
            do_wolfram = True
    if do_wolfram:
        eq_wolf_format = ';'.join(math_eq_format[1:]).replace('equ:','').replace(' ','')
        sol = solve_with_wolfram(eq_wolf_format)

    if isinstance(sol,dict):
        eval_sol = [parse_expr(v, **kw_parser).evalf(subs=dict(sol)) for v in var_str.split(',')]
        if sol == []:
            return []
        elif integer_flag and not all([y.is_integer for x, y in sol.items()]):
            return []
        else:
            return eval_sol
    elif isinstance(sol,list):
        eval_sol = []
        for cur_sol in sol:
            eval_itr = [parse_expr(v, **kw_parser).evalf(subs=cur_sol) for v in var_str.split(',')]
            if sol == []:
                continue
            elif integer_flag and not all([y.is_integer for x,y in cur_sol.items() if y.is_number]):
                continue
            else:
                eval_sol += [eval_itr]
        return eval_sol

    # print(solve(parse_eq_list, sym_var))
    # print(solve(parse_eq_list, (x,y)))
    # print(solve([-x + y - 4, 2*x - 5*y + 11], sym_var))
    # print(solve([-x + y - 4, 2*x - 5*y + 11], (x,y)))

def sparse_binary_jaccard(v1, v2):
    v1_nz = set(v1.nonzero()[1])
    v2_nz = set(v2.nonzero()[1])
    return len(v1_nz.intersection(v2_nz))/len(v1_nz.union(v2_nz))

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

def is_same_result(real_ans,pred_ans):
    if len(pred_ans)>0 and isinstance(pred_ans,list) and isinstance(pred_ans[0],list):
        for cur_pred_ans in pred_ans:
            if are_close(cur_pred_ans,real_ans):
                return True
    elif are_close(pred_ans,real_ans):
        return True
    return False

def are_close(l1,l2):
    try:
        res = len(l1) == len(l2) and np.allclose(np.sort(l1).astype(float), np.sort(l2).astype(float),rtol=0.001)
        return res
    except Exception as e:
        #print(e)
        return False

def use_wolfram(equations):
    use_wolfram = False
    for equation in equations:
        if '<' in equation or '>' in equation:
            use_wolfram = True
        elif '=' not in equation:
            use_wolfram = True
    return use_wolfram
