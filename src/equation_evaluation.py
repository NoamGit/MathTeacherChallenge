import pandas as pd
import utils

math_test = pd.read_json(r"..\Data\dolphin-number_word_std\number_word_std.test.json")

def solve(problem):
    if problem["equations"] == ['unkn: x,y', 'equ: x+y>12', 'equ: x+10=2*y']:
        a=1
    try:
        return utils.solve_eq_string(problem["equations"])
    except Exception as e:
        print(problem["equations"])
        print('****')
        print(e)
        print()
        return []

reals_ans = math_test['ans_simple']
preds_ans = math_test.apply(solve,axis=1)

correct, total = 0, 0
for real_ans,pred_ans in zip(reals_ans,preds_ans):
    if utils.is_same_result(real_ans,pred_ans):
        correct += 1
    total += 1

print(correct,total,correct/total)
