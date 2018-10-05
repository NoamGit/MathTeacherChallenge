import pandas as pd
from models.SIM import SIM
import utils
from load_data import load_alldata

use_additional_data = True
use_ans = True

math_train = pd.read_json(r"..\Data\dolphin-number_word_std\number_word_std.dev.json")
if use_additional_data:
    additional_data = load_alldata()
    math_train = pd.concat([math_train[['text','ans','ans_simple','equations']],additional_data])
math_test = pd.read_json(r"..\Data\dolphin-number_word_std\number_word_std.test.json")

model = SIM()
model.fit(math_train)

print(f'equation score on train: {model.equation_score(math_train)}')
print(f'equation score on test: {model.equation_score(math_test)}')

print(f'result score on train: {model.result_score(math_train,frac=0.1,verbose=False,use_ans=use_ans)}')
print(f'result score on test: {model.result_score(math_test,frac=1,verbose=True,use_ans=use_ans)}')

#df.to_csv(r'..\results\error_analysis\all_test_analysis.csv',index=False)
