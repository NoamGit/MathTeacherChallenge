import pandas as pd
from models.SIM import SIM
import utils

math_train = pd.read_json(r"..\Data\dolphin-number_word_std\number_word_std.dev.json")
math_test = pd.read_json(r"..\Data\dolphin-number_word_std\number_word_std.test.json")

model = SIM()
model.fit(math_train)

print(f'equation score on train: {model.equation_score(math_train)}')
print(f'equation score on test: {model.equation_score(math_test)}')

print(f'result score on train: {model.result_score(math_train,frac=0.1,verbose=False)}')

acc, df = model.result_score(math_test, frac=0.1, verbose=True,output_errors=True)
print(f'result score on test: {acc}')
df.to_csv(r'..\results\test_error_analysis.csv',index=False)
