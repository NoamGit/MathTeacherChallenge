import pandas as pd
from models.models import SIM

math_train = pd.read_json(r"../data/dev_data.json")
math_test = pd.read_json(r"../data/test_data.json")

## model
model = SIM()
model.fit(math_train)

## print evaluation result
print(f'result score on train: {model.score(math_train,frac=0.1,verbose=False,use_ans=True)}')
print(f'result score on test: {model.score(math_test,frac=1,verbose=True,use_ans=True)}')
