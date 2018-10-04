import pandas as pd
import numpy as np
import re

ALL_DATA_PATH = r"..\Data\additional_train_data\AllData.json"

def to_numeric_results(ans_list):
    return [float(x) for x in ans_list]

def infer_query_vars(qv,leq):
    if qv is None or qv is np.nan:
        qv = [re.sub(r'[^A-Za-z]+', '', leq[0])]
    return ['unkn: ' + ','.join(qv)]

def to_dolphin_equation(l_equation, i_queryvars):
    unkn = infer_query_vars(i_queryvars,l_equation)
    return unkn + ['equ: ' + l_equation[0]]


def load_alldata():
    df = pd.read_json(ALL_DATA_PATH)
    df = df[~df['lEquations'].isnull()]
    df['ans'] = df['lSolutions'].map(to_numeric_results)
    df['ans_simple'] = df['ans']
    df['equantions'] = df.apply(lambda row: to_dolphin_equation(row['lEquations'],row['lQueryVars']),axis=1)
    df['text'] = df['sQuestion']
    return df[['text','ans','ans_simple','equantions']]

if __name__ == '__main__':
    df = load_alldata()
    print(df)