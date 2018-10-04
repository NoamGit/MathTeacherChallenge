import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from numberMapping import number_parsing, test_number_parsing

import utils

class SIM(NearestNeighbors):
    def __init__(self, **kwargs):
        super(SIM, self).__init__(**kwargs)

    def fit(self, corpus_df, y=None):
        '''
        fit a tf-idf model and then a nearest neighbor model
        Args:
            corpus_df: a Dataframe, where in column 'text' is the equation and in column 'numbers' are the list of numbers
            y: ignored

        Returns:
            model after training
        '''
        # extract numbers
        corpus_df[['equations_symbol', 'numbers', 'text_symbol', 'var_list', 'text_num_list']] = corpus_df.apply(
            lambda problem: pd.Series(number_parsing(problem['equations'], problem['text'])), axis=1)

        # save parameters
        self.train_df = corpus_df.copy()
        self.num_neighbors = len(self.train_df)
        self.num_variables = self.train_df['var_list'].map(len).values

        # tfidf
        self.tfidf_model = TfidfVectorizer()
        corpus = self.train_df['text_symbol'].values
        self.tfidf_model.fit(corpus)
        X = self.tfidf_model.transform(corpus)

        # nearest neighbors
        super(SIM, self).fit(X)

        return self

    def predict(self, corpus_df):
        '''
        Receives the data and predicts the equations only using the word problem text itself

        Args:
            corpus_df: dataset

        Returns:
        corpus_df with additional columns:
            'final_neighbor_prediction' - the nearest neighbor in from the train set
            'predicted_equations' - the new equations after inserting the current numbers
        '''
        # extract numbers
        corpus_df[['text_symbol', 'numbers']] = corpus_df.apply(lambda row: pd.Series(test_number_parsing(row['text'])),
                                                         axis=1)

        # tfidf
        X = self.tfidf_model.transform(corpus_df['text_symbol'].values)

        # nearest neighbors
        neighbors_pred = super(SIM, self).kneighbors(X, n_neighbors=self.num_neighbors, return_distance=False)
        corpus_df['neighbors_prediction'] = [cur_neighbors_pred for cur_neighbors_pred in neighbors_pred]

        # only neighbors with the same numbers
        final_neighbor_pred = []
        # correct = []
        for i,(_, problem) in enumerate(corpus_df.iterrows()):
            relevant_neighbors = problem['neighbors_prediction'][
                self.num_variables[problem['neighbors_prediction']] == len(problem['numbers'])]
            # if i != relevant_neighbors[0]:
            #     correct.append(False)
            # else:
            #     correct.append(True)
            if len(relevant_neighbors) > 0:
                final_neighbor_pred.append(relevant_neighbors[0])
            else:
                final_neighbor_pred.append(problem['neighbors_prediction'][0])
        corpus_df['final_neighbor_prediction'] = final_neighbor_pred
        # corpus_df['correct'] = correct

        # transform to equations
        predicted_equations = []
        for _, problem in corpus_df.iterrows():
            equations = self.train_df['equations_symbol'].iloc[problem['final_neighbor_prediction']]
            var_list = self.train_df['var_list'].iloc[problem['final_neighbor_prediction']]
            if len(var_list) != len(problem['numbers']):
                predicted_equations.append(equations)
                continue

            new_equations = [equations[0]]
            for equation in equations[1:]:
                for i, var in enumerate(var_list):
                    if var[1] == 'v':
                        continue
                    equation = equation.replace(var, str(problem['numbers'][i]))
                new_equations.append(equation)
            predicted_equations.append(new_equations)

        corpus_df['predicted_equations'] = predicted_equations

        return corpus_df

    def result_score(self, corpus_df, frac=1, verbose=False, output_errors=False):

        def solve(problem):
            try:
                a = utils.solve_eq_string(problem["predicted_equations"], integer_flag=utils.is_number(problem["text"]))
                return a
            except Exception as e:
                #print(e)
                return []

        corpus_df = self.predict(corpus_df)

        error_list = []
        correct, total = 0, 0
        for k,problem in corpus_df.sample(frac=frac).iterrows():
            pred_ans = solve(problem)
            real_ans = problem['ans_simple']
            if utils.is_same_result(real_ans, pred_ans):
                correct += 1
            else:
                error_list += [(k, ';'.join(problem['equations']).replace('equ:', ''),
                                ';'.join(problem['predicted_equations']).replace('equ:', ''), problem['text'])]

            total += 1
            if verbose: print(correct,total,correct/total)

        # reals_ans = corpus_df['ans_simple']
        # preds_ans = corpus_df.apply(solve, axis=1)
        # for real_ans,pred_ans in zip(reals_ans,preds_ans):
        #     if utils.is_same_result(real_ans,pred_ans):
        #         correct += 1
        #     total += 1

        if output_errors:
            return correct / total, pd.DataFrame(error_list, columns=['ind', 'equations', 'predicted_equations','text'])
        else:
            return correct / total

    def equation_score(self, corpus_df, output_errors=False):
        corpus_df = self.predict(corpus_df)

        not_correct, total = 0, 0
        error_list = []
        for k, problem in corpus_df.iterrows():
            total += 1
            for real, pred in zip(problem['equations'], problem['predicted_equations']):
                if pred.replace(' ', '') != real.replace(' ', ''):
                    if output_errors:
                        error_list += [(k, ';'.join(problem['equations']).replace('equ: ', ''), ';'.join(problem['predicted_equations']).replace('equ: ', ''), problem['correct'],problem['text'])]
                    not_correct += 1
                    break
        if output_errors:
            return (total - not_correct) / total, pd.DataFrame(error_list, columns=['ind', 'real', 'parsed','right_neighb','text'])
        else:
            return (total - not_correct) / total
