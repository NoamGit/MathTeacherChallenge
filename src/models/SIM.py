import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from numberMapping import number_parsing

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
        corpus_df[['equations', 'numbers', 'text']] = corpus_df.apply(
            lambda row: pd.Series(number_parsing(row['equations'],row['text'])),axis=1)

        # save parameters
        self.train_df = corpus_df
        self.num_neighbors = len(self.train_df)
        self.num_numbers = self.train_df['numbers'].map(len).values

        # tfidf
        self.tfidf_model = TfidfVectorizer()
        corpus = self.train_df['text'].values
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
        corpus_df[['equations', 'numbers', 'text']] = corpus_df.apply(
            lambda row: pd.Series(number_parsing(row['equations'], row['text'])), axis=1)

        # tfidf
        X = self.tfidf_model.transform(corpus_df['text'].values)

        # nearest neighbors
        neighbors_pred = super(SIM, self).kneighbors(X, n_neighbors=self.num_neighbors, return_distance=False)
        corpus_df['neighbors_prediction'] = [pd.Series(cur_neighbors_pred) for cur_neighbors_pred in neighbors_pred]

        # only neighbors with the same numbers
        final_neighbor_pred = []
        for _,row in corpus_df.iterrows():
            relevant_neighbors = row['neighbors_prediction'][self.num_numbers[row['neighbors_prediction']] == len(row['numbers'])]
            final_neighbor_pred.append(relevant_neighbors[0])
        corpus_df['final_neighbor_prediction'] = final_neighbor_pred

        # transform to equations
        predicted_equations = []
        for _, row in corpus_df.iterrows():
            equations = self.train_df['equations'].iloc[row['final_neighbor_prediction']]

            new_equations = [equations[0]]
            for equation in equations[1:]:
                for i,number in enumerate(row['numbers']):
                    equation.replace(f'$n{i}',number)
                new_equations.append(equation)
            predicted_equations.append(new_equations)

        corpus_df['predicted_equations'] = predicted_equations

        return corpus_df




