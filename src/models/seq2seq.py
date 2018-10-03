import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from gensim.models import Word2Vec

from keras.models import Model
from keras.layers import Input, LSTM, Dense, RepeatVector

class seq2seq(NearestNeighbors):
    def __init__(self, latent_dim=128, **kwargs):
        self.latent_dim = latent_dim
        super(seq2seq, self).__init__(**kwargs)

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

        # save parameters
        self.train_df = corpus_df
        self.num_neighbors = len(self.train_df)
        self.num_numbers = self.train_df['numbers'].map(len).values

        # prepare data X,Y
        corpus = self.train_df['text'].values
        self.word2vec_model = Word2Vec(corpus, size=self.latent_dim, window=5, min_count=1)
        X, self.max_X_length = self.get_X(corpus, self.word2vec_model)
        Y, self.max_Y_length = self.get_Y(self.train_df['equations'])

        # seq2seq
        inputs = Input(shape=(self.max_X_length, self.latent_dim))
        encoded = LSTM(self.latent_dim)(inputs)

        decoded = RepeatVector(self.max_Y_length)(encoded)
        decoded = LSTM(Y.shape[1], return_sequences=True)(decoded)

        self.sequence_autoencoder = Model(inputs, decoded)
        self.encoder = Model(inputs, encoded)

        self.sequence_autoencoder.compile(loss='categorical_crossentropy', optimizer='adam')

        self.sequence_autoencoder.fit(X,Y)

        # nearest neighbors
        Z = self.encoder.predict(X)
        super(seq2seq, self).fit(Z)

        return self

    def get_X(self, corpus, word2vec_model, max_length=None):
        if max_length == None:
            max_length = np.max([len(sentence) for sentence in corpus])

        X = np.array([np.concatenate([np.zeros((max_length - len(sentence), self.latent_dim)),
                             np.array([word2vec_model.wv[word] for word in sentence])]) for sentence in corpus])

        return X, max_length

    def get_Y(self, equations, max_length=None):
        equations = [list(';'.join(cur_equations[1:])) for cur_equations in equations]

        new_equations = []
        digits = ['0','1','2','3','4','5','6','7','8','9']
        for equation in equations:
            new_equation = []
            is_number = False
            for char in equation:
                if char == '$':
                    new_equation.append(char)
                    is_number = True
                elif is_number and char in digits+['n']:
                    new_equation[-1] += char
                else:
                    is_number = False
                    new_equation.append(char)
            new_equations.append(new_equation)


        if max_length == None:
            max_length = np.max([len(equation) for equation in new_equations])

        unique_values = np.unique([elm for equation in new_equations for elm in equation])
        # integer encode
        label_encoder = LabelEncoder()
        value_labels = label_encoder.fit_transform(unique_values)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = value_labels.reshape(len(value_labels), 1)
        onehot_encoded = onehot_encoder.fit(integer_encoded)

        labeled_equations = [label_encoder.transform(equation) for equation in new_equations]
        onehot_equations = [onehot_encoded.transform(labeled_equation.reshape(len(labeled_equation), 1)) for labeled_equation in labeled_equations]

        Y = np.array([np.concatenate([np.zeros((max_length - len(onehot_equation), self.latent_dim)),onehot_equations]) for onehot_equation in onehot_equations])

        return Y, max_length

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


        # seq2seq
        X, _ = self.get_X(corpus_df['text'].values, self.word2vec_model, max_length=self.max_X_length)
        Z = self.encoder.predict(X)

        # nearest neighbors
        neighbors_pred = super(seq2seq, self).kneighbors(Z, n_neighbors=self.num_neighbors, return_distance=False)
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




