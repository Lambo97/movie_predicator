# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
from contextlib import contextmanager

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline

@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))


def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path,encoding = "ISO-8859-1",dtype=object)





def make_submission(y_predict, user_movie_ids, file_name='submission',
                    date=True):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    y_predict: array [n_predictions]
        The predictions to write in the file. `y_predict[i]` refer to the
        user `user_ids[i]` and movie `movie_ids[i]`
    user_movie_ids: array [n_predictions, 2]
        if `u, m = user_movie_ids[i]` then `y_predict[i]` is the prediction
        for user `u` and movie `m`
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """

    # Naming the file
    if date:
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    # Writing into the file
    with open(file_name, 'w') as handle:
        handle.write('"USER_ID_MOVIE_ID","PREDICTED_RATING"\n')
        for (user_id, movie_id), prediction in zip(user_movie_ids,
                                                 y_predict):

            if np.isnan(prediction):
                raise ValueError('The prediction cannot be NaN')
            line = '{:d}_{:d},{}\n'.format(int(user_id), int(movie_id), prediction)
            handle.write(line)
    return file_name


if __name__ == '__main__':
    prefix = 'data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_user_movie_pairs = load_from_csv(os.path.join(prefix,
                                                           'data_train.csv'))
    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))
    training_users_attributes = load_from_csv(os.path.join(prefix, 'data_user.csv'))
    training_movies_attributes = load_from_csv(os.path.join(prefix, 'data_movie.csv'))
    
    # Join all the features together and drop the less useful ones
    train_data = pd.merge(training_user_movie_pairs,training_movies_attributes,how='left')
    train_data = pd.merge(train_data,training_users_attributes,how='left')
    train_data = train_data.drop(['movie_title','video_release_date','IMDb_URL','zip_code','release_date','movie_id','user_id'], axis=1)
    train_data = pd.get_dummies(train_data,columns=['occupation'])
    train_data = pd.get_dummies(train_data,columns=['gender'])

    # Outputs
    y_ls = training_labels

    #Normalize the data, moslty useful with MLPRegressor
    train_data = normalize(train_data)
    start = time.time()

    # Compute the score of MLPRegressor according to different hidden_layer_sizes
    hidden_layer_sizes = range(100)
    Scores = []
    for i in hidden_layer_sizes:
        model = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(i+1, ), random_state=1)
        with measure_time('Training'):
            print('Training MLPRegressor with '+str(i+1) + 'hidden_layer_sizes')
            scores = cross_val_score(model, train_data, y_ls, cv=3)
            Scores.append(scores.mean())
    plt.plot(hidden_layer_sizes,Scores)
    plt.savefig('MLPRegressor.png')

    # Get the best hyper-parameter to predict the set data
    model = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=( Scores.index(max(Scores))+1, ), random_state=1)
    model.fit(train_data, y_ls)


    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    test_user_movie_pairs = load_from_csv(os.path.join(prefix,
                                                           'data_test.csv'))
    test_data = pd.merge(test_user_movie_pairs,training_movies_attributes,how='left')
    test_data = pd.merge(test_data,training_users_attributes,how='left')
    test_data = test_data.drop(['movie_title', 'video_release_date','IMDb_URL','zip_code','release_date','movie_id','user_id'], axis=1)
    test_data = pd.get_dummies(test_data,columns=['occupation'])
    test_data = pd.get_dummies(test_data,columns=['gender'])

    # Predict
    test_data = normalize(test_data)
    y_pred = model.predict(test_data)

    #Correct the prediction if they are too big or too low
    for i in range(len(y_pred)):
        a = float(y_pred[i])
        y_pred[i] = a
        if a>5:
            y_pred[i]=5
        if a<0:
            y_pred[i]=0
    
    test_user_movie_pairs = test_user_movie_pairs.values.squeeze()
    # Making the submission file
    fname = make_submission(y_pred, test_user_movie_pairs, 'toy_example')
    print('Submission file "{}" successfully written'.format(fname))
