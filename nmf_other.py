import numpy as np
import pandas as pd
from mf import MF
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor


class MoviePredicator(BaseEstimator, ClassifierMixin):

    def __init__(self, rating_matrix = None, K = 100):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        """
        self.rating_matrix = rating_matrix
        self.K = K

    def fit(self, X, y):
        if(self.rating_matrix is None):
            matrix_factorization = MF(self.rating_matrix, self.K, 1e-5, 0.02, 5000).fit()
            self.rating_matrix = matrix_factorization.full_matrix()
            np.savetxt("matrix_K{}.csv".format(self.K), self.rating_matrix)
        else:
            self.rating_matrix = load_from_csv("matrix_K{}.csv".format(self.K))
        X_ls = create_learning_matrices(self.rating_matrix, X)
        self.model = MLPRegressor(solver='lbfgs', alpha=1e-5, random_state=1)
        #self.model = GradientBoostingRegressor(max_depth= 2)
        self.model.fit(X_ls, y)

        return self

    def predict(self, X):
        X_ts = create_learning_matrices(self.rating_matrix, X)
        return self.model.predict(X_ts)

def create_learning_matrices(rating_matrix, user_movie_pairs):
    """
    Create the learning matrix `X` from the `rating_matrix`.

    If `u, m = user_movie_pairs[i]`, then X[i] is the feature vector
    corresponding to user `u` and movie `m`. The feature vector is composed
    of `n_users + n_movies` features. The `n_users` first features is the
    `u-th` row of the `rating_matrix`. The `n_movies` last features is the
    `m-th` columns of the `rating_matrix`

    In other words, the feature vector for a pair (user, movie) is the
    concatenation of the rating the given user made for all the movies and
    the rating the given movie receive from all the user.

    Parameters
    ----------
    rating_matrix: matrix [n_users, n_movies]
        The rating matrix. i.e. `rating_matrix[u, m]` is the rating given
        by the user `u` for the movie `m`.
    user_movie_pairs: array [n_predictions, 2]
        If `u, m = user_movie_pairs[i]`, the i-th raw of the learning matrix
        must relate to user `u` and movie `m`

    Return
    ------
    X: array [n_predictions, n_users + n_movies]
        The learning matrix in csr sparse format
    """
    # Feature for users
    user_features = rating_matrix[user_movie_pairs[:, 0]-1]

    # Features for movies
    movie_features = rating_matrix[:, user_movie_pairs[:, 1]].transpose()

    X = np.hstack((user_features, movie_features))
    return X
    
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
    return pd.read_csv(path, delimiter=delimiter).values.squeeze()

    