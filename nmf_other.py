import numpy as np
from mf import MF
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPRegressor

class MoviePredicator(BaseEstimator, ClassifierMixin):

    def __init__(self, rating_matrix, K, alpha, beta, iterations, hidden_layer_sizes):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        self.rating_matrix = rating_matrix
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, y):
        matrix_factorization = MF(self.rating_matrix, self.K, self.alpha, self.beta, self.iterations).fit()
        matrix = matrix_factorization.full_matrix()
        X_ls = self.__create_learning_matrices(matrix, X)
        self.mlp = MLPRegressor(hidden_layer_sizes=(self.hidden_layer_sizes, ),solver='lbfgs', alpha=1e-5, random_state=1)
        self.mlp.fit(X_ls, y)

        return self

    def predict(self, X_ts):
        return self.mlp.predict(X_ts)

    def __create_learning_matrices(rating_matrix, user_movie_pairs):
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



    