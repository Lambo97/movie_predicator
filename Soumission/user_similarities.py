import numpy
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import time
import datetime
from contextlib import contextmanager
import pandas as pd
import numpy as np

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

def load_from_csv_list(path, delimiter=','):
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
    D: list
        A list of the data contained in the file
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
            line = '{:d}_{:d},{}\n'.format(user_id, movie_id, prediction)
            handle.write(line)
    return file_name

if __name__ == '__main__':
	prefix = 'data/'

	training_users_attributes = load_from_csv_list(os.path.join(prefix, 'data_user.csv'))

	# Get the features names
	features = list(training_users_attributes.columns.values)

	# Format the user data matrix
	users = np.zeros((911, 1))
	for i in range(len(features))[1:]:
	    if i == 1:
	        users = np.reshape(training_users_attributes[features[i]].tolist(), (911, 1))
	    else:
	    	users = np.hstack((users,np.reshape(training_users_attributes[features[i]].tolist(), (911, 1))))


	nb_user = len(users)
	nb_movie = 1541
	nb_attribute = len(users[0])

	#similar_usersX[i] retunrs the user_ids having X features in common with the user i+1 
	similar_users4 = []
	similar_users3 = []
	similar_users2 = []
	similar_users1 = []
	for i in range(nb_user):
		similar_users4.append([])
		similar_users3.append([])
		similar_users2.append([])
		similar_users1.append([])
		for j in range(nb_user):
			match_score = 0
			if(i!=j):
				for k in range(nb_attribute):
					if(users[i][k]==users[j][k]):
						match_score +=1
				if(match_score==4):
					similar_users4[i].append(j+1)
				if(match_score==3):
					similar_users3[i].append(j+1)
				if(match_score==2):
					similar_users2[i].append(j+1)
				if(match_score==1):
					similar_users1[i].append(j+1)

	training_user_movie_pairs = load_from_csv_list(os.path.join(prefix,
	                                                           'data_train.csv'))

	#user_having_watched_this_movie[i] returns the users having watched the movie i+1
	user_having_watched_this_movie = []
	for i in range(nb_movie):
		a = training_user_movie_pairs.loc[training_user_movie_pairs['movie_id'] == str(i+1)]['user_id'].tolist()
		user_having_watched_this_movie.append(list(map(int,[a][0])))
		if(a==None):
			print(a)

	# COntent base matrix, final_matrix[i,j] returns the rating of the user i for the movie j
	final_matrix = []
	for i in  range(nb_user+1):
		final_matrix.append([])
		for j in range(nb_movie+1):
			#Initialize the values to 2.5, the esperance of the outputs
			final_matrix[i].append(2.5)

	# Load and format the train data
	training_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_train.csv')) 
	training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv')) 
	user_movie_rating_triplets = np.hstack((training_user_movie_pairs, training_labels.reshape((-1, 1))))

	#Assoiate a rating to the pairs user, movie from the train set
	for triplet in user_movie_rating_triplets:
		final_matrix[triplet[0]][triplet[1]] = triplet[2]

	#Fit the matrix to the data
	for movie in range(nb_movie):
		for i in range(nb_user):
			if(final_matrix[i+1][movie+1]==2.5):
				similar = list(set(user_having_watched_this_movie[movie]).intersection(similar_users4[i]))
				if(similar==[]):
					similar = list(set(user_having_watched_this_movie[movie]).intersection(similar_users3[i]))
				if(similar==[]):
					similar = list(set(user_having_watched_this_movie[movie]).intersection(similar_users2[i]))
				if(similar==[]):
					similar = list(set(user_having_watched_this_movie[movie]).intersection(similar_users1[i]))
				if(similar==[]):
					continue
				mean_rating = 0
				for user in similar:
					mean_rating += final_matrix[user][movie+1] 
				mean_rating = mean_rating/len(similar)
				final_matrix[i+1][movie+1] = mean_rating

    # Load test data
	test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

	# Predict
	y_pred = []
	for test in test_user_movie_pairs:
		y_pred.append(final_matrix[test[0]][test[1]])


	# Making the submission file
	fname = make_submission(y_pred, test_user_movie_pairs, 'toy_example')
	print('Submission file "{}" successfully written'.format(fname))
