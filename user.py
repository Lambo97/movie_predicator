import numpy
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import time
import datetime
from contextlib import contextmanager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from cross_val import TwoFoldCrossValidation
from sklearn.neighbors import KNeighborsRegressor

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

#matplotlib inline
rng = numpy.random
prefix = 'data/'

# Training Data
training_user_movie_pairs = load_from_csv(os.path.join(prefix,
                                                           'data_train.csv'))
training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

training_users_attributes = load_from_csv(os.path.join(prefix, 'data_user.csv'))
training_movies_attributes = load_from_csv(os.path.join(prefix, 'data_movie.csv'))

#merging
train_data = pd.merge(training_user_movie_pairs,training_movies_attributes,how='left')
train_data = pd.merge(train_data,training_users_attributes,how='left')

#drop too complex attributes
train_data = train_data.drop(['movie_title','video_release_date','IMDb_URL','zip_code','release_date','user_id'], axis=1)
train_data = pd.get_dummies(train_data,columns=['occupation'])
train_data = pd.get_dummies(train_data,columns=['gender'])
y_ls = training_labels

#get attributes names
a = list(train_data.columns.values)
nb_objects = len(train_data)

y_ls = y_ls['rating'].tolist()
for i in range(len(y_ls)):
    y_ls[i] = float(y_ls[i])


movie_inputs = []
user_inputs = []

#Dataframe => list
for i in range(len(a))[:19]:
    #print("movie attributes:  "+a[i])
    movie_inputs.append(train_data[a[i]].tolist())

for i in range(len(a))[19:]:
    #print("user attributes:  "+a[i])
    user_inputs.append(train_data[a[i]].tolist())


#Get only user features as input
training_set = []
for i in range(nb_objects):
    training_set.append([])
    for j in range(len(user_inputs)):
            training_set[i].append(float(user_inputs[j][i]))

for i in range(50):
    reg = RandomForestRegressor(max_depth=i+1)
    print(" RandomForestRegressor(n_estimators=100+ max_depth:"+str(i+1)+") sans normalize data")
    scores = TwoFoldCrossValidation(reg, training_set, y_ls)
    print(scores)

reg = MLPRegressor()
reg = reg.fit(training_set, y_ls)
print(reg.score(training_set, y_ls))

test_user_movie_pairs = load_from_csv(os.path.join(prefix,'data_test.csv'))
train_data = pd.merge(test_user_movie_pairs,training_movies_attributes,how='left')
train_data = pd.merge(train_data,training_users_attributes,how='left')
train_data = train_data.drop(['movie_title','video_release_date','IMDb_URL','zip_code','release_date','user_id'], axis=1)
train_data = pd.get_dummies(train_data,columns=['occupation'])
train_data = pd.get_dummies(train_data,columns=['gender'])


nb_objects = len(train_data)

movie_inputs = []
user_inputs = []

for i in range(len(a))[:19]:
    #print("movie attributes:  "+a[i])
    movie_inputs.append(train_data[a[i]].tolist())

for i in range(len(a))[19:]:
    #print("user attributes:  "+a[i])
    user_inputs.append(train_data[a[i]].tolist())



#Get only user features as input
training_set = []
for i in range(nb_objects):
    training_set.append([])
    for j in range(len(user_inputs)):
            training_set[i].append(float(user_inputs[j][i]))

y_pred = reg.predict(training_set)


for i in range(len(y_pred)):
    a = y_pred[i]
    #print(a)
    y_pred[i] = a
    if a>5:
        y_pred[i]=5
    if a<0:
        y_pred[i]=0



test_user_movie_pairs = test_user_movie_pairs.values.squeeze()
# Making the submission file
fname = make_submission(y_pred, test_user_movie_pairs, 'toy_example')




