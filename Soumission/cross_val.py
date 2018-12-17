from sklearn.linear_model import LinearRegression
import numpy as np
import copy

def error(model,test_set_data,test_set_output,training_set_data,training_set_output):

	if(len(training_set_data)!= len(training_set_output)):
		print("Sizes must match!")
		return -1
	if(len(test_set_data)!= len(test_set_output)):
		print("Sizes must match!")
		return -1
	model.fit(training_set_data,training_set_output)
	prediction = model.predict(test_set_data)
	squarred_error = 0
	for i in range(len(prediction)):
		squarred_error += (prediction[i]-test_set_output[i])**2
	return squarred_error/len(test_set_data)

def OneFoldCrossValidation(reg,X,y):
	separator = int(len(X)/10)
	return error(reg,X[-separator:],	y[-separator:],X[:-separator],y[:-separator])

def TwoFoldCrossValidation(reg,X,y):
	separator = int(len(X)/10)
	Error = error(copy.deepcopy(reg),X[:separator],	y[:separator],X[separator:],y[separator:])
	Error += error(copy.deepcopy(reg),X[-separator:],	y[-separator:],X[:-separator],y[:-separator])
	return Error/2




