import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn import svm
from sklearn import preprocessing

def normalizeData(data):
	'''data: dataframe type
	'''
	normalized_data = preprocessing.normalize(data)
	return normalized_data


def KFoldValidation(num_of_splits):
	'''
	This function is written for K-Fold cross validation.
	Return: KFold cross validation generator
	'''
	kf = KFold(n_splits=num_of_splits)
	return kf