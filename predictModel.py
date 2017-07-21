import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt

def KFoldValidation(num_of_splits):
	'''
	This function is written for K-Fold cross validation.
	Return: KFold cross validation generator
	'''
	kf = KFold(n_splits=num_of_splits)
	return kf

def CSupportVectorClassify(X, Y):
	'''C-Support Vector Classification.
	C: float, Penalty parameter C of the error term.
	C default=1.0'''
	
	score_means = list()
	score_stds = list()
	k_values = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	'''
	Evaluate a score by cross-validation
	'''
	svc = svm.SVC()

	'''
	for train_index, test_index in kf.split(X):
		x_train, x_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
	'''
	for k_value in k_values:
		this_scores = cross_val_score(svc, X, Y, cv=k_value, n_jobs=1)
		score_means.append(this_scores.mean())
		score_stds.append(this_scores.std())

	plt.errorbar(k_values, score_means, np.array(score_stds))
	plt.title('Performance of the SVM varying the percentile of the number of folds')
	plt.xlabel('Number of Folds')
	plt.ylabel('Prediction rate')
	plt.axis([0, 20, 0, 1])
	plt.savefig('CSupportVectorClassify.png')

