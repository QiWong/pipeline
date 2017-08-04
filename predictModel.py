import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def KFoldValidation(num_of_splits):
	'''
	This function is written for K-Fold cross validation.
	Return: KFold cross validation generator
	'''
	kf = KFold(n_splits=num_of_splits)
	return kf

def decisionTreeClassify(X, Y):
	'''A decision tree classifier.
	'''
	
	score_means = list()
	score_stds = list()
	k_values = [2,3,4,5]
	'''
	Evaluate a score by cross-validation
	'''
	decision_tree = DecisionTreeClassifier()
	workers = -1 # this will use all your CPU power
	
	for k_value in k_values:
		this_scores = cross_val_score(decision_tree, X, Y, cv=k_value, n_jobs=workers)
		score_means.append(this_scores.mean())
		score_stds.append(this_scores.std())

	plt.errorbar(k_values, score_means, np.array(score_stds))
	plt.title('Performance of the DecisionTreeClassifier varying the number of folds')
	plt.xlabel('Number of Folds')
	plt.ylabel('Prediction rate')
	plt.axis([1, 5, 0, 1])
	plt.savefig('DecisionTreeClassifier.png')

def CSupportVectorClassify(X, Y):
	'''C-Support Vector Classification.
	C: float, Penalty parameter C of the error term.
	C default=1.0'''
	
	score_means = list()
	score_stds = list()
	k_values = [2,3,4,5]
	'''
	Evaluate a score by cross-validation
	'''
	svc = svm.SVC()
	workers = -1 # this will use all your CPU power
	'''
	for train_index, test_index in kf.split(X):
		x_train, x_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
	'''
	for k_value in k_values:
		this_scores = cross_val_score(svc, X, Y, cv=k_value, n_jobs=workers)
		score_means.append(this_scores.mean())
		score_stds.append(this_scores.std())

	'''Then visualize the result'''
	plt.errorbar(k_values, score_means, np.array(score_stds))
	plt.title('Performance of the SVM varying the percentile of the number of folds')
	plt.xlabel('Number of Folds')
	plt.ylabel('Prediction rate')
	plt.axis([1, 5, 0, 1])
	plt.savefig('CSupportVectorClassify.png')

