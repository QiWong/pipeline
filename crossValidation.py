import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def decisionTreeCrossValidate(X, Y):
	'''
	Estimate the accuracy of the decision tree using k-fold cross validation.
	'''
	k_value = 5
	'''k_value is the value of k in the k-fold cross validation. Let k equal to 5'''.
	
	decision_tree = DecisionTreeClassifier()
	workers = -1 #This will prevent using all your CPU power
	
	accuracy = cross_val_score(decision_tree, X, Y, scoring='accuracy', cv = k_value, n_jobs=workers).mean()
	print("Estimated Accuracy of Decision Tree Classifier is: " , accuracy)
	return accuracy

def CSupportVectorCrossValidate(X, Y):
	'''
	Estimate the accuracy of the C-Support Vector Classification. using k-fold cross validation.
	'''
	k_value = 5
	'''k_value is the value of k in the k-fold cross validation. Let k equal to 5'''

	svc = svm.SVC()
	workers = -1 # this will use all your CPU power

	accuracy = cross_val_score(svc, X, Y, scoring='accuracy', cv = k_value, n_jobs=workers).mean()
	print("Estimated Accuracy of C-Support Vector is: " , accuracy)
	return accuracy

def linearSVMCrossValidate(X, Y):
	'''Estimate the accuracy of the Linear Support Vector Classification'''
	linearsvc = svm.LinearSVC(random_state=0)
	k_value = 5
	workers = -1 # this will use all your CPU power
	accuracy = cross_val_score(svc, X, Y, scoring='accuracy', cv = k_value, n_jobs=workers).mean()
	print("Estimated Accuracy of Linear Support Vector is: " , accuracy)
	return accuracy


def RfCrossValidate(X, Y):
	'''Estimate the accuracy of the Random Forest Classification'''
	rf_class = RandomForestClassifier(n_estimators=10)
	print("Random Forests: ")
	k_value = 5
	accuracy = cross_val_score(rf_class, X, Y, scoring='accuracy', cv = k_value).mean() * 100
	print("Accuracy of Random Forests is: " , accuracy)



