'''This module will provide functions for dimensionality reduction. Keep only the interesting features
and eliminate some features'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2

def standardCorrCoefficient(dataset):
	'''
	This function computes pairwise correlation of columns.
	Parameters:
	@dataset: pandas DataFrame type.
	Output:
	The function returns a correlation coefficient matrix of features. Its type is pandas data frame type.
	'''
	return dataset.corr(method='pearson')

def visualizeCorrMatrix(corr_matrix):
	'''This function will visualize correlation coefficient using heatmap, the figure will be saved in local folder'''
	'''Parameters:
	@corr_matrix: two dimensional array. This parameter is the array return by the standardCorrCoefficient function in this file.
	Output:
	This function will generate a figure which visualizes the correlation matrix of features. The figure's name is CorrelationCoefficient.png.
	The figure is saved at the same directory as this file.
	'''
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	sns.heatmap(corr_matrix, cmap=cmap, vmax=.3, center=0,square=True,xticklabels=False,yticklabels=False, linewidths=.5)
	plt.title("Correlation Coefficient")
	plt.savefig('CorrelationCoefficient.png')

def pcaAnalysis(dataset):
	'''
	This function will do the Principal component analysis of the dataset to reduce the dimensionality of dataset.
	Parameters:
	@dataset: in pandas DataFrame type.
	Output:
	return the dataset after going through the PCA analysis.
	It will also output the shape of the new dataset.
	'''
	pca = PCA()
	dataset = dataset.astype(float)
	pcains = pca.fit(dataset)
	print "The sum of explained_variance_ratio_ is"
	print np.sum(pcains.explained_variance_ratio_)
	dataset2 = pca.fit_transform(dataset)
	print "After PCA analysis the data shape is :" 
	print dataset2.shape
	return dataset2

def selectKBestFeaturesChi2(dataset, target, k_value):
	selector = SelectKBest(score_func=chi2, k= k_value)
	fitted_data = selector.fit_transform(dataset, target)
	return fitted_data
	