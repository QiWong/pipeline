'''This module will provide functions for dimensionality reduction. Keep only the interesting features
and eliminate some features'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

def standardCorrCoefficient(dataset):
	'''dataset: in pandas DataFrame type
	This function computes pairwise correlation of columns'''
	return dataset.corr(method='pearson')

def visualizeCorrMatrix(corr_matrix):
	'''This function will visualize correlation coefficient using heatmap, the figure will be saved in local folder'''
	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	sns.heatmap(corr_matrix, cmap=cmap, vmax=.3, center=0,square=True,xticklabels=False,yticklabels=False, linewidths=.5)
	plt.title("Correlation Coefficient")
	plt.savefig('CorrelationCoefficient.png')