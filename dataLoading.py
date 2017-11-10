'''This module is written for loading dataset.
Read the data from the file then store data in specialized pandas data structure.
'''
import pandas as pd
from pandas import ExcelFile
from sklearn import preprocessing

def readFile(file_name, index_column):
	'''This function is to read the data from the source file.
	The file could be csv format or xls format.
	The first line should be the header
	Usually, one column of the dataset is the index column, not the feature, we will not use this column in our prediction.
	Parameters:
	@file_name: string type. The name of the file which contains the dataset. The file name should end with .xls or .csv
	@index_column: integer. Usually is 0. Because usually the first column is the index column, it's not a feature.
	Output:
	@raw_dataset: pandas dataframe type. The dataframe contains the dataset.
	'''
	raw_dataset = None
	if file_name.endswith('.xls'):
		raw_dataset = pd.read_excel(file_name, index_col= index_column)
		'''Read the dataset from the first sheet of xls file'''
	elif file_name.endswith('.csv'):
		raw_dataset = pd.read_csv(file_name, index_col= index_column)
	'''raw_dataset is in pandas DataFrame type'''
	return raw_dataset

def dropColumns(raw_dataset, column_name_list):
	'''Delete columns in dataset, because some features like Date we probably don't want to use it in learning phase. The raw_dataset is in pandas DataFrame type.
	Delete column by column name.
	Parameters:
	@raw_dataset: pandas dataframe type. The dataset return by readFile function.
	@column_name_list: an array of strings. The array contains the name of the columns that you would like to delete. 
	Output:
	Return the dataset which these columns have been deleted from.
	'''
	for column_name in column_name_list:
		del raw_dataset[column_name]
	return raw_dataset

def dropColumn(raw_dataset, column_name):
	'''Delete the column in dataset, because some features like Date we probably don't want to use it in learning phase. The raw_dataset is in pandas DataFrame type.
	Delete column by column name.
	Parameters:
	@raw_dataset: pandas dataframe type. The dataset return by readFile function.
	@column_name_list: string. It's the name of the column that we would like to delete. 
	Output:
	Return the dataset which the column has been deleted from.
	'''
	del raw_dataset[column_name]
	return raw_dataset

def convertTextData(column_data):
	'''Some columns contain text data. For example, 'Risk' is a categorical feature. It's also the target that we want to predict. It has three values, 'Low', 'Intermediate', 'High'.
	But the text data cannot be used in machine learning. This function will map the categorical data to numerical data.
	Parameters:
	@column_data: list.
	Output:
	The dataset which the column has been deleted from.
	'''
	unique_values = set()
	for value in column_data:
		unique_values.add(value)

	'''column_dict is the mapping from text data to numerical data.'''
	column_dict = {}
	numerical_value = 0
	for unique_value in unique_values:
		column_dict[unique_value] = numerical_value
		numerical_value += 1

	new_column_data = []
	for value in column_data:
		new_column_data.append(column_dict[value])

	result = []
	result.append(column_dict)
	result.append(new_column_data)
	return result

def describeData(dataset):
	'''Generates descriptive statistics that summarize the dataset.
	Parameters:
	@dataset: pandas Dataframe type. The dataset.
	Output:
	Series/DataFrame of summary statistics
	'''
	return dataset.describe()

def normalizeData(data):
	'''
	Parameters:
	@data: dataframe type.
	Output:
	return the dataset which has been normalized.
	'''
	normalized_data = preprocessing.normalize(data)
	return normalized_data
