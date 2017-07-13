#This module is written for loading dataset.
#Read the data from the file then store data in specialized pandas data structure.

import pandas as pd
from pandas import ExcelFile

def readFile(file_name, index_column):
	#Read the raw data from the source file
	#The file could be csv format or xls format
	#The first line should be the header
	#Usually, one column of the dataset is the index column, not the feature, we will not use this column in our prediction.
	raw_dataset = None
	if file_name.endswith('.xls'):
		raw_dataset = pd.read_excel(file_name, index_col= index_column)
		#Read the dataset from the first sheet of xls file
	elif file_name.endswith('.csv'):
		raw_dataset = pd.read_csv(file_name, index_col= index_column)
	#raw_dataset is in pandas DataFrame type
	return raw_dataset

def dropColumns(raw_dataset, column_name_list):
	#Delete columns in dataset, because some features like Date we probably don't want to use it in learning phase. The raw_dataset is in pandas DataFrame type.
	#Delete column by name
	for column_name in column_name_list:
		del raw_dataset[column_name]
	return raw_dataset

