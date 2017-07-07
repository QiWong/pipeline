#This module is written for loading dataset.
#Read the data from the file then store data in specialized pandas data structure.

import pandas as pd
from pandas import ExcelFile

def readFile(file_name):
	#Read the raw data from the source file
	#The file could be csv format or xls format
	#The first line should be the header
	raw_dataset = None
	if file_name.endswith('.xls'):
		raw_dataset = pd.read_excel(file_name)
		#Read the dataset from the first sheet of xls file
	elif file_name.endswith('.csv'):
		raw_dataset = pd.read_csv(file_name)
	#raw_dataset is in pandas DataFrame type
	return raw_dataset

def dataSelection(raw_dataset):
	#Choose useful columns in dataset. The raw_dataset is in pandas DataFrame type.


def handleProblematicData():
	
