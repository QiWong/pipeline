#This is main work flow file in this pipeline. It will call other modules to do
#the data processing.
#!/usr/bin/python

import dataLoading
import dimensionalityReduction as DR
import sys
import crossValidation as CV
import numpy
import overSampling

class  PipeLine:
	def __init__(self, file_name):
		#We assume the first column of the dataset is the index column, which will not be used in our learning phase.
		self.all_data = dataLoading.readFile(file_name, 0)
		self.features_data = self.all_data

	def dropColumns(self, col_name_list):
		self.all_data = dataLoading.dropColumns(self.features_data, col_name_list)

	def corrCof(self):
		#call function in dimensionalityReduction to calculate correlation between features.
		#The result will be written to the file called  'corr_matrix.txt' 
		corr_matrix = DR.standardCorrCoefficient(self.features_data)
		file = open("corr_matrix.txt","w")
		file.write(str(corr_matrix))
		file.close()
		return corr_matrix

	def visualizeCorr(self):
		#Will generate a heatmap visualizing correlation coefficient matrix.
		DR.visualizeCorrMatrix(self.corrCof())
		

	def setTarget(self, column_name):
		target_column_data = self.features_data[column_name]
		converted_result = dataLoading.convertTextData(target_column_data)
		new_target_column_data = converted_result[1]
		self.target_dict = converted_result[0]
		self.target = new_target_column_data
		self.features_data = dataLoading.dropColumn(self.all_data, column_name)
		

	def pca(self):
		self.features_data = DR.pcaAnalysis(self.features_data)


	def overSampling(self):
		'''There are three over sampling methods - randomOverSampling, smoteOverSampling and adasynOverSampling in the overSampling module.''' 
		self.features_data, self.target = overSampling.randomOverSampling(self.features_data, self.target)
		

	def crossValidate(self):
		'''Estimate the accuracy of machine learning algorithms using cross validation.'''
		self.features_data = numpy.asarray(self.features_data)
		self.target = numpy.asarray(self.target)
		
		CV.decisionTreeCrossValidate(self.features_data, self.target)
		CV.CSupportVectorCrossValidate(self.features_data, self.target)
		CV.linearSVMCrossValidate(self.features_data, self.target)
		

def main():
	'''Temporarily we use mpMRI_FeatureList_CQ.csv for testing purpose.'''
	''''''
	file_name = 'mpMRI_FeatureList_CQ.csv'

	pipeline = PipeLine(file_name)

	'''There are five columns in the mpMRI_FeatureList_CQ dataset that we won't use in our learning phase.
	Then we delete these five columns from the dataset'''
	unused_columns = ['StudyDate','PSA','Gleason','Location','PIRADS']
	pipeline.dropColumns(unused_columns)

	'''Set which column is the target'''
	pipeline.setTarget('Risk')

	#pipeline.corrCof()

	'''Do PCA analysis to reduce the dimensionality.'''
	pipeline.pca()

	'''Do the over sampling.'''
	pipeline.overSampling()

		
	pipeline.crossValidate()
		

if __name__ == "__main__":
	main()
	pass

