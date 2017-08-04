#This is main work flow file in this pipeline. It will call other modules to do
#the data processing.
#!/usr/bin/python

import dataLoading
import dimensionalityReduction as dR
import predictModel as ML
import sys
import crossValidation

#file_name is the name of the file which stores the dataset
class  PipeLine:
	def __init__(self, file_name):
		#We assume the first column of the dataset is the index column, which will not be used in our learning phase.
		self.data_frame = dataLoading.readFile(file_name, 0)

	def dropColumns(self, col_name_list):
		self.data_frame = dataLoading.dropColumns(self.data_frame, col_name_list)

	def corrCof(self):
		#call function in dimensionalityReduction to calculate correlation between columns.
		return dR.standardCorrCoefficient(self.data_frame)

	def visualizeCorr(self):
		#Will generate a heatmap visualizing correlation coefficient matrix.
		dR.visualizeCorrMatrix(self.corrCof())

	def setTarget(self, column_name):
		target_column_data = self.data_frame[column_name]
		#The target column may not be categorial data
		converted_result = dataLoading.convertTextData(target_column_data)
		new_target_column_data = converted_result[1]
		self.target_dict = converted_result[0]
		self.target = new_target_column_data
		self.data_frame = dataLoading.dropColumn(self.data_frame, column_name)
		#print self.data_frame.dtypes

	def pca(self):
		self.data_frame_pca = dR.pcaAnalysis(self.data_frame)

	def classify(self):
		self.data_frame = crossValidation.normalizeData(self.data_frame)
		ML.CSupportVectorClassify(self.data_frame, self.target)
		#ML.decisionTreeClassify(self.data_frame, self.target)

def main(argv):
	if not argv:
		'''Temporarily we use mpMRI_FeatureList_CQ.csv'''
		#print "The file name is empty";
		argv.append('mpMRI_FeatureList_CQ.csv')
		pipeline = PipeLine(argv[0])

		'''There are five columns in the mpMRI_FeatureList_CQ dataset that we won't use in our learning phase.
		Then we delete these five columns from the dataset'''
		unused_columns = ['StudyDate','PSA','Gleason','Location','PIRADS']
		pipeline.dropColumns(unused_columns)

		'''Set which column is the target'''
		pipeline.setTarget('Risk')

		'''Do PCA analysis to reduce the dimensionality.'''
		'''Doesn't affect the result, why?'''
		#pipeline.pca()

		'''Use Machine Learning algorithm to predict result.'''
		pipeline.classify()
		
		
	else:
		print "The file name is "+argv[0]

if __name__ == "__main__":
	main(sys.argv[1:])
	pass

