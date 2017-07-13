#This is main work flow file in this pipeline. It will call other modules to do
#the data processing.
#!/usr/bin/python

import dataLoading
import dimensionalityReduction as dR
import sys

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

	def pca(self):
		self.data_frame_pca = dR.pcaAnalysis(self.data_frame)


def main(argv):
	if not argv:
		'''Temporarily we use mpMRI_FeatureList_CQ.csv'''
		#print "The file name is empty";
		argv.append('mpMRI_FeatureList_CQ.csv')
		pipeline = PipeLine(argv[0])

		'''There are five columns in the mpMRI_FeatureList_CQ dataset that we won't use in our learning phase.
		Then we delete these five columns from the dataset'''
		unused_columns = ['StudyDate','PSA','Gleason','Location','PIRADS','Risk']
		pipeline.dropColumns(unused_columns)

		'''Do PCA analysis to reduce the dimensionality.'''
		pipeline.pca()		
	else:
		print "The file name is "+argv[0]

if __name__ == "__main__":
	main(sys.argv[1:])
	pass

