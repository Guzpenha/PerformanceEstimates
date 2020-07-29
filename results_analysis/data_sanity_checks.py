import optparse
from IPython import embed
import pandas as pd
import numpy as np
import sys
sys.path.append("../experiment2/")
from config import *

from scipy import stats
import itertools


def main():			
	parser = optparse.OptionParser()
	parser.add_option('-d', '--datasets', 
						dest="datasets")

	options, remainder = parser.parse_args()	

	datasets = options.datasets.split(",")

	print(datasets)

	train_rs_dfs = []
	train_ensembles_dfs = []
	val_dfs = []
	test_dfs = []
		

	for dataset_name in datasets:		
		print(dataset_name)
		train_rs_df = pd.read_csv("../experiment2/created_data/"+dataset_name+"_train.csv", names = ["userId","movieId","rating","timestamp"])
		train_rs_df["dataset"] = dataset_name
		train_rs_df["set"] = "Train RS"
		print(train_rs_df.timestamp.max())
		train_rs_dfs.append(train_rs_df)
		val_df = pd.read_csv("../experiment2/created_data/"+dataset_name+"_validation_set.csv", names = ["userId","movieId","rating","timestamp"])
		val_df["dataset"] = dataset_name
		val_df["set"] = "Validation"
		print(val_df.timestamp.max())
		val_dfs.append(val_df)
		train_ensembles_df = pd.read_csv("../experiment2/created_data/"+dataset_name+"_train_ensembles.csv", names = ["userId","movieId","rating","timestamp"])
		train_ensembles_df["dataset"] = dataset_name
		train_ensembles_df["set"] = "Train Ensembles"
		print(train_ensembles_df.timestamp.max())
		train_ensembles_dfs.append(train_ensembles_df)
		test_df = pd.read_csv("../experiment2/created_data/"+dataset_name+"_test_ensembles.csv", names = ["userId","movieId","rating","timestamp","is_negative_sample"])
		test_df = test_df[test_df["is_negative_sample"]==False]
		test_df["dataset"] = dataset_name
		test_df["set"] = "Test"
		print(test_df.timestamp.max())
		test_dfs.append(test_df[[c for c in test_df.columns if c !="is_negative_sample"]])		

		
	pd.concat(train_rs_dfs + val_dfs + train_ensembles_dfs + test_dfs).to_csv("./created_data/divided_data.csv",index=False)

if __name__ == '__main__':
	main()