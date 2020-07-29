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

	labels_dfs = []
		

	for dataset_name in datasets:
		df_no_bias = pd.read_csv("../experiment2/created_data/tmp/"+dataset_name+"_with_relevance.csv")		
		users_in_test = pd.read_csv("../experiment3/created_data/predictions/predictions_borda_count_"+dataset_name+".csv").userId.unique()		
		df_no_bias = df_no_bias[df_no_bias.userId.isin(users_in_test)]
		predictions = pd.read_csv("../experiment2/created_data/tmp/predictions_H2"+dataset_name+".csv")[["userId","movieId","rating","prediction_ensemble"]]
		predictions_with_relevance = predictions.merge(df_no_bias,on=["userId","movieId"],how="left")
		predictions_with_relevance.loc[predictions_with_relevance['rating_x'] == -1, 'relevance'] = 0
		predictions_with_relevance["dataset"] = dataset_name
		labels_dfs.append(predictions_with_relevance[["userId","movieId","rating_x","rating_y","relevance","dataset"]])	

	pd.concat(labels_dfs).to_csv("./created_data/labels_data.csv",index=False)	

if __name__ == '__main__':
	main()