import optparse
from IPython import embed
import pandas as pd
import numpy as np
import sys
sys.path.append("../experiment2/")
from config import *

from scipy import stats
import itertools
import os

def agg_errors(r):
	cor = []
	# base_column = [c for c in r.columns if c !="userId" and c!="movieId"][0]
	all_columns = [c for c in r.columns if c !="userId" and c!="movieId"]
	for pair in itertools.combinations(all_columns,2):
		# print(pair)
		cor.append(-r[pair[0]].corr(r[pair[1]]))
		# print(r[pair[0]].corr(r[pair[1]]))
	
	return np.array(cor).mean()

def average_absolute_mean_error(df):	
	discordance_df = df.groupby("userId").apply(lambda r, f = agg_errors: f(r))

	return discordance_df.reset_index(name="numeric_discordance").fillna(0.0)

def kendaltau(r):
	base_rank_predictions = []
	base_column = [c for c in r.columns if c !="userId" and c !="movieId"][0]

	distances = []

	all_columns = [c for c in r.columns if c !="userId" and c!="movieId" and "ensemble" not in c]
	# print(all_columns)

	for pair in itertools.combinations(all_columns,2):
		# print(pair)
		r1 = r.sort_values(pair[0],ascending=False)["movieId"].tolist()
		r2 = r.sort_values(pair[1],ascending=False)["movieId"].tolist()
		# print(stats.kendalltau(r1, r2)[0])
		distances.append(-stats.kendalltau(r1, r2)[0])

	return np.array(distances).mean()


def ranking_discordance(df):

	user_ranks_by_borda_count = df.groupby("userId").apply(lambda r,f = kendaltau: f(r))	

	return user_ranks_by_borda_count.reset_index(name="list_discordance")

def main():			
	parser = optparse.OptionParser()
	parser.add_option('-d', '--datasets', 
						dest="datasets")

	options, remainder = parser.parse_args()	

	datasets = options.datasets.split(",")

	print(datasets)

	ef_test_concat = []
	
	for dataset in datasets:
		ef_test = pd.read_csv("../experiment2/created_data/results/raw_eval.csv")
		ef_test = ef_test.fillna(0.0)
		rs_predictions = pd.read_csv("../experiment2/created_data/tmp/predictions_H2"+dataset+".csv")
		rs_predictions = rs_predictions[[c for c in rs_predictions.columns if ("prediction" in c and "ensemble" not in c) or "userId" in c or "movieId" in c]]
		# numeric_discordance = average_absolute_mean_error(rs_predictions)
		# numeric_discordance["dataset"] = dataset				
		# ef_test = ef_test.merge(numeric_discordance,on=["userId","dataset"])		

		rd = ranking_discordance(rs_predictions)
		rd["dataset"] = dataset
		ef_test = ef_test.merge(rd,on=["userId","dataset"])
		# ef_test["numeric_discordance"] = - ef_test["numeric_discordance"] 
		ef_test["list_discordance"] = - ef_test["list_discordance"] 
		ef_test_concat.append(ef_test)

	pd.concat(ef_test_concat)[[c for c in pd.concat(ef_test_concat).columns if "Unnamed" not in c and "VAR" not in c]].to_csv("./created_data/discordance_data.csv",index=False)
	os.system("Rscript discordance_analysis.R")

if __name__ == "__main__":
	main()
