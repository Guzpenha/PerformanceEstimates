import optparse
from IPython import embed
import os
import itertools
import operator
import pandas as pd
import csv
import sys
import pickle
sys.path.append("../experiment2/")
from config import *

def generate_predictions_file(rankings,dataset_name,true_ratings):	
	preds_df = []	
	for idx, r in rankings.iterrows():
		for t in r["predicted_list"]:
			preds_df.append([r["userId"],t[0],t[1]])
	preds_df = pd.DataFrame(preds_df,columns= ["userId","movieId","prediction_ensemble"])	
	preds_df.merge(true_ratings[["userId","movieId","rating"]]).to_csv("./created_data/predictions/predictions_borda_count_"+dataset_name+".csv",index=False)

def aggregate_by_borda_count(dataset):
	filehandler = open("../experiment2/created_data/tmp/predictions_all_H2_"+dataset+"_LinearReg_none_.pkl","rb")
	predictons_base_rs = pickle.load(filehandler)
	filehandler.close()
	# predictons_base_rs = pd.read_csv("../experiment2/created_data/tmp/predictions_H2_"+dataset+".csv")	
	user_ranks_by_borda_count = predictons_base_rs.groupby("userId").apply(lambda r,f = borda_count: f(r))	
	return user_ranks_by_borda_count.reset_index(name="predicted_list"),predictons_base_rs

def borda_count(r):
	base_rank_predictions = []
	for c in r.columns:
		if("prediction" in c and "ensemble" not in c and "pondered" not in c):
			base_rank_predictions.append(r.sort_values(c,ascending=False)["movieId"].tolist())
	
	agg_ranks = {}
	for item in base_rank_predictions[0]:
		agg_ranks[item] = 0

	max_rank = len(agg_ranks)
	for r in base_rank_predictions:
		for i,item in enumerate(r):
			agg_ranks[item] += max_rank - i	
	# print(agg_ranks)
	sorted_x = sorted(agg_ranks.items(), key=operator.itemgetter(1),reverse=True)
	return sorted_x

def main():
	parser = optparse.OptionParser()
	parser.add_option('-d', '--datasets', 
						dest="datasets")

	options, remainder = parser.parse_args()	

	datasets = options.datasets.split(",")

	print(datasets)
	for dataset in datasets:
		rankings,true_ratings = aggregate_by_borda_count(dataset)		
		generate_predictions_file(rankings,dataset,true_ratings)

if __name__ == '__main__':
	main()