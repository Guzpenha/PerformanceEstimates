import optparse
from IPython import embed
import pandas as pd
import numpy as np
import sys
sys.path.append("../experiment2/")
from config import *
import os
import functools

def get_winner(r):
	winner =  r.loc[r["NDCG"].idxmax()]["model"]
	score = r.loc[r["NDCG"].idxmax()]["NDCG"]
	if (r[r["NDCG"]==score].shape[0]>1):
		winner="tie"
	
	return winner

def main():			
	parser = optparse.OptionParser()
	parser.add_option('-d', '--datasets', 
						dest="datasets")

	options, remainder = parser.parse_args()	

	datasets = options.datasets.split(",")

	print(datasets)
		
	user_best_models_all = []
	users_raw_enhanced_all = []
	users_delta_mf_vs_ef = []
	for dataset in datasets:
		ef_test = pd.read_csv("../experiment2/created_data/results/raw_eval.csv")
		robustness_df = pd.read_csv("../experiment2/created_data/results/robustness_analysis.csv")
		means_df = robustness_df.groupby(["ensemble","features","dataset","model"])["NDCG"].mean().reset_index()
		means_df.loc[means_df.groupby(["ensemble","dataset"])["NDCG"].idxmax() ,"model_new"] = means_df.loc[means_df.groupby(["ensemble","dataset"])["NDCG"].idxmax()].apply(lambda r : r["ensemble"]+"-" + r["features"],axis=1)		
		m = means_df[~means_df["model_new"].isnull()][["model","dataset","ensemble"]].as_matrix().tolist()		

		def is_winning_model(r,m):
			for winning in m:
				if r["dataset"] in winning and r["model"] in winning and r["ensemble"] in winning:
					return True
			return False

		means_df["model_new"] = means_df.apply(lambda r, m=m,f=is_winning_model: f(r,m),axis=1)
		means_df = means_df[means_df["model_new"]==True]		
		means_df["model_new"] = means_df["ensemble"] +"-" + means_df["features"]
		
		means_df = robustness_df.merge(means_df[["ensemble","features","dataset","model","model_new"]],on=["ensemble","features","dataset","model"])
		means_df = means_df[["userId","ensemble","features","dataset","model","model_new","NDCG"]]
		means_df["model"] = means_df["model_new"]
		means_df.drop(columns=["model_new"])

		mf = pd.read_csv("../experiment2/created_data/tmp/h1_"+dataset+"_user_features_df.csv")
		mf = mf.fillna(0.0)
		for c in mf.columns:
			if c != "userId":
				avg = mf[c].median()
				mf[c+"_above_median"] = mf.apply(lambda r, avg=avg: r[c] >avg, axis=1)
			
		print("Dataset: "+ dataset)		
		comparison_groups = [("LTRERS",["LTRERS-none","LTRERS-meta-features","LTRERS-error-features","LTRERS-error-features-val","LTRERS-all"]),
							 ("STREAM",["STREAM-none","STREAM-meta-features","STREAM-error-features","STREAM-error-features-val","STREAM-all"]),
							 ("SCB",["SCB-meta-features","SCB-error-features","SCB-error-features-val","SCB-all"])]

		user_best_models = []
		user_data_raw_dfs = []
		user_deltas = []
		for (name, group) in comparison_groups:

			df_used_for_delta = means_df[means_df["dataset"]==dataset]
			df_used_for_delta = df_used_for_delta[df_used_for_delta["ensemble"]==name]
			df_used_for_delta = df_used_for_delta[df_used_for_delta["model"].isin(group)]

			ef_of_group = df_used_for_delta[df_used_for_delta["model"].isin(group)]
			user_winners = ef_of_group.groupby("userId").agg(lambda r: get_winner(r))
			user_winners = user_winners[[user_winners.columns[0]]].reset_index()
			user_winners.columns = ["userId","winner_"+name]
			user_best_models.append(user_winners)
		
			user_data_raw = df_used_for_delta[df_used_for_delta["model"].isin(group)]
			user_data_raw["group"] = name
			user_data_raw["dataset"] = dataset
			user_data_raw = user_data_raw.merge(user_winners[["userId","winner_"+name]],on="userId")
			user_data_raw_dfs.append(user_data_raw)

			delta_df_0 = df_used_for_delta[df_used_for_delta["model"].isin([c for c in group if "error-features-val" in c])][["userId","NDCG"]]
			delta_df_1 = df_used_for_delta[df_used_for_delta["model"].isin([c for c in group if "meta-features" in c])][["userId","NDCG"]]
			delta_df = delta_df_0.merge(delta_df_1, on="userId", suffixes=("_EF","_MF"))
			delta_df["delta_EF-val_MF"] = delta_df["NDCG_EF"] - delta_df["NDCG_MF"]
			delta_df = delta_df[["userId","delta_EF-val_MF","NDCG_EF","NDCG_MF"]]
			
			delta_df_0 = df_used_for_delta[df_used_for_delta["model"].isin([c for c in group if "error-features-val" in c])][["userId","NDCG"]]
			delta_df_1 = df_used_for_delta[df_used_for_delta["model"].isin([c for c in group if "error-features" in c and "-val" not in c])][["userId","NDCG"]]
			delta_df_0 = delta_df_0.merge(delta_df_1, on=["userId"], suffixes=("_EF-val","_EF-train"))
			delta_df_0["delta_EF-val_EF-train"] = delta_df_0["NDCG_EF-val"] - delta_df_0["NDCG_EF-train"]
			delta_df = delta_df.merge(delta_df_0, on=["userId"])
			delta_df = delta_df[["userId","delta_EF-val_MF","delta_EF-val_EF-train","NDCG_EF","NDCG_MF","NDCG_EF-train"]]
			
			delta_df_0 = df_used_for_delta[df_used_for_delta["model"].isin([c for c in group if "All" in c or "all" in c])][["userId","NDCG"]]
			delta_df_1 = df_used_for_delta[df_used_for_delta["model"].isin([c for c in group if "meta-features" in c])][["userId","NDCG"]]
			delta_df_3 = delta_df_0.merge(delta_df_1, on=["userId"], suffixes=("_EF-val","_All"))
			delta_df_3["delta_All-MF"] = delta_df_3["NDCG_EF-val"] - delta_df_3["NDCG_All"]
			delta_df = delta_df.merge(delta_df_3, on=["userId"])
			delta_df = delta_df[["userId","delta_EF-val_MF","delta_EF-val_EF-train","delta_All-MF","NDCG_EF","NDCG_MF","NDCG_EF-train","NDCG_All"]]

			if(name !="SCB"):
				delta_df_0 = df_used_for_delta[df_used_for_delta["model"].isin([c for c in group if "error-features-val" in c])][["userId","NDCG"]]
				delta_df_1 = df_used_for_delta[df_used_for_delta["model"].isin([c for c in group if "none" in c])][["userId","NDCG"]]
				delta_df_3 = delta_df_0.merge(delta_df_1, on=["userId"], suffixes=("_EF-val","_None"))
				delta_df_3["delta_EF-val_None"] = delta_df_3["NDCG_EF-val"] - delta_df_3["NDCG_None"]
				delta_df = delta_df.merge(delta_df_3, on=["userId"])
				delta_df = delta_df[["userId","delta_EF-val_MF","delta_EF-val_EF-train","delta_All-MF","delta_EF-val_None","NDCG_EF","NDCG_MF","NDCG_EF-train","NDCG_All","NDCG_None"]]

			delta_df["group"] = name
			delta_df["dataset"] = dataset			
			delta_df = delta_df.merge(mf,on="userId")
			# embed()
			
			
			user_deltas.append(delta_df)
		user_best_models = reduce(lambda x,y: x.merge(y,on="userId"),user_best_models)
		user_data_raw_dfs = pd.concat(user_data_raw_dfs)
		user_deltas = pd.concat(user_deltas)
		user_best_models["dataset"] = dataset

		user_best_models_all.append(user_best_models)
		users_raw_enhanced_all.append(user_data_raw_dfs)
		users_delta_mf_vs_ef.append(user_deltas)

	pd.concat(user_best_models_all).to_csv("./created_data/user_ensemble_winners.csv",index=False)
	pd.concat(users_raw_enhanced_all).to_csv("./created_data/users_raw_enhanced.csv",index=False)
	pd.concat(users_delta_mf_vs_ef).to_csv("./created_data/users_delta.csv",index=False)
	deltas_all = pd.concat(users_delta_mf_vs_ef)
	# embed()
	# os.system("Rscript break_down_analysis.R")
	
if __name__ == "__main__":
	main()

