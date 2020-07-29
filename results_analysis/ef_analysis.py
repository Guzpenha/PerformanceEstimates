import optparse
from IPython import embed
import pandas as pd
import numpy as np
import sys
sys.path.append("../experiment2/")
from config import *
import os

def main():			
	parser = optparse.OptionParser()
	parser.add_option('-d', '--datasets', 
						dest="datasets")

	options, remainder = parser.parse_args()	

	datasets = options.datasets.split(",")

	print(datasets)
	
	final_means_df = []
	ef_test_concat = []
	for dataset in datasets:
		ef = pd.read_csv("../experiment2/created_data/tmp/h2_"+dataset+"_user_train_time_features.csv")	
		ef = ef[[c for c in ef.columns if c!= "userId.1"]]
		ef_val = pd.read_csv("../experiment2/created_data/tmp/h2_"+dataset+"_user_train_time_features_val_set.csv")
		ef_test = pd.read_csv("../experiment2/created_data/tmp/h2_"+dataset+"_user_train_time_features_test_set.csv")
		ef_test["dataset"] = dataset
		ef_test_concat.append(ef_test[[c for c in ef_test if "VAR" not in c]])
		means_df = []
		for c in ef.columns:
			if(c!="userId" and "VAR" not in c):
				test_mean = 0.0
				if c in ef_test.columns:
					test_mean = ef_test[c].mean()
				if "AP" in c:					
					test_mean = ef_test["M"+c].mean()
				elif "Precision" in c:					
					test_mean = ef_test["Precision_"+c.split("_")[1]+"@"+c.split("@")[1].split("_")[0]].mean()

				means_df.append([c,c.split("_")[0],ef[c].mean(),ef_val[c+"_val_set"].mean(),test_mean])
		means_df = pd.DataFrame(means_df, columns = ["metric_full","metric","mean_train_set","mean_val_set","mean_test_set"])
		means_df["dataset"] = dataset

		def multiply_precision(r):
			if("Precision" in r["metric_full"]):
				r["mean_test_set"] = r["mean_test_set"] * 50
				r["mean_val_set"] = r["mean_val_set"] * 50
				r["mean_train_set"] = r["mean_train_set"] * 50
				r["metric_full"] = r["metric_full"] + " (*50)"
			return r
		means_df = means_df.apply(lambda r: multiply_precision(r),axis=1)
		final_means_df.append(means_df)		

	final_means_df = pd.concat(final_means_df)
	final_means_df.to_csv("./created_data/ef_means_data.csv",index=False)
	ef_test_concat = pd.concat(ef_test_concat)
	ef_test_concat.to_csv("./created_data/ef_test_data.csv",index=False)

	os.system("Rscript ef_analysis.R")

if __name__ == "__main__":
	main()
