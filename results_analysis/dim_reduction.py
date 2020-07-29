import optparse
from IPython import embed
import pandas as pd
import numpy as np
import sys
sys.path.append("../experiment2/")
from config import *
import os
from sklearn.manifold import TSNE

def main():
	parser = optparse.OptionParser()
	parser.add_option('-d', '--datasets', 
						dest="datasets")

	options, remainder = parser.parse_args()	

	datasets = options.datasets.split(",")

	print(datasets)
	
	all_datasets_df = []			
	for dataset in datasets:
		ef = pd.read_csv("../experiment2/created_data/tmp/h2_"+dataset+"_user_train_time_features.csv")	
		ef = ef[[c for c in ef.columns if c!= "userId.1"]]
		ef_val = pd.read_csv("../experiment2/created_data/tmp/h2_"+dataset+"_user_train_time_features_val_set.csv")				
		mf = pd.read_csv("../experiment2/created_data/tmp/h1_"+dataset+"_user_features_df.csv")
		mf = mf.fillna(0.0)
		best_rs = pd.read_csv("../experiment2/created_data/tmp/predictions_H1"+dataset+"_oracle_labels.csv")
		
		tsne = TSNE(n_components =2,random_state=0,perplexity = 25)		
		
		mf_reduced = pd.DataFrame(tsne.fit_transform(mf[[c for c in mf.columns if c not in ["label","RS","userId"]]].as_matrix()),columns=["TSNE_0","TSNE_1"])
		mf["feature"] = "Meta-Features"
		mf = mf.join(mf_reduced)
		mf = mf.merge(best_rs,on="userId")		

		ef = ef.merge(best_rs,on="userId")
		ef_reduced = pd.DataFrame(tsne.fit_transform(ef[[c for c in ef.columns if c not in ["label","RS","userId"]]].as_matrix()),columns=["TSNE_0","TSNE_1"])
		ef = ef.join(ef_reduced)
		ef["feature"] = "Error-Features trainset"

		ef_val = ef_val.merge(best_rs,on="userId")
		ef_reduced = pd.DataFrame(tsne.fit_transform(ef_val[[c for c in ef_val.columns if c not in ["label","RS","userId"]]].as_matrix()),columns=["TSNE_0","TSNE_1"])
		ef_val = ef_val.join(ef_reduced)
		ef_val["feature"] = "Error-Features valset"


		cols = ["feature","TSNE_0","TSNE_1","label"]
		final_df = pd.concat([ef[cols],mf[cols],ef_val[cols]])
		final_df["dataset"] = dataset
		all_datasets_df.append(final_df)

	pd.concat(all_datasets_df).to_csv("./created_data/dim_reduction.csv")
	os.system("Rscript dim_reduction.R")

if __name__ == "__main__":
	main()
