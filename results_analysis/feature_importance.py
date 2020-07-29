import optparse
from IPython import embed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,RandomForestRegressor, ExtraTreesRegressor
import sys
sys.path.append("../experiment2/")
from config import *
from h2_ensemble import *
import pyltr
import os

def feature_type(c):
	if("_val" in c):
		return "PE"		
	if("val_set" in c):
		return "EF-Val"
	elif("prediction_" in c):
		return "RS prediction"
	elif("SlopeOne" in c or "SVD" in c or "KNNBaseline" in c or "CoClustering" in c or "NMF" in c ):
		return "EF-Train"
	else:
		return "MF"

def classification_task(datasets):
	final_df = []
	for dataset in datasets:
		df = pd.read_csv("../experiment2/created_data/hypothesis_data/H1_"+dataset+".csv")

		X = df[[c for c in df.columns if c!= "label"]]
		y = df["label"]

		np.random.seed(41)

		clf = RandomForestClassifier(n_estimators=200)
		# clf = ExtraTreesClassifier()
		clf.fit(X,y)

		importances = clf.feature_importances_
		std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
		indices = np.argsort(importances)[::-1]

		# Print the feature ranking
		print("Feature ranking:")

		i=0
		for f in range(X.shape[1]): 
			i+=1
			print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))
			if(i==20):
				break

		types = [feature_type(c) for c in X.columns[indices]]
		f_importance_results = pd.DataFrame([[i,j,k,l,dataset] for i,j,k,l in zip(X.columns[indices],importances[indices],std[indices],types)], columns = ["feature","importance","std","feature_type","dataset"])		
		final_df.append(f_importance_results)		
	pd.concat(final_df).to_csv("./created_data/clf_f_importance.csv")

def regression_task(datasets):
	final_df = []
	for dataset in datasets:
		df = pd.read_csv("../experiment2/created_data/hypothesis_data/H2_"+dataset+".csv")


		#Analysis of PP
		X = df[[c for c in df.columns if c!= "label" and c!="userId.1"]]
		X = X[list(X.columns[0:46])]
		y = df["label"]

		np.random.seed(41)

		clf = RandomForestRegressor(n_estimators=200)
		# clf = ExtraTreesRegressor()
		clf.fit(X,y)

		importances = clf.feature_importances_
		std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
		indices = np.argsort(importances)[::-1]
		# Print the feature ranking
		# print("Feature ranking:")
		
		# i=0
		# for f in range(X.shape[1]): 
		# 	i+=1
		# 	print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))
		# 	if(i==20):
		# 		break

		types = [feature_type(c) for c in X.columns[indices]]
		f_importance_results = pd.DataFrame([[i,j,k,l,dataset] for i,j,k,l in zip(X.columns[indices],importances[indices],std[indices],types)], columns = ["feature","importance","std","feature_type","dataset"])		
		f_importance_results["importance_raw"] = f_importance_results["importance"]
		f_importance_results["importance"] = f_importance_results.apply(lambda r: ("%.3f" % (r["importance"])),axis=1)
		f_importance_results["std"] = f_importance_results.apply(lambda r: ("%.3f" % (r["std"])),axis=1)
		f_importance_results["importance"] = f_importance_results["importance"] + " +- " + f_importance_results["std"]
		print(f_importance_results[f_importance_results["feature_type"]=="MF"].sort_values("importance",ascending=False)[["feature","importance"]][0:10])
		f_importance_results[f_importance_results["feature_type"]=="MF"].sort_values("importance",ascending=False)[["feature","importance"]][0:10].to_csv("../f_importance_pp.csv")
		f_importance_results["analysis"] = "PP"
		final_df.append(f_importance_results)

		#Analysis of RP
		X = df[[c for c in df.columns if c!= "label" and c!="userId.1" and "prediction" in c]]		
		y = df["label"]

		np.random.seed(41)

		clf = RandomForestRegressor(n_estimators=200)
		# clf = ExtraTreesRegressor()
		clf.fit(X,y)

		importances = clf.feature_importances_
		std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
		indices = np.argsort(importances)[::-1]

		types = [feature_type(c) for c in X.columns[indices]]
		f_importance_results = pd.DataFrame([[i,j,k,l,dataset] for i,j,k,l in zip(X.columns[indices],importances[indices],std[indices],types)], columns = ["feature","importance","std","feature_type","dataset"])		
		f_importance_results["importance_raw"] = f_importance_results["importance"]
		f_importance_results["analysis"] = "RP"
		f_importance_results["importance"] = f_importance_results.apply(lambda r: ("%.3f" % (r["importance"])),axis=1)
		f_importance_results["std"] = f_importance_results.apply(lambda r: ("%.3f" % (r["std"])),axis=1)
		f_importance_results["importance"] = f_importance_results["importance"] + " +- " + f_importance_results["std"]
		print(f_importance_results[f_importance_results["feature_type"]=="RS prediction"].sort_values("importance",ascending=False)[["feature","importance","std"]][0:10])
		f_importance_results[f_importance_results["feature_type"]=="RS prediction"].sort_values("importance",ascending=False)[["feature","importance"]][0:10].to_csv("../f_importance_rp.csv")
		final_df.append(f_importance_results)

		##Analysis of PE
		preds = df[[c for c in df.columns if c!= "label" and c!="userId.1" and "prediction" in c]]
		error_estimates = df[[c for c in df.columns if c!= "label" and c!="userId.1" and "RMSE" in c and "val_set" in c]]
		X = ponder_rs_preds_by_ef(preds,error_estimates,"_val")
		y = df["label"]


		clf = RandomForestRegressor(n_estimators=200)
		# clf = ExtraTreesRegressor()
		clf.fit(X,y)

		importances = clf.feature_importances_
		std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
		indices = np.argsort(importances)[::-1]
		types = [feature_type(c) for c in X.columns[indices]]
		f_importance_results = pd.DataFrame([[i,j,k,l,dataset] for i,j,k,l in zip(X.columns[indices],importances[indices],std[indices],types)], columns = ["feature","importance","std","feature_type","dataset"])		
		f_importance_results["importance_raw"] = f_importance_results["importance"]
		f_importance_results["importance"] = f_importance_results.apply(lambda r: ("%.3f" % (r["importance"])),axis=1)
		f_importance_results["std"] = f_importance_results.apply(lambda r: ("%.3f" % (r["std"])),axis=1)
		f_importance_results["importance"] = f_importance_results["importance"] + " +- " + f_importance_results["std"]

		print(f_importance_results[f_importance_results["feature_type"]=="PE"].sort_values("importance",ascending=False)[["feature","importance","std"]][0:10])
		f_importance_results[f_importance_results["feature_type"]=="PE"].sort_values("importance",ascending=False)[["feature","importance"]][0:10].to_csv("../f_importance_pe.csv")
		f_importance_results["analysis"] = "PE"
		final_df.append(f_importance_results)		
		# embed()
	pd.concat(final_df).to_csv("./created_data/regr_f_importance.csv")

def ltr_task(datasets):
	final_df = []
	for dataset in datasets:
		with open('../experiment3/created_data/train/l2r_'+dataset+'_all.csv') as trainfile, \
				open('../experiment3/created_data/test/l2r_'+dataset+'_all.csv') as evalfile:
			TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(trainfile)
			EX, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)

			np.random.seed(41)
			metric = pyltr.metrics.NDCG(k=20)
			model = pyltr.models.LambdaMART(
				metric=metric,
				n_estimators=150,
				learning_rate=0.02,
				max_features=0.5,
				query_subsample=0.5,
				max_leaf_nodes=10,
				min_samples_leaf=64,
				verbose=1,
			)
			model.fit(TX, Ty, Tqids)
			Epred = model.predict(EX)			
			features_names = pd.read_csv("../experiment3/created_data/test/l2r_"+dataset+"_all_features_order.csv")
			features = features_names["0"].tolist()

			importances = model.feature_importances_
			std = np.std([tree[0].feature_importances_ for tree in model.estimators_], axis=0)			
			indices = np.argsort(importances)[::-1]

			# Print the feature ranking
			print("Feature ranking:")
			
			i=0
			for f in range(indices.shape[0]): 
				i+=1
				print("%d. feature %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))
				if(i==20):
					break
			types = [feature_type(c) for c in np.array(features)[indices]]
			f_importance_results = pd.DataFrame([[i,j,k,l,dataset] for i,j,k,l in zip(np.array(features)[indices],importances[indices],std[indices],types)], columns = ["feature","importance","std","feature_type","dataset"])
			final_df.append(f_importance_results)		
	pd.concat(final_df).to_csv("./created_data/ltr_f_importance.csv")

def main():
	parser = optparse.OptionParser()
	parser.add_option('-d', '--datasets', 
						dest="datasets")

	options, remainder = parser.parse_args()	

	datasets = options.datasets.split(",")

	print(datasets)
	# classification_task(datasets)
	regression_task(datasets)
	# ltr_task(datasets)

	# os.system("Rscript f_importance.R")

if __name__ == "__main__":
	main()