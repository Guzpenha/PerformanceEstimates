import optparse
from IPython import embed
import os
import itertools
import sys
sys.path.append("../experiment2/")
from config import *
from calculate_user_features import UserFeatures
from calculate_item_features import ItemFeatures
from h2_ensemble import *
from learn_to_rank import *
import pandas as pd
import csv
from surprise import dump
from surprise import Dataset, Reader


def incremental_analysis(dataset, base_features, ordered_new_features, features):
	

	#generating training sets


	filehandler = open("../experiment2/created_data/tmp/"+dataset+"_with_relevance.pkl",'rb')
	rel_df = pickle.load(filehandler)
	filehandler.close()
	rel_df = rel_df[["userId","movieId","relevance"]]

	train_df = pd.read_csv("../experiment2/created_data/l2r/"+dataset+"_train.csv")
	train_df = train_df[[c for c in train_df.columns if c != "timestamp"]]
	train_df = train_df.merge(rel_df,on=["userId","movieId"],how="left")
	train_df.loc[train_df['rating'] == -1, 'relevance'] = 0

	users_with_train_data = train_df[train_df["relevance"] != 0 ].groupby("userId").filter(lambda x: len(x) >1)		
	# train_df = train_df[train_df["userId"].isin(users_with_train_data.userId.unique())]
	cols_for_all = ["userId","relevance","movieId"]
	pred_cols = [c for c in train_df.columns if "prediction" in c]


	def_ef_val_pondered_rmse = ponder_rs_preds_by_ef(train_df[pred_cols + cols_for_all], train_df[[c for c in train_df.columns if  ("RMSE" in c) and "val_set"  in c]], "val")
	train_df = train_df.merge(def_ef_val_pondered_rmse[[c for c in def_ef_val_pondered_rmse.columns if "val_"in c or c =="userId" or c=="movieId"]], on = ["userId","movieId"])
	
	added_features = []
	i=0
	for new_feature in ordered_new_features[0:]:
		i+=1
		added_features.append(new_feature)
		X = train_df[base_features + cols_for_all + added_features]
		print(X.columns)
		pd_to_l2r(X,dataset).to_csv("./created_data/train/incremental_analysis_"+dataset+"_"+str(i)+".csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)	

	# generating models


	i=0
	for new_feature in ordered_new_features[0:]:
		i+=1
		path = "./created_data/train/incremental_analysis_"+dataset+"_"+str(i)+".csv"

		incremental_analyis_command = "java -jar RankLib-2.8.jar -train "+path+" -ranker " + "7 -epoch 100 " + "-metric2t NDCG@20 -save ./created_data/models/incremental_analysis_"+dataset+"_"+str(i)
	 	incremental_analyis_command += "> created_data/models/incremental_analysis_"+dataset+"_"+str(i)+"_log.txt"
	 	print(incremental_analyis_command)
		os.system(incremental_analyis_command)
	
	
	# generating test sets

	filehandler = open("../experiment2/created_data/tmp/"+dataset+"_with_relevance.pkl",'rb')
	rel_df = pickle.load(filehandler)
	filehandler.close()
	rel_df = rel_df[["userId","movieId","relevance"]]

	filehandler = open("../experiment2/created_data/tmp/predictions_all_H2_"+dataset+"_LinearReg_none_.pkl","rb")
	rs_preds = pickle.load(filehandler)
	filehandler.close()		
	rs_preds = rs_preds[[c for c in rs_preds.columns if "prediction" in c and "ensemble" not in c and "pondered" not in c] + ["userId","movieId","rating"]]

	preds_with_rel = rs_preds.merge(rel_df, on =["userId","movieId"],how="left")
	rs_preds_shape = rs_preds.shape[0]
	del(rs_preds)
	del(rel_df)
	preds_with_rel.loc[preds_with_rel['rating'] == -1, 'relevance'] = 0

	reader = Reader(line_format='user item rating timestamp', sep=',')
	train = Dataset.load_from_file("../experiment2/created_data/"+dataset+"_train.csv", reader=reader)
	uf = UserFeatures(pd.DataFrame(train.raw_ratings,columns = ["userId","movieId","rating","timestamp"]),False)
	user_features = uf.get_all_user_features()

	itemF = ItemFeatures(pd.DataFrame(train.raw_ratings,columns = ["userId","movieId","rating","timestamp"]),False)
	del(train)
	item_features = itemF.get_all_item_features()
			
	preds_with_rel_mf = preds_with_rel.merge(item_features,on = ["movieId"], how = "left").fillna(0.0)
	preds_with_rel_mf = preds_with_rel_mf.merge(user_features,on = ["userId"], how= "left").fillna(0.0)
	del(user_features)
	del(item_features)	
	error_features_val_set = pd.read_csv("../experiment2/created_data/tmp/h2_"+dataset+"_user_train_time_features_val_set.csv")
	preds_and_ef_pondered_val_set = ponder_rs_preds_by_ef(preds_with_rel, error_features_val_set[[c for c in error_features_val_set if "RMSE" in c or "userId" in c]], "val")
	preds_with_rel_mf = preds_with_rel_mf.merge(preds_and_ef_pondered_val_set[[c for c in preds_and_ef_pondered_val_set if "val_" in c or c=="userId" or c=="movieId"]],on=["userId","movieId"])	

	i=0
	added_features = []
	for new_feature in ordered_new_features[0:]:
		i+=1
		added_features.append(new_feature)
		X_test = preds_with_rel_mf[base_features + cols_for_all + added_features]
		print(X_test.columns)
		pd_to_l2r(X_test.sort_values("userId"),dataset).to_csv("./created_data/test/incremental_analysis_"+dataset+"_"+str(i)+".csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)



	# generating predictions

	i=0	
	for new_feature in ordered_new_features[0:]:
		i+=1
		predictions_command_robustness = "java -jar RankLib-2.8.jar -rank ./created_data/test/incremental_analysis_"+dataset+"_"+str(i)+".csv " + \
		"-metric2t NDCG@20 -load ./created_data/models/incremental_analysis_"+dataset+"_"+str(i) +  " -score ./created_data/predictions/incremental_analysis_"+dataset+"_"+str(i)
		os.system(predictions_command_robustness)


	# writting predictions for evaluation
	i=0	
	for new_feature in ordered_new_features[0:]:
		i+=1
		l2r_test_file = pd.read_csv("./created_data/test/incremental_analysis_"+dataset+"_"+str(i)+".csv", sep = " ",header= None)
		doc_column = l2r_test_file.columns[-1]
		l2r_test_file["rating"] = 1000
		l2r_test_file.loc[l2r_test_file[0] == 0, 'rating'] = -1
			
		if("amazon" in dataset):
			l2r_test_file["movieId"] = l2r_test_file.apply(lambda r,c = doc_column: (str(r[c]).split("#")[1]) ,axis=1)
		else:
			l2r_test_file["movieId"] = l2r_test_file.apply(lambda r,c = doc_column: int(str(r[c]).split("#")[1]) ,axis=1)
		
		if("amazon" in dataset):
			l2r_test_file['userId'] = l2r_test_file.apply(lambda r: (str(r[1]).split(":")[1]) ,axis=1)
		else:
			l2r_test_file['userId'] = l2r_test_file.apply(lambda r: int(str(r[1]).split(":")[1]) ,axis=1)
		l2r_test_file['docId'] = l2r_test_file.groupby('userId').cumcount()
				
		l2r_predictions_file = pd.read_csv("./created_data/predictions/incremental_analysis_"+dataset+"_"+ str(i), sep="	", names = ["userId","docId","score"])				
		doc_column = l2r_test_file.columns[-1]
		preds_with_item_id = l2r_predictions_file.merge(l2r_test_file, on = ["userId","docId"])
		preds_with_item_id["prediction_ensemble"] = preds_with_item_id["score"]
		preds_with_item_id[["userId","movieId","prediction_ensemble","rating"]].to_csv("./created_data/predictions/predictions_incremental_analysis_"+ dataset+"_"+str(i)+"_"+features+".csv",index=False)


def main():
	parser = optparse.OptionParser()
	parser.add_option('-d', '--datasets', 
						dest="datasets")

	options, remainder = parser.parse_args()	

	datasets = options.datasets.split(",")

	print(datasets)

	df = pd.read_csv("/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/regr_f_importance.csv")

	ordered_list_pp = list(df[df["analysis"]=="PP"][df["feature_type"]=="MF"].sort_values("importance_raw",ascending=False)["feature"])
	ordered_list_pe = list(df[df["analysis"]=="PE"][df["feature_type"]=="PE"].sort_values("importance_raw",ascending=False)["feature"])
	ordered_list_pe = [c[1:] for c in ordered_list_pe]
	base_features = list(df[df["analysis"]=="PP"][df["feature_type"]=="RS prediction"]["feature"])

	# incremental_analysis(datasets[0],base_features,ordered_list_pp,"pp")
	# incremental_analysis(datasets[0],base_features[0:2],base_features[2:],"rp")
	incremental_analysis(datasets[0], base_features, ordered_list_pe,"pe")

if __name__ == "__main__":
	main()