from config import *
from IPython import embed
import os

import pandas as pd

from surprise import SVD, KNNBaseline, CoClustering, NMF, SlopeOne, NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans
from surprise import Dataset, Reader, GridSearch
from surprise import dump
from surprise import accuracy

from calculate_user_features import UserFeatures
from calculate_item_features import ItemFeatures
from evaluate_ensembles import *
from functools import reduce

import math
import random
import optparse

RS_ranges = [
	{
	"name": "KNNWithMeans",
	"algo": KNNWithMeans,
	"params":{
		"k":[10,50]
		}
	},
	{
	"name": "KNNBasic",
	"algo": KNNBasic,
	"params":{
		 "k":[10,50]
		}
	},
	{
	"name": "BaselineOnly",
	"algo": BaselineOnly,
	"params":{}
	},
	{
	"name": "NormalPredictor",
	"algo": NormalPredictor,
	"params":{}
	},
	
	#This wont run for amazon_books:
	# {
	# "name": "SlopeOne",
	# "algo": SlopeOne,
	# "params":{}
	# },
	{
	"name": "SVD",
	"algo": SVD,
	"params":{
		 'lr_all': [0.01, 0.001],
		 'reg_all': [0.01, 0.09]
		}
	},
	{ # KNNBaseline will not run on datasets_knn_mem_error of config.py (memory error)
	"name": "KNNBaseline",
	"algo": KNNBaseline,
	"params":{
		'k': [10,50]
		}
	},
	{
	"name": "CoClustering",
	"algo": CoClustering,
	"params":{
		'n_cltr_u': [1,5],
		'n_cltr_i': [1,5],
		'n_epochs': [20,50]
		} 
	},
	{
	"name": "NMF",
	"algo": NMF,
	"params":{
		 'n_epochs': [20, 100],
		 'n_factors': [20, 100]
 		}
	}	
]

RS = []

random_RS_number = 5

for rs in RS_ranges:
	if(len(rs["params"].keys())!=0):
		for i in range(random_RS_number):
			params = {}
			random.seed(i+10)				
			for k in rs["params"].keys():
				if(type(rs["params"][k][0]) == float):
					rand_number = random.uniform(rs["params"][k][0],rs["params"][k][1])
					params[k] = rand_number
				else:
					rand_number = random.randint(rs["params"][k][0],rs["params"][k][1])
					params[k] = rand_number
			rs_info = {
				"name": rs["name"] +"_"+ '_'.join([k +"_" + str(params[k]) for k in params.keys()]),
				"algo": rs["algo"],
				"params": params
			}
			RS.append(rs_info)
	else:
		rs_info = {
			"name": rs["name"],
			"algo": rs["algo"],
			"params": {}
		}

		RS.append(rs_info)

def grid_search_rec(data,algo,param_grid):	
	"""
	Grid search a RS algorithm using surprise lib.

	Inputs
	---------
		data: surprise Dataset trainset format for ratings
		algo: surprise algorithm for grid searching 
		param_grid: parameter map to grid search

	Returns
	--------
		best_estimator: surprise algorithm with best hyperparameters

	"""

	grid_search = GridSearch(algo, param_grid, measures=['RMSE'],verbose=True)
	grid_search.evaluate(data)

	return grid_search.best_estimator["RMSE"]

def random_search_all_RS(datasets):
	""" 	
	Grid searches all algorithms on RS variable and for all datasets in config.py.
	Serializes fitted algorithms on best hyperparameters combination found to dump_files.	
	"""

	for dataset_name in datasets:
		reader = Reader(line_format='user item rating timestamp', sep=',')		
		reader_no_timestamp = Reader(line_format='user item rating', sep=',')		
		
		user_train_time_meta_features = []
		for rs in RS:
			train = Dataset.load_from_file("./created_data/"+dataset_name+"_train.csv", reader=reader)
			trainset = train.build_full_trainset()
			testset = trainset.build_testset()			
			negative_sampling = Dataset.load_from_file("./created_data/"+dataset_name+"_train_negative_sample.csv", reader=reader_no_timestamp)		
			negative_sampling_trainset = negative_sampling.build_full_trainset()
			testset_ns = negative_sampling_trainset.build_testset()
			del(negative_sampling_trainset)
			del(negative_sampling)			
			validation = Dataset.load_from_file("./created_data/"+dataset_name+"_validation_set.csv", reader=reader)
			validation = validation.build_full_trainset()
			validation = validation.build_testset()

			#Memory error for 16GB machine or float division error for lastfm
			if("KNN" in rs["name"] and dataset_name in datasets_knn_mem_error):
				continue
			
			file_name = os.path.expanduser('./created_data/trained_RS/dump_file_'+dataset_name+'_'+rs["name"])
			if os.path.exists(file_name):
				print("Loading " + rs["name"] +" for "+ dataset_name )
				_, estimator = dump.load(file_name)
					
			else: 
				print("Training " + rs["name"] +" on "+ dataset_name)			
				estimator = rs["algo"](**rs["params"])
				estimator.train(trainset)
				#dump estimator to file
				file_name = os.path.expanduser('./created_data/trained_RS/dump_file_'+dataset_name+'_'+rs["name"])
				dump.dump(file_name, algo=estimator)

			del(train)
			del(trainset)
						

			## ERROR-FEATURES USING TRAIN SET
			preds = estimator.test(testset)
			del(testset)
			predictions_df = pd.DataFrame(preds,columns = ["userId","movieId","rating","prediction","details"])					

			# Calculating user MAE
			# predictions_df["error"] = abs(predictions_df["prediction"]-predictions_df["rating"])
			# avg_errors = predictions_df.groupby("userId")["error"].mean().rename("MAE_"+rs["name"]).to_frame().reset_index() 
			# avg_errors_var = predictions_df.groupby("userId")["error"].var().rename("MAE_VAR_"+rs["name"]).to_frame().reset_index() 
			# avg_errors = avg_errors.merge(avg_errors_var,on="userId")

			# Calculating user MSE and RMSE
			predictions_df["squared_error"] = (predictions_df["prediction"]-predictions_df["rating"]) * (predictions_df["prediction"]-predictions_df["rating"])
			avg_squared_errors = predictions_df.groupby("userId")["squared_error"].mean().rename("MSE_"+rs["name"]).to_frame().reset_index() 
			avg_squared_errors_var = predictions_df.groupby("userId")["squared_error"].var().rename("MSE_VAR_"+rs["name"]).to_frame().reset_index() 			
			avg_squared_errors["RMSE_"+rs["name"]] = avg_squared_errors.apply(lambda r,rs = rs["name"],math=math: math.sqrt(r["MSE_"+rs]),axis=1)
			# avg_errors = avg_errors.merge(avg_squared_errors.merge(avg_squared_errors_var,on="userId"), on="userId")
			avg_errors = avg_squared_errors.merge(avg_squared_errors_var,on="userId")[["RMSE_"+rs["name"],"MSE_VAR_"+rs["name"],"userId"]]
			
			# Calculating user NDCG
			preds_ns = estimator.test(testset_ns)			
			del(testset_ns)			
			predictions_ns_df = pd.DataFrame(preds_ns,columns = ["userId","movieId","rating","prediction","details"])					
			predictions_ns_df.to_csv("./created_data/l2r/predictions_train_ns_"+dataset_name+"_"+rs["name"]+".csv", index=False)
			predictions_df = pd.concat([predictions_df,predictions_ns_df])
			del(predictions_ns_df)
			predictions_with_relevance = remove_dataset_bias(predictions_df, has_ns = True)
			del(predictions_df)
			scores = predictions_with_relevance.groupby("userId").agg(lambda r,f = calculate_ndcg_score: f(r,"prediction"))
			scores = scores[[scores.columns[0]]].rename(index=str,columns={scores.columns[0]:"NDCG_"+rs["name"]}).reset_index()
			
			# # Calculating RR
			# scores_rr = predictions_with_relevance.groupby("userId").agg(lambda r,f = calculate_reciprocal_rank: f(r,"prediction"))
			# scores_rr = scores_rr[[scores_rr.columns[0]]].rename(index=str,columns={scores_rr.columns[0]:"RR_"+rs["name"]}).reset_index()
			# scores = scores.merge(scores_rr,on="userId")

			# # Calculating user MAP
			# scores_map = predictions_with_relevance.groupby("userId").agg(lambda r,f = calculate_ap_score: f(r,"prediction"))
			# scores_map = scores_map[[scores_map.columns[0]]].rename(index=str,columns={scores.columns[0]:"AP_"}).reset_index()
			# scores_map.columns = ["userId","AP_"+rs["name"]]
			# scores = scores.merge(scores_map,on="userId")

			# # Calculating user Precision
			# for k in [5,10,20]:

			# 	scores_precision = predictions_with_relevance.groupby("userId").agg(lambda r,f = calculate_precision_score,s=k: f(r,"prediction",s))
			# 	scores_precision = scores_precision[[scores_precision.columns[0]]].rename(index=str,columns={scores_precision.columns[0]:"Precision@"+str(k)+"_"+rs["name"]}).reset_index()

			# 	scores = scores.merge(scores_precision,on="userId")
			
			user_wise_train_errors = avg_errors.merge(scores,on="userId")					

			## ERROR-FEATURES USING VALIDATION SET
			preds = estimator.test(validation)
			del(validation)
			del(estimator)
			predictions_df = pd.DataFrame(preds,columns = ["userId","movieId","rating","prediction","details"])

			# # Calculating user MAE
			# predictions_df["error"] = abs(predictions_df["prediction"]-predictions_df["rating"])
			# avg_errors = predictions_df.groupby("userId")["error"].mean().rename("MAE_"+rs["name"]+"_val_set").to_frame().reset_index() 
			# avg_errors_var = predictions_df.groupby("userId")["error"].var().rename("MAE_VAR_"+rs["name"]+"_val_set").to_frame().reset_index() 
			# avg_errors = avg_errors.merge(avg_errors_var,on="userId")

			# Calculating user MSE and RMSE
			predictions_df["squared_error"] = (predictions_df["prediction"]-predictions_df["rating"]) * (predictions_df["prediction"]-predictions_df["rating"])
			avg_squared_errors = predictions_df.groupby("userId")["squared_error"].mean().rename("MSE_"+rs["name"]+"_val_set").to_frame().reset_index() 
			avg_squared_errors_var = predictions_df.groupby("userId")["squared_error"].var().rename("MSE_VAR_"+rs["name"]+"_val_set").to_frame().reset_index() 
			avg_squared_errors["RMSE_"+rs["name"]+"_val_set"] = avg_squared_errors.apply(lambda r,rs = rs["name"],math=math: math.sqrt(r["MSE_"+rs+"_val_set"]),axis=1)
			avg_errors = avg_errors.merge(avg_squared_errors.merge(avg_squared_errors_var,on="userId"), on="userId")[["RMSE_"+rs["name"]+ "_val_set","MSE_VAR_"+rs["name"]+ "_val_set","userId"]]

			# Calculating user NDCG			
			predictions_ns_df = pd.DataFrame(preds_ns,columns = ["userId","movieId","rating","prediction","details"])
			predictions_df = pd.concat([predictions_df,predictions_ns_df])
			del(predictions_ns_df)			
			predictions_with_relevance = remove_dataset_bias(predictions_df, has_ns = True)
			del(predictions_df)
			scores = predictions_with_relevance.groupby("userId").agg(lambda r,f = calculate_ndcg_score: f(r,"prediction"))
			scores = scores[[scores.columns[0]]].rename(index=str,columns={scores.columns[0]:"NDCG_"+rs["name"]+"_val_set"}).reset_index()			

			# Calculating RR
			# scores_rr = predictions_with_relevance.groupby("userId").agg(lambda r,f = calculate_reciprocal_rank: f(r,"prediction"))
			# scores_rr = scores_rr[[scores_rr.columns[0]]].rename(index=str,columns={scores_rr.columns[0]:"RR_"+rs["name"]+"_val_set"}).reset_index()
			# scores = scores.merge(scores_rr,on="userId")

			# # Calculating user MAP
			# scores_map = predictions_with_relevance.groupby("userId").agg(lambda r,f = calculate_ap_score: f(r,"prediction"))
			# scores_map = scores_map[[scores_map.columns[0]]].rename(index=str,columns={scores.columns[0]:"AP_"}).reset_index()
			# scores_map.columns = ["userId","AP_"+rs["name"]+"_val_set"]

			# scores = scores.merge(scores_map,on="userId")

			# # Calculating user Precision
			# for k in[5,10,20]:

			# 	scores_precision = predictions_with_relevance.groupby("userId").agg(lambda r,f = calculate_precision_score,s=k: f(r,"prediction",s))
			# 	scores_precision = scores_precision[[scores_precision.columns[0]]].rename(index=str,columns={scores_precision.columns[0]:"Precision@"+str(k)+"_"+rs["name"]+"_val_set"}).reset_index()

			# 	scores = scores.merge(scores_precision,on="userId")

			user_wise_train_errors_val_set = avg_errors.merge(scores,on="userId")			

			all_error_features = user_wise_train_errors.merge(user_wise_train_errors_val_set, on="userId",how="left")
			# some variances are zero, as sometimes there are only one rating um val set 
			all_error_features = all_error_features.fillna(0.0)
			assert user_wise_train_errors.userId.shape[0] == user_wise_train_errors_val_set.shape[0]

			# Using only validation set error features
			user_train_time_meta_features.append(all_error_features)			

		all_features = reduce(lambda x,y: x.merge(y,on="userId"), user_train_time_meta_features)
		all_features[["userId"]+ [c for c in all_features if "val_set" not in c]].to_csv("./created_data/tmp/h2_"+dataset_name+"_user_train_time_features.csv",header=True,index=False)
		all_features[["userId"]+ [c for c in all_features if "val_set" in c]].to_csv("./created_data/tmp/h2_"+dataset_name+"_user_train_time_features_val_set.csv",header=True,index=False)

def create_best_RS_userwise_dataset(errors_df,user_features_df):
	"""
	Creates dataset that for a classify task, where X are user features and y is 
	the best RS on average.

	Inputs
	---------
		errors_df: pandas DataFrame containg average errors (avg_error) and user ids (userId) columns
		user_features_df: pandas DataFrame containing user features as columns and user ids (userId)

	Returns
	--------
		Xy: pandas DataFrame containing user ids (userID), features and labels (label).

	"""		
	# this was used when mae was the criterea for creating the H1 dataset
	# min_avg_errors = errors_df.loc[errors_df.groupby("userId")["avg_error"].idxmin()]		
	min_avg_errors = errors_df.loc[errors_df.groupby("userId")["NDCG"].idxmax()]
	df = min_avg_errors.merge(user_features_df,on="userId")	
	Xy = df[[c for c in df.columns if c != "avg_error"]]
	Xy["label"]= Xy["RS"]
	Xy = Xy[[c for c in Xy.columns if c != "RS"]]	
	return Xy

def create_H1_datasets(datasets):
	""" 
	Creates datasets for H1 according to config.py inputs .
	Requires serialized models (from grid_search) in ./created_data.	
	"""

	H1_datasets = []

	for dataset_name in datasets:
		reader = Reader(line_format='user item rating timestamp', sep=',')
		train_ensembles = Dataset.load_from_file("./created_data/"+dataset_name+"_train_ensembles.csv", reader=reader)

		negative_sampling = Dataset.load_from_file("./created_data/"+dataset_name+"_train_negative_sample.csv", reader=reader)
		negative_sampling_trainset = negative_sampling.build_full_trainset()
		testset_ns = negative_sampling_trainset.build_testset()
			
		uf = UserFeatures(pd.DataFrame(train_ensembles.raw_ratings,columns = ["userId","movieId","rating","timestamp"]),False)
		user_features_df = uf.get_all_user_features()		

		user_train_time_features = pd.read_csv("./created_data/tmp/h2_"+dataset_name+"_user_train_time_features.csv")
		user_train_time_features["userId"] = user_train_time_features["userId"].astype(str)

		user_train_time_val_features = pd.read_csv("./created_data/tmp/h2_"+dataset_name+"_user_train_time_features_val_set.csv")
		user_train_time_val_features["userId"] = user_train_time_val_features["userId"].astype(str)

		all_user_features_df = (user_features_df.merge(user_train_time_features,on="userId")).merge(user_train_time_val_features,on="userId",how="left")		
		all_user_features_df = all_user_features_df.fillna(0.0)
		assert user_features_df.userId.shape[0] == user_train_time_features.shape[0]
		assert user_features_df.userId.shape[0] == user_train_time_val_features.shape[0]

		recs_avg_errors = []		
		for rs in RS:
			#Memory error for 16GB machine or float division error for lastfm
			if("KNN" in rs["name"] and dataset_name in datasets_knn_mem_error):
				continue
			file_name = os.path.expanduser('./created_data/trained_RS/dump_file_'+dataset_name+'_'+rs["name"])
			_, loaded_algo = dump.load(file_name)
	
			predictions = loaded_algo.test(train_ensembles.build_full_trainset().build_testset())
			predictions_df = pd.DataFrame(predictions,columns = ["userId","movieId","rating","prediction","details"])

			preds_ns = loaded_algo.test(testset_ns)
			predictions_ns_df = pd.DataFrame(preds_ns,columns = ["userId","movieId","rating","prediction","details"])					
			predictions_df = pd.concat([predictions_df,predictions_ns_df])
			predictions_with_relevance = remove_dataset_bias(predictions_df, has_ns = True)

			scores = predictions_with_relevance.groupby("userId").agg(lambda r,f = calculate_ndcg_score: f(r,"prediction"))
			scores = scores[[scores.columns[0]]].rename(index=str,columns={scores.columns[0]:"NDCG"}).reset_index()
			scores["RS"] = rs["name"]			

			recs_avg_errors.append(scores)

		all_avg_errors = pd.concat(recs_avg_errors).reset_index()

		hypothesis_df = create_best_RS_userwise_dataset(all_avg_errors,all_user_features_df)		
		
		H1_datasets.append(hypothesis_df[[c for c in hypothesis_df.columns if c not in ["userId","userId.1","index","NDCG"]]])

	return H1_datasets

def create_H2_datasets(datasets):
	""" 
	Creates datasets for H2 according to config.py inputs .
	Requires serialized models (from grid_search) in ./created_data.	
	"""

	H2_datasets = []

	for dataset_name in datasets:
		reader = Reader(line_format='user item rating timestamp', sep=',')		
		train = Dataset.load_from_file("./created_data/"+dataset_name+"_train.csv", reader=reader)
		train_ensembles = Dataset.load_from_file("./created_data/"+dataset_name+"_train_ensembles.csv", reader=reader)
			
		uf = UserFeatures(pd.DataFrame(train.raw_ratings,columns = ["userId","movieId","rating","timestamp"]),False)
		user_features_df = uf.get_all_user_features()

		itemF = ItemFeatures(pd.DataFrame(train.raw_ratings,columns = ["userId","movieId","rating","timestamp"]),False)
		del(train)
		item_features_df = itemF.get_all_item_features()
		item_features_df.to_csv("./created_data/tmp/h1_"+dataset_name+"_item_features_df.csv",index=False)

		user_train_time_features = pd.read_csv("./created_data/tmp/h2_"+dataset_name+"_user_train_time_features.csv")
		user_train_time_features["userId"] = user_train_time_features["userId"].astype(str)

		user_train_time_features_val_set = pd.read_csv("./created_data/tmp/h2_"+dataset_name+"_user_train_time_features_val_set.csv")
		user_train_time_features_val_set["userId"] = user_train_time_features_val_set["userId"].astype(str)
		
		recs_predictions = pd.DataFrame(train_ensembles.raw_ratings, columns = ["userId","movieId","rating","timestamp"])
		recs_predictions["label"] = recs_predictions["rating"]
		
		recs_predictions_with_ns = pd.read_csv("./created_data/"+dataset_name+"_train_negative_sample.csv", names = ["userId","movieId","rating","timestamp"])
		recs_predictions_with_ns["label"] = recs_predictions_with_ns["rating"]

		for rs in RS:
			#Memory error for 16GB machine or float division error for lastfm
			if("KNN" in rs["name"] and dataset_name in datasets_knn_mem_error):
				continue
			file_name = os.path.expanduser('./created_data/trained_RS/dump_file_'+dataset_name+'_'+rs["name"])
			_, loaded_algo = dump.load(file_name)
	
			predictions = loaded_algo.test(train_ensembles.build_full_trainset().build_testset())			
			predictions_df = pd.DataFrame(predictions,columns = ["userId","movieId","rating","prediction_"+rs["name"],"details"])			
			recs_predictions = recs_predictions.merge(predictions_df[["userId","movieId","prediction_"+rs["name"]]],on = ["userId","movieId"])			
						
			predictions_ns = pd.read_csv("./created_data/l2r/predictions_train_ns_"+dataset_name+"_"+rs["name"]+".csv")
			predictions_ns["prediction_"+rs["name"]] = predictions_ns["prediction"]
			predictions_ns = predictions_ns[[c for c in predictions_ns.columns if c != "prediction"]]
			recs_predictions_with_ns = recs_predictions_with_ns.merge(predictions_ns[["userId","movieId","prediction_"+rs["name"]]],on = ["userId","movieId"])
		
		del(loaded_algo)
		del(train_ensembles)

		H2_dataset = recs_predictions.merge(user_features_df,on="userId")
		H2_dataset = H2_dataset.merge(item_features_df,on="movieId")
		user_train_time_features = user_train_time_features.fillna(0.0)
		H2_dataset = H2_dataset.merge(user_train_time_features,on="userId")
		user_train_time_features_val_set = user_train_time_features_val_set.fillna(0.0)
		H2_dataset = H2_dataset.merge(user_train_time_features_val_set,on="userId",how="left")
		#this is ok as users with only one rating do not have some meta-features
		H2_dataset = H2_dataset.fillna(0.0)				
		
		H2_dataset = H2_dataset[[c for c in H2_dataset.columns if c not in ["index","userId","movieId","timestamp","rating"]]]

		H2_datasets.append(H2_dataset)
		del(H2_dataset)

		#dataset for l2r		
		recs_predictions_with_ns['userId'] = recs_predictions_with_ns['userId'].astype('str') 
		recs_predictions_with_ns['movieId'] = recs_predictions_with_ns['movieId'].astype('str') 
		L2R_dataset = pd.concat([recs_predictions_with_ns,recs_predictions])
		del(recs_predictions)
		del(recs_predictions_with_ns)
		L2R_dataset = L2R_dataset.merge(user_features_df,on="userId")
		del(user_features_df)
		L2R_dataset = L2R_dataset.merge(item_features_df,on="movieId")
		del(item_features_df)
		L2R_dataset = L2R_dataset.merge(user_train_time_features,on="userId")
		del(user_train_time_features)
		L2R_dataset = L2R_dataset.merge(user_train_time_features_val_set,on="userId",how="left")
		del(user_train_time_features_val_set)		

		L2R_dataset.to_csv("./created_data/l2r/"+dataset_name+"_train.csv",index=False)

		del(L2R_dataset)

	return H2_datasets

def main():
	"""
	This script runs grid_search on all recommnders in RS variable, and it 
	creates hypothesis data for datasets in "config.py", for both H1 and H2.
	"""

	parser = optparse.OptionParser()
	parser.add_option('-d', '--datasets', 
						dest="datasets")

	options, remainder = parser.parse_args()	

	datasets = options.datasets.split(",")

	print(datasets)

	random_search_all_RS(datasets)

	# H1_datasets = create_H1_datasets(datasets)	
	# for dataset_name,df in zip(datasets,H1_datasets):
	# 	df.to_csv("./created_data/hypothesis_data/H1_"+dataset_name+".csv",index=False)
	
	H2_datasets = create_H2_datasets(datasets)
	for dataset_name,df in zip(datasets,H2_datasets):
		df.to_csv("./created_data/hypothesis_data/H2_"+dataset_name+".csv",index=False)

if __name__ == "__main__":
	main()