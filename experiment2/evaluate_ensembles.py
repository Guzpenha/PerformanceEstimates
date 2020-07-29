import optparse
from config import *
from IPython import embed

import os.path
import os
import sys
sys.path.append("../experiment3/")
from learn_to_rank import *
import pandas as pd
import numpy as np

from functools import reduce
from scipy import stats
from numpy import copy
import math
from h2_ensemble import *
from h1_ensemble import *
from create_hypothesis_dataset import *

precision_levels = [5,10,20]

import numpy as np
from functools import reduce

import warnings

warnings.filterwarnings('ignore')

import scipy as sp
import scipy.stats

def confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return h

def calculate_reciprocal_rank(r,prediction_column):		
	
	def reciprocal_rank(r):		
		pos_relevant = [i for i,v in enumerate(r) if v>0]		
		if(len(pos_relevant)>0):
			return 1.0/float(pos_relevant[0]+1)
		else: 
			return 0
	np.random.seed(42)
	r = r.iloc[np.random.permutation(len(r))]
	predicted = r.sort_values(prediction_column,ascending=False)["relevance"].tolist()	
	# print(predicted)
	# print(reciprocal_rank(predicted))
	return reciprocal_rank(predicted)

"""
https://gist.github.com/bwhite/3726239
"""

def calculate_ap_score(r,prediction_column):
	def precision_at_k(r, k):
		assert k >= 1
		r = np.asarray(r)[:k] != 0
		if r.size != k:
			raise ValueError('Relevance score length < k')
		return np.mean(r)


	def average_precision(r):
		r = np.asarray(r) != 0
		out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
		if not out:
			return 0.
		return np.mean(out)

	np.random.seed(42)
	r = r.iloc[np.random.permutation(len(r))]
	predicted = r.sort_values(prediction_column,ascending=False)["relevance"].tolist()	
	# print(predicted)
	# print(average_precision(predicted))
	# true_relevance = r.sort_values("relevance",ascending=False)["movieId"].tolist()
	# print(predicted)
	# print(true_relevance)	
	return average_precision(predicted)

#https://github.com/CoBiG2/cobig_misc_scripts/blob/master/FDR.py
def multiple_testing_correction(pvalues, correction_type="Bonferroni"):
    """
    Consistent with R - print
    correct_pvalues_for_multiple_testing([0.0, 0.01, 0.029, 0.03, 0.031, 0.05,
                                          0.069, 0.07, 0.071, 0.09, 0.1])
    """
    from numpy import array, empty
    pvalues = array(pvalues)
    sample_size = pvalues.shape[0]
    qvalues = empty(sample_size)
    if correction_type == "Bonferroni":
        # Bonferroni correction
        qvalues = sample_size * pvalues
    elif correction_type == "Bonferroni-Holm":
        # Bonferroni-Holm correction
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        for rank, vals in enumerate(values):
            pvalue, i = vals
            qvalues[i] = (sample_size-rank) * pvalue
    elif correction_type == "FDR":
        # Benjamini-Hochberg, AKA - FDR test
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = sample_size - i
            pvalue, index = vals
            new_values.append((sample_size/rank) * pvalue)
        for i in range(0, int(sample_size)-1):
            if new_values[i] < new_values[i+1]:
                new_values[i+1] = new_values[i]
        for i, vals in enumerate(values):
            pvalue, index = vals
            qvalues[index] = new_values[i]
    return qvalues


def remove_dataset_bias(df, has_ns = False):	
	users_avgs = df[df["rating"] != -1].groupby("userId")["rating"].mean().rename("userMeanRating").to_frame().reset_index()	
	df_with_user_avgs = df.merge(users_avgs,on=["userId"], how="inner")

	# this is OK as the cases when there are null values are caused by users who after removing items not in train still remain in NS for test
	# Check if still hapens after running timeb_bbased_split again
	# assert df_with_user_avgs["userMeanRating"].isnull().values.any() == False	

	del(df)
	del(users_avgs)
	df_with_user_avgs["relevance"] = df_with_user_avgs.apply(lambda r: 2 if float(r["rating"])>= float(r["userMeanRating"]) else 1,axis=1)
	
	if(has_ns):
		df_with_user_avgs.loc[df_with_user_avgs['rating'] == -1, 'relevance'] = 0

	return df_with_user_avgs[[c for c in df_with_user_avgs.columns if c in ["userId","movieId","rating","timestamp","userMeanRating","relevance","prediction"]]]


def calculate_ndcg_score(r,prediction_column):
	def dcg_score(y_true, y_score, k=20, gains="exponential"):
		"""Discounted cumulative gain (DCG) at rank k
		Parameters
		----------
		y_true : array-like, shape = [n_samples]
		Ground truth (true relevance labels).
		y_score : array-like, shape = [n_samples]
		Predicted scores.
		k : int
		Rank.
		gains : str
		Whether gains should be "exponential" (default) or "linear".
		Returns
		-------
		DCG @k : float
		"""
		order = np.argsort(y_score)[::-1]
		y_true = np.take(y_true, order[:k])

		if gains == "exponential":
			gains = 2 ** y_true - 1
		elif gains == "linear":
			gains = y_true
		else:
			raise ValueError("Invalid gains option.")

		# highest rank is 1 so +2 instead of +1
		discounts = np.log2(np.arange(len(y_true)) + 2)
		return np.sum(gains / discounts)

	def ndcg_score(y_true, y_score, k=20, gains="exponential"):
		"""Normalized discounted cumulative gain (NDCG) at rank k
		Parameters
		----------
		y_true : array-like, shape = [n_samples]
		Ground truth (true relevance labels).
		y_score : array-like, shape = [n_samples]
		Predicted scores.
		k : int
		Rank.
		gains : str
		Whether gains should be "exponential" (default) or "linear".
		Returns
		-------
		NDCG @k : float
		"""
		best = dcg_score(y_true, y_true, k, gains)		
		actual = dcg_score(y_true, y_score, k, gains)		
		return float(actual) / best
	
	np.random.seed(42)
	r = r.iloc[np.random.permutation(len(r))]	
	predicted = r.sort_values(prediction_column,ascending=False)["relevance"].tolist()	
	true_relevance = r.sort_values("relevance",ascending=False)["relevance"].tolist()			
	# print(true_relevance)
	# print(ndcg_score(true_relevance,predicted))
	return ndcg_score(true_relevance,predicted)


def calculate_precision_score(r, prediction_column, k=20):
	np.random.seed(42)
	r = r.iloc[np.random.permutation(len(r))]
	true_relevance = r.sort_values(prediction_column,ascending=False)["relevance"].tolist()
	list_size = len(true_relevance)
	number_of_relevants =0
	for rel_score in true_relevance[0: min(k,len(true_relevance))]:
		if(rel_score == 2 or rel_score == 1):
			number_of_relevants+=1
	return number_of_relevants/float(list_size)

def evaluate_hypothesis(df_no_bias,predictions):	
	predictions.userId = predictions.userId.astype(str)
	predictions.movieId = predictions.movieId.astype(str)
	df_no_bias.userId = df_no_bias.userId.astype(str)
	df_no_bias.movieId = df_no_bias.movieId.astype(str)
	
	predictions_with_relevance = predictions.merge(df_no_bias,on=["userId","movieId"],how="left")
	predictions_with_relevance.loc[predictions_with_relevance['rating_y'].isnull(),'relevance'] = 0
	# predictions_with_relevance.loc[predictions_with_relevance['rating_x'] == -1, 'relevance'] = 0	

	scores = predictions_with_relevance.groupby("userId").agg(lambda r,f = calculate_ndcg_score: f(r,"prediction_ensemble"))
	scores = scores[[scores.columns[0]]].rename(index=str,columns={scores.columns[0]:"NDCG"}).reset_index()

	# scores_map = predictions_with_relevance.groupby("userId").agg(lambda r,f = calculate_ap_score: f(r,"prediction_ensemble"))
	# scores_map = scores_map[[scores_map.columns[0]]].rename(index=str,columns={scores.columns[0]:"MAP"}).reset_index()
	# scores_map.columns = ["userId","MAP"]

	# scores = scores.merge(scores_map,on="userId")

	# for k in precision_levels:

	# 	scores_precision = predictions_with_relevance.groupby("userId").agg(lambda r,f = calculate_precision_score,s=k: f(r,"prediction_ensemble",s))
	# 	scores_precision = scores_precision[[scores_precision.columns[0]]].rename(index=str,columns={scores_precision.columns[0]:"Precision@"+str(k)}).reset_index()

	# 	scores = scores.merge(scores_precision,on="userId")

	# predictions_no_ns = predictions[predictions["rating"]!= -1]
	# predictions_no_ns["SE"] = predictions_no_ns.apply(lambda r: (float(r["prediction_ensemble"])-float(r["rating"]))**2, axis=1)
	# rmse = math.sqrt(predictions_no_ns.mean()["SE"])


	# #MAE
	# predictions_no_ns["error"] = abs(predictions_no_ns["prediction_ensemble"]-predictions_no_ns["rating"])
	# avg_errors = predictions_no_ns.groupby("userId")["error"].mean().rename("MAE").to_frame().reset_index() 
	# avg_errors_var = predictions_no_ns.groupby("userId")["error"].var().rename("MAE_VAR").to_frame().reset_index()	
	# avg_errors_var = avg_errors_var.fillna(0.0)
	# avg_errors = avg_errors.merge(avg_errors_var,on="userId")

	# # Calculating user MSE and RMSE
	# predictions_no_ns["squared_error"] = (predictions_no_ns["prediction_ensemble"]-predictions_no_ns["rating"]) * (predictions_no_ns["prediction_ensemble"]-predictions_no_ns["rating"])
	# avg_squared_errors = predictions_no_ns.groupby("userId")["squared_error"].mean().rename("MSE").to_frame().reset_index() 
	# avg_squared_errors_var = predictions_no_ns.groupby("userId")["squared_error"].var().rename("MSE_VAR").to_frame().reset_index() 			
	# avg_squared_errors_var = avg_squared_errors_var.fillna(0.0)	
	# avg_squared_errors["RMSE"] = avg_squared_errors.apply(lambda r,math=math: math.sqrt(r["MSE"]),axis=1)
	# avg_errors = avg_errors.merge(avg_squared_errors.merge(avg_squared_errors_var,on="userId"), on="userId")
	# avg_errors["userId"] = avg_errors["userId"].astype(str)

	# # Calculating RR
	# scores_rr = predictions_with_relevance.groupby("userId").agg(lambda r,f=calculate_reciprocal_rank: f(r,"prediction_ensemble"))
	# scores_rr = scores_rr[[scores_rr.columns[0]]].rename(index=str,columns={scores_rr.columns[0]:"RR"}).reset_index()
	# scores = scores.merge(scores_rr,on="userId")

	# scores = scores.merge(avg_errors,on="userId",how="left")	

	print("finished evaluate_hypothesis")
	print(scores.shape)
	# return scores, rmse
	return scores, []

def evaluate_base_RS(df_no_bias,dataset_name, RS):
	filehandler = open("./created_data/tmp/predictions_all_H2_"+dataset_name+"_LinearReg_none_.pkl","rb")
	predictions = pickle.load(filehandler)
	predictions = predictions[[c for c in predictions if "pondered" not in c]]
	filehandler.close()
	df_no_bias.userId = df_no_bias.userId.astype(str)
	df_no_bias.movieId = df_no_bias.movieId.astype(str)
	predictions_with_relevance = predictions.merge(df_no_bias,on=["userId","movieId"],how="left")
	predictions_with_relevance.loc[predictions_with_relevance['rating_x'] == -1, 'relevance'] = 0

	all_scores = []
	rmse = []
	for rs in RS:
		if("amazon" in dataset_name and rs == "SlopeOne"):
			continue
		print("evaluate_base_RS"+ rs)
		preds = predictions_with_relevance[["userId","movieId","relevance","prediction_"+rs]]		
		scores = preds.groupby("userId").agg(lambda r,f = calculate_ndcg_score,rss=rs: f(r,"prediction_"+rss))
		scores = scores[[scores.columns[0]]].rename(index=str,columns={scores.columns[0]:"NDCG_"+rs}).reset_index()

		# scores_map = preds.groupby("userId").agg(lambda r,f = calculate_ap_score,rss=rs: f(r,"prediction_"+rss))
		# scores_map = scores_map[[scores_map.columns[0]]].rename(index=str,columns={scores.columns[0]:"MAP"}).reset_index()
		# scores_map.columns = ["userId","MAP_"+rs]

		# scores = scores.merge(scores_map,on="userId")

		# for k in precision_levels:
		# 	scores_precision = preds.groupby("userId").agg(lambda r,f = calculate_precision_score,s=k,rss=rs: f(r,"prediction_"+rss,s))
		# 	scores_precision = scores_precision[[scores_precision.columns[0]]].rename(index=str,columns={scores_precision.columns[0]:"Precision_"+rs+"@"+str(k)}).reset_index()
		# 	scores = scores.merge(scores_precision,on="userId")

		# # scores_precision = preds.groupby("userId").agg(lambda r,f = calculate_precision_score,rss=rs: f(r,"prediction_"+rss))
		# # scores_precision = scores_precision[[scores_precision.columns[0]]].rename(index=str,columns={scores_precision.columns[0]:"Precision_"+rs}).reset_index()
		
		# predictions_no_ns = predictions[predictions["rating"]!= -1][["userId","movieId","rating","prediction_"+rs]]
		# predictions_no_ns["SE"] = predictions_no_ns.apply(lambda r,rss=rs: (float(r["prediction_"+rss])-float(r["rating"]))**2, axis=1)		
		# rmse.append(math.sqrt(predictions_no_ns.mean()["SE"]))
		# # scores = scores.merge(scores_precision,on="userId")

		# #MAE
		# predictions_no_ns["error"] = abs(predictions_no_ns["prediction_"+rs]-predictions_no_ns["rating"])
		# avg_errors = predictions_no_ns.groupby("userId")["error"].mean().rename("MAE_"+rs).to_frame().reset_index() 
		# avg_errors_var = predictions_no_ns.groupby("userId")["error"].var().rename("MAE_VAR_"+rs).to_frame().reset_index()
		# avg_errors_var = avg_errors_var.fillna(0.0)		
		# avg_errors = avg_errors.merge(avg_errors_var,on="userId")

		# # Calculating user MSE and RMSE
		# predictions_no_ns["squared_error"] = (predictions_no_ns["prediction_"+rs]-predictions_no_ns["rating"]) * (predictions_no_ns["prediction_"+rs]-predictions_no_ns["rating"])
		# avg_squared_errors = predictions_no_ns.groupby("userId")["squared_error"].mean().rename("MSE_"+rs).to_frame().reset_index() 
		# avg_squared_errors_var = predictions_no_ns.groupby("userId")["squared_error"].var().rename("MSE_VAR_"+rs).to_frame().reset_index() 			
		# avg_squared_errors_var = avg_squared_errors_var.fillna(0.0)
		# avg_squared_errors["RMSE_"+rs] = avg_squared_errors.apply(lambda r,rs = rs,math=math: math.sqrt(r["MSE_"+rs]),axis=1)
		# avg_errors = avg_errors.merge(avg_squared_errors.merge(avg_squared_errors_var,on="userId"), on="userId")
		# avg_errors["userId"] = avg_errors["userId"].astype(str)

		# # Calculating RR
		# scores_rr = preds.groupby("userId").agg(lambda r,f=calculate_reciprocal_rank,rs=rs: f(r,"prediction_"+rs))
		# scores_rr = scores_rr[[scores_rr.columns[0]]].rename(index=str,columns={scores_rr.columns[0]:"RR_"+rs}).reset_index()
		# scores = scores.merge(scores_rr,on="userId")

		# scores = scores.merge(avg_errors,on="userId",how="left")		
		# assert scores.isnull().values.any() == False
		print(scores.shape)
		print(scores["NDCG_"+rs].mean())
		all_scores.append(scores)

	return reduce(lambda x,y: x.merge(y,on="userId"), all_scores), []

# def statistical_evaluation(eval_dfs, metric, df_names):
# 	tests = {}
# 	for first in range(len(eval_dfs)):
# 		for second in range(first+1,len(eval_dfs)):
# 			name = df_names[first]
# 			name_2 = df_names[second]
# 			df = eval_dfs[first]
# 			df_2 = eval_dfs[second]								
# 			tests[(name,name_2)] = stats.ttest_rel(df[metric],df_2[metric])
	
# 	pvalues_keys = []
# 	pvalues = []
# 	t_values = []
# 	for k,v in tests.items():
# 		pvalues.append(round(v[1], 6))
# 		t_values.append(abs(round(v[0],6)))
# 		pvalues_keys.append(k)	
	
# 	pvalues_corrected = multiple_testing_correction(pvalues)

# 	correct_t_tests = {}
# 	for idx, k in enumerate(pvalues_keys):
# 		correct_t_tests[k] = (t_values[idx],pvalues[idx])

# 	return correct_t_tests		

# def map_to_table(eval_dfs,names,t_tests_ndcg):
# 	t_tests_df_ndcg = []
# 	for first in range(len(eval_dfs)):
# 		line = [names[first]]
# 		for second in range(first+1,len(eval_dfs)):
# 			line.append(t_tests_ndcg[(names[first],names[second])])
# 		if(len(line)>1):
# 			t_tests_df_ndcg.append(line)	
# 	names.reverse()
# 	t_tests_df_ndcg = pd.DataFrame(t_tests_df_ndcg,columns = [""] + names[0:-1])
# 	return t_tests_df_ndcg


# def create_final_results_table(overall_t_tests,datasets):	
# 	table_final = pd.DataFrame([["Solo RS"],["SCB (best CV)"],["SCB (RF)"],["Stacking (best CV)"],["Stacking (RF)"],["FWLS"],["STREAM (best CV)"],["STREAM (RF)"],["Borda-count"],["LTRERS (LambdaMart)"]])
# 	best_solo_rs_name = ""
# 	model_dict = {}
# 	for dataset in datasets:
# 		for metric in ["precision@05","precision@10","precision@20","map","ndcg","rmse"]:
# 			df = pd.read_csv("./created_data/results/avg_metrics_"+metric+".csv")					
# 			column = []			
# 			for model in ["Solo RS","SCB","SCB (RF)","Stacking", "Stacking (RF)","FWLS","STREAM","STREAM (RF)","Borda-count","LTRERS"]:
# 				if(model=="Solo RS"):
# 					if(dataset in datasets_knn_mem_error):
# 						column.append(df[(df["dataset"]==dataset)][[rs["name"] for rs in RS if "KNN" not in rs["name"]]].as_matrix().max())
# 						best_solo_rs_name = [rs["name"] for rs in RS if "KNN" not in rs["name"]][df[df["dataset"]==dataset][[rs["name"] for rs in RS if "KNN" not in rs["name"]]].as_matrix().argmax()]
# 						model_dict["Solo RS"] = best_solo_rs_name
# 					else:						
# 						column.append(df[df["dataset"]==dataset][[rs["name"] for rs in RS]].as_matrix().max())
# 						best_solo_rs_name = [rs["name"] for rs in RS][df[df["dataset"]==dataset][[rs["name"] for rs in RS]].as_matrix().argmax()]
# 						model_dict["Solo RS"] = best_solo_rs_name
# 				if(model=="SCB"):
# 					column.append(df[df["dataset"]==dataset][["H1","H1-E","H1-E-val","H1-M"]].as_matrix().max())					
# 					model_dict["SCB (best CV)"] = ["H1","H1-E","H1-E-val","H1-M"][df[df["dataset"]==dataset][["H1","H1-E","H1-E-val","H1-M"]].as_matrix().argmax()]
# 				if(model=="SCB (RF)"):
# 					column.append(df[df["dataset"]==dataset][["H1>fixed","H1-E>fixed","H1-E-val>fixed","H1-M>fixed"]].as_matrix().max())
# 					model_dict["SCB (RF)"] = ["H1>fixed","H1-E>fixed","H1-E-val>fixed","H1-M>fixed"][df[df["dataset"]==dataset][["H1>fixed","H1-E>fixed","H1-E-val>fixed","H1-M>fixed"]].as_matrix().argmax()]
# 				if(model=="Stacking"):
# 					column.append(df[df["dataset"]==dataset][["Stacking"]].as_matrix().max())
# 					model_dict["Stacking (best CV)"] = "Stacking"
# 				if(model=="Stacking (RF)"):
# 					column.append(df[df["dataset"]==dataset][["Stacking>fixed"]].as_matrix().max())
# 					model_dict["Stacking (RF)"] = "Stacking>fixed"					
# 				if(model=="FWLS"):
# 					column.append(df[df["dataset"]==dataset][["FWLS","F-E","F-E-val","F-M"]].as_matrix().max())															
# 					model_dict["FWLS"] = ["FWLS","F-E","F-E-val","F-M"][df[df["dataset"]==dataset][["FWLS","F-E","F-E-val","F-M"]].as_matrix().argmax()]
# 				if(model=="STREAM"):
# 					column.append(df[df["dataset"]==dataset][["STREAM","S-E","S-E-val","S-M"]].as_matrix().max())										
# 					model_dict["STREAM (best CV)"] = ["STREAM","S-E","S-E-val","S-M"][df[df["dataset"]==dataset][["STREAM","S-E","S-E-val","S-M"]].as_matrix().argmax()]
# 				if(model=="STREAM (RF)"):
# 					column.append(df[df["dataset"]==dataset][["STREAM>fixed","S-E>fixed","S-E-val>fixed","S-M>fixed"]].as_matrix().max())
# 					model_dict["STREAM (RF)"] = ["STREAM>fixed","S-E>fixed","S-E-val>fixed","S-M>fixed"][df[df["dataset"]==dataset][["STREAM>fixed","S-E>fixed","S-E-val>fixed","S-M>fixed"]].as_matrix().argmax()]
# 				if(model=="Borda-count"):
# 					column.append(df[df["dataset"]==dataset][["borda-count"]].as_matrix().max())										
# 					model_dict["Borda-count"] = "borda-count"
# 				if(model=="LTRERS"):
# 					column.append(df[df["dataset"]==dataset][['l2r-all', 'l2r-E','l2r-E-val','l2r-M', 'l2r-0']].as_matrix().max())
# 					model_dict["LTRERS (LambdaMart)"] = ['l2r-all', 'l2r-E','l2r-E-val','l2r-M', 'l2r-0'][df[df["dataset"]==dataset][['l2r-all', 'l2r-E','l2r-E-val','l2r-M', 'l2r-0']].as_matrix().argmax()]

# 			table_final[dataset+"-"+metric] = pd.DataFrame(column)	

# 	metric_dict = {"precision@05": 'Precision@5',"precision@10":'Precision@10',"precision@20":"Precision@20","map":"MAP","ndcg":"NDCG","rmse":"RMSE"}		
# 	for c in table_final.columns:	
# 		if(c != 0 and c.split("-")[1] != "rmse"):
# 			t_tests_res = ""
# 			table_final[c] = table_final[c].astype(float)
# 			winning_model = table_final.loc[table_final[c].idxmax(),0]
# 			for idx, r in table_final.iterrows():
# 				if(r[0]!= winning_model):
# 					if(c.split("-")[0] in datasets_knn_mem_error and ("KNN" in winning_model  or "KNN" in model_dict[r[0]])):
# 						continue
# 					if((model_dict[winning_model],model_dict[r[0]]) in overall_t_tests[c.split("-")[0]][metric_dict[c.split("-")[1]]]):
# 						t_value,p_value = overall_t_tests[c.split("-")[0]][metric_dict[c.split("-")[1]]][(model_dict[winning_model],model_dict[r[0]])]
# 					else:
# 						t_value,p_value = overall_t_tests[c.split("-")[0]][metric_dict[c.split("-")[1]]][(model_dict[r[0]],model_dict[winning_model])]
# 					if(p_value<0.05):
# 						t_tests_res+= str(idx)
# 			table_final.loc[table_final[c].idxmax(),c] = "<b>" + str(table_final.loc[table_final[c].idxmax()][c])  + "</b>^"+ t_tests_res
# 			table_final[c] = table_final.apply(lambda r,c=c: str(r[c]).replace(".",","),axis=1)
# 		elif(type(c) == str and c.split("-")[1] == "rmse"):
# 			table_final[c] = table_final.apply(lambda r,c=c: str(r[c]).replace(".",","),axis=1)
# 	table_final[0] = table_final[0].apply(lambda r,b=best_solo_rs_name: 'Solo RS ('+b+')' if r == "Solo RS" else r)		
# 	table_final.to_csv("./results_scripts/final_results.csv",sep="|")

# def create_ef_vs_mf_table(overall_t_tests,datasets):	
# 	table_final = pd.DataFrame([["SCB-EF (RF)"],["SCB-EF-val (RF)"],["SCB-MF (RF)"],["SCB-All (RF)"],[" SCB-oracle"],["FWLS-EF"],["FWLS-EF-val"],["FWLS-MF"],["FWLS-All"],["Stacking (RF)"], ["STREAM-EF (RF)"],["STREAM-EF-val (RF)"],["STREAM-MF (RF)"],["STREAM-All (RF)"],["LTRERS"],["LTRERS-EF"],["LTRERS-EF-val"],["LTRERS-MF"],["LTRERS-All"]])
# 	counts = [0,0,0,0]
# 	for dataset in datasets:
# 		for metric in ["precision@05","precision@10","precision@20","map","ndcg","rmse"]:
# 			df = pd.read_csv("./created_data/results/avg_metrics_"+metric+".csv")
# 			column = []			
# 			combs = ["EF","MF","All"]
# 			model_values = []
# 			for model in ["H1-E>fixed","H1-E-val>fixed","H1-M>fixed","H1>fixed","H1 oracle","F-E","F-E-val","F-M","FWLS","Stacking>fixed","S-E>fixed","S-E-val>fixed","S-M>fixed","STREAM>fixed","l2r-0","l2r-E","l2r-E-val","l2r-M","l2r-all"]:				
# 				column.append(df[df["dataset"]==dataset][model].as_matrix()[0])						
# 			table_final[dataset+"-"+metric] = pd.DataFrame(column)

# 	table_final["method"] = table_final.apply(lambda r: r[0].split("-")[0],axis=1)

# 	metric_dict = {"precision@05": 'Precision@5',"precision@10":'Precision@10',"precision@20":"Precision@20","map":"MAP","ndcg":"NDCG", "rmse": "RMSE"}	
# 	model_dict = {"Stacking (RF)": "Stacking>fixed","SCB-EF (RF)": "H1-E>fixed" ,"SCB-EF-val (RF)": "H1-E-val>fixed" ,"SCB-MF (RF)": "H1-M>fixed" ,"SCB-All (RF)": "H1>fixed" ," SCB-oracle": "H1 oracle" ,"FWLS-EF": "F-E" ,"FWLS-EF-val": "F-E-val" ,"FWLS-MF": "F-M" ,"FWLS-All": "FWLS" ,"STREAM-EF (RF)": "S-E>fixed" ,"STREAM-EF-val (RF)": "S-E-val>fixed" ,"STREAM-MF (RF)": "S-M>fixed" ,"STREAM-All (RF)": "STREAM>fixed" ,"LTRERS": "l2r-0","LTRERS-EF": "l2r-E" ,"LTRERS-EF-val": "l2r-E-val" ,"LTRERS-MF": "l2r-M" ,"LTRERS-All": "l2r-all"}
	
# 	replacements = {}
# 	for c in table_final:
# 		if (c != "method" and c!= 0 and c.split("-")[1] != "rmse"):			
# 			table_final.loc[table_final.groupby("method")[c].idxmax(),c] = table_final.loc[table_final.groupby("method")[c].idxmax()][c].apply(lambda r: "<b>"+ str(r) + "</b>")
# 			table_final[c] = table_final.apply(lambda r,c=c: str(r[c]).replace(".",","),axis=1)

# 			for idx, r in table_final.iterrows():
# 				if("<b>" in r[c] and "oracle" not in r[0]):
# 					m = r[0].split("-")[0]
# 					winning_model = r[0]
# 					t_tests_res = ""
# 					i=0
# 					for idx2, r2 in table_final.iterrows():
# 						m2 = r2[0].split("-")[0]
# 						if(m == m2):
# 							if(r2[0]!= r[0]):
# 								if((model_dict[winning_model],model_dict[r2[0]]) in overall_t_tests[c.split("-")[0]][metric_dict[c.split("-")[1]]]):
# 									t_value,p_value = overall_t_tests[c.split("-")[0]][metric_dict[c.split("-")[1]]][(model_dict[winning_model],model_dict[r2[0]])]
# 								else:
# 									t_value,p_value = overall_t_tests[c.split("-")[0]][metric_dict[c.split("-")[1]]][(model_dict[r2[0]],model_dict[winning_model])]

# 								if(p_value<0.05):
# 									t_tests_res+= str(i)
# 							i+=1
# 					replacements[(winning_model,c)] = t_tests_res
# 		elif(type(c) == str and c!="method" and c.split("-")[1] == "rmse"):
# 			table_final[c] = table_final.apply(lambda r,c=c: str(r[c]).replace(".",","),axis=1)

# 	for c in table_final:
# 		table_final[c] = table_final.apply(lambda r,c=c,m=replacements: r[c] + " ^"+ m[(r[0],c)] if (r[0],c) in m else r[c],axis=1)
	
# 	table_final[[c for c in table_final if c != "method"]].to_csv("./results_scripts/MF_vs_EF.csv",sep="|")

def main():
	parser = optparse.OptionParser()
	parser.add_option('-d', '--datasets', 
						dest="datasets")

	options, remainder = parser.parse_args()	

	datasets = options.datasets.split(",")

	print(datasets)
	
	avg_metrics_ndcg_df = []
	avg_metrics_precision_df = [[] for i in range(0,len(precision_levels))]
	avg_metrics_rmse = []
	avg_metrics_map = []
	all_scores_raw_df = []
	robustness_dfs = []
	overall_t_tests = {}

	for dataset_name in datasets:
		overall_t_tests[dataset_name] = {}

		print("Dataset "+dataset_name+ "\n")
		if(not os.path.exists("./created_data/tmp/"+dataset_name+"_with_relevance.pkl")):
			filehandler = open("./created_data/tmp/predictions_all_H2_"+dataset_name+"_LinearReg_none_.pkl","rb")
			predictions_example = pickle.load(filehandler)
			filehandler.close()
			users_in_test = predictions_example.userId.unique()

			print("creating dataset with relevance labels")
			df = pd.read_csv("./created_data/"+dataset_name+".csv")
			df = df[df.userId.isin(users_in_test)]

			df_no_bias = remove_dataset_bias(df)

			filehandler = open("./created_data/tmp/"+dataset_name+"_with_relevance.pkl",'wb')
			pickle.dump(df_no_bias,filehandler)
			filehandler.close()
			# df_no_bias.to_csv("./created_data/tmp/"+dataset_name+"_with_relevance.csv",index=False,header=True)

		filehandler = open("./created_data/tmp/"+dataset_name+"_with_relevance.pkl", 'rb')
		# df_no_bias = pd.read_csv("./created_data/tmp/"+dataset_name+"_with_relevance.csv")
		df_no_bias = pickle.load(filehandler)
		filehandler.close()
		scores_robustness_analysis_pp = []
		scores_robustness_analysis_pe = []
		
		incremental_analysis = False
		if(incremental_analysis):
			for i in range(1,32):
				scores = evaluate_hypothesis(df_no_bias,pd.read_csv("../experiment3/created_data/predictions/predictions_incremental_analysis_"+dataset_name+"_"+str(i)+"_pe.csv")[["userId","movieId","rating","prediction_ensemble"]])
				ci = confidence_interval(list(scores[0]["NDCG"]))
				scores_robustness_analysis_pe.append([i,scores[0]["NDCG"].mean(),ci])
				print(scores[0]["NDCG"].mean())
			# for i in range(1,3):
			for i in range(1,15):
				scores = evaluate_hypothesis(df_no_bias,pd.read_csv("../experiment3/created_data/predictions/predictions_incremental_analysis_"+dataset_name+"_"+str(i)+"_pp.csv")[["userId","movieId","rating","prediction_ensemble"]])
				ci = confidence_interval(list(scores[0]["NDCG"]))
				scores_robustness_analysis_pp.append([i,scores[0]["NDCG"].mean(),ci])
				print(scores[0]["NDCG"].mean())

			scores_robustness_analysis_pp = pd.DataFrame(scores_robustness_analysis_pp,columns = ["f_index","NDCG@20","ci"])
			scores_robustness_analysis_pp["input_space"] = "PP"
			scores_robustness_analysis_pe = pd.DataFrame(scores_robustness_analysis_pe,columns = ["f_index","NDCG@20","ci"])
			scores_robustness_analysis_pe["input_space"] = "PE"

			print("PP")
			print(scores_robustness_analysis_pp)
			print("PE")
			print(scores_robustness_analysis_pe)
			pd.concat([scores_robustness_analysis_pp,scores_robustness_analysis_pe]).to_csv("./created_data/incremental_analysis.csv")
			# embed()
			# exit()
		scores_robustness_analysis = []
		print("Calculating Scores for all STREAM regressors")
		for m in models:
			for features_used in ["none","MF","EF-val-raw","EF-train-raw","EF-val-pondered","EF-train-pondered"]:
				if(features_used in ["none","MF"]):
					estimates = [""]
				else:
					estimates = ["NDCG","RMSE"]

				for performance_estimate in estimates:
					filehandler = open("./created_data/tmp/predictions_all_H2_"+dataset_name+"_"+m[0]+"_" + features_used +"_"+performance_estimate+".pkl",'rb')
					df = pickle.load(filehandler)
					filehandler.close()				
					score_model_stream,rmse_model_stream = evaluate_hypothesis(df_no_bias,df[["userId","movieId","rating","prediction_ensemble"]])
					score_model_stream["model"] = m[0]
					score_model_stream["features"] = features_used
					score_model_stream["performance_estimate"] = performance_estimate
					score_model_stream["ensemble"] = "STREAM"
					score_model_stream["is_all"] = False
					print(m[0])
					print(features_used)
					print(performance_estimate)
					print(score_model_stream["NDCG"].mean())
					scores_robustness_analysis.append(score_model_stream)		
					print("\n")

			for features_used in ["EF-val-raw_MF","EF-train-raw_MF","EF-val-pondered_MF","EF-train-pondered_MF"]:
				estimates = ["NDCG","RMSE"]
				for performance_estimate in estimates:
					filehandler = open("./created_data/tmp/predictions_all_H2_"+dataset_name+"_"+m[0]+"_" + features_used +"_"+performance_estimate+".pkl",'rb')
					df = pickle.load(filehandler)
					filehandler.close()				
					score_model_stream,rmse_model_stream = evaluate_hypothesis(df_no_bias,df[["userId","movieId","rating","prediction_ensemble"]])
					score_model_stream["model"] = m[0]
					score_model_stream["features"] = features_used.split("_")[0]
					score_model_stream["performance_estimate"] = performance_estimate
					score_model_stream["ensemble"] = "STREAM"
					score_model_stream["is_all"] = True
					print(m[0])
					print(features_used)
					print(performance_estimate)
					print(score_model_stream["NDCG"].mean())
					scores_robustness_analysis.append(score_model_stream)		
					print("\n")	

		print("Calculating Scores for all LTRERS l2r")
		for m in L2R:
			for features_used in ["none_","MF_","EF-val-pondered_RMSE","EF-train-pondered_RMSE","EF-val-pondered_MF_RMSE","EF-train-pondered_MF_RMSE","EF-val-pondered_NDCG","EF-train-pondered_NDCG","EF-val-pondered_MF_NDCG","EF-train-pondered_MF_NDCG","EF-val-raw_RMSE","EF-train-raw_RMSE","EF-val-raw_MF_RMSE","EF-train-raw_MF_RMSE","EF-val-raw_NDCG","EF-train-raw_NDCG","EF-val-raw_MF_NDCG","EF-train-raw_MF_NDCG"]:
				score_model_ltrers,rmse_model_ltrers = evaluate_hypothesis(df_no_bias,pd.read_csv("../experiment3/created_data/predictions/predictions_l2r_robustness_analysis_"+ m["name"]+"_"+dataset_name+"_"+features_used+".csv")[["userId","movieId","rating","prediction_ensemble"]])
				score_model_ltrers["model"] = m["name"]
				score_model_ltrers["features"] = features_used.split("_")[0]
				score_model_ltrers["ensemble"] = "LTRERS"
				score_model_ltrers["is_all"] = (features_used != "MF_" and "MF" in features_used)				
				p_estimate = ""
				if "NDCG" in features_used:
					p_estimate = "NDCG"
				if "RMSE" in features_used:
					p_estimate = "RMSE"

				score_model_ltrers["performance_estimate"] = p_estimate

				print(m["name"])
				print(features_used)
				print(score_model_ltrers["NDCG"].mean())
				print("\n")
				scores_robustness_analysis.append(score_model_ltrers)
		

		print("Borda-count")
		scores_borda_count,rmse_borda_count = evaluate_hypothesis(df_no_bias,pd.read_csv("../experiment3/created_data/predictions/predictions_borda_count_"+dataset_name+".csv")[["userId","movieId","rating","prediction_ensemble"]])
		scores_borda_count["model"] = "borda-count"
		print(scores_borda_count["NDCG"].mean())


		print("Calculating base RS scores.")
		rs_list = [rs["name"] for rs in RS]
		if(dataset_name in datasets_knn_mem_error):
			rs_list = [rs["name"] for rs in RS if "KNN" not in rs["name"]]
		if("amazon" in dataset_name):
			rs_list = [rs["name"] for rs in RS if "SlopeOne" not in rs["name"]]

		rs_scores, RMSEs = evaluate_base_RS(df_no_bias,dataset_name,rs_list)
		rs_scores.to_csv("./created_data/tmp/h2_"+dataset_name+"_user_train_time_features_test_set.csv")


		# df_test = pd.read_csv("./created_data/tmp/predictions_H1"+dataset_name+"_all.csv")
		# users_with_only_negative = df_test.userId.unique().shape[0] - df_test[df_test.rating != -1].userId.unique().shape[0]
		# print(users_with_only_negative)			
	
		# l2r_features_map = {"all":"all","preds_only":"none","meta_features_only":"meta-features","error_features_only":"error-features","error_features_val_set_only":"error-features-val","error_features_val_set_pondered":"EF-val-pondered"}

		# print("Calculating ensemble scores.")

		# print("FWLS")
		# scores_fwls_baseline,rmse__fwls_baseline = evaluate_hypothesis(df_no_bias,pd.read_csv("./created_data/tmp/predictions_H2"+dataset_name+"_FWLS_baseline_all.csv")[["userId","movieId","rating","prediction_ensemble"]])
		# scores_fwls_baseline["model"] = "FWLS-All"
		# scores_fwls_error_only,rmse__fwls_error_only = evaluate_hypothesis(df_no_bias,pd.read_csv("./created_data/tmp/predictions_H2"+dataset_name+"_FWLS_baseline_error-features.csv")[["userId","movieId","rating","prediction_ensemble"]])
		# scores_fwls_error_only["model"] = "FWLS-EF-train"
		# scores_fwls_val_error_only,rmse__fwls_val_error_only = evaluate_hypothesis(df_no_bias,pd.read_csv("./created_data/tmp/predictions_H2"+dataset_name+"_FWLS_baseline_error-features-val.csv")[["userId","movieId","rating","prediction_ensemble"]])
		# scores_fwls_val_error_only["model"] = "FWLS-EF-val"
		# scores_fwls_meta_only,rmse__fwls_meta_only = evaluate_hypothesis(df_no_bias,pd.read_csv("./created_data/tmp/predictions_H2"+dataset_name+"_FWLS_baseline_meta-features.csv")[["userId","movieId","rating","prediction_ensemble"]])						
		# scores_fwls_meta_only["model"] = "FWLS-MF"
		
		# score_model_fwls = scores_fwls_baseline.copy()
		# score_model_fwls["model"] = "LinearRegression"
		# score_model_fwls["features"] = "all"
		# score_model_fwls["ensemble"] = "FWLS"
		# scores_robustness_analysis.append(score_model_fwls)

		# score_model_fwls = scores_fwls_error_only.copy()
		# score_model_fwls["model"] = "LinearRegression"
		# score_model_fwls["features"] = "error-features"
		# score_model_fwls["ensemble"] = "FWLS"
		# scores_robustness_analysis.append(score_model_fwls)
		
		# score_model_fwls = scores_fwls_val_error_only.copy()
		# score_model_fwls["model"] = "LinearRegression"
		# score_model_fwls["features"] = "error-features-val"
		# score_model_fwls["ensemble"] = "FWLS"
		# scores_robustness_analysis.append(score_model_fwls)

		# score_model_fwls = scores_fwls_meta_only.copy()
		# score_model_fwls["model"] = "LinearRegression"
		# score_model_fwls["features"] = "meta-features"
		# score_model_fwls["ensemble"] = "FWLS"
		# scores_robustness_analysis.append(score_model_fwls)

		scores_robustness_analysis_df = pd.concat(scores_robustness_analysis)
		scores_robustness_analysis_df["dataset"] = dataset_name		
		robustness_dfs.append(scores_robustness_analysis_df)

		all_scores = [scores_borda_count]
		all_scores = pd.concat(all_scores)
		all_scores["dataset"] = dataset_name
		all_scores_raw_df.append(all_scores)
		# """
		#  	  AVG NDCG, PRECISION and RMSE tables
		# """
		# print("calculating lines for avg metrics table.")		
		# #MAP
		# avg_metrics_line_map = [dataset_name,
		# 	scores_borda_count.mean()["MAP"], \
		# 	scores_l2r_all.mean()["MAP"],scores_l2r_error_only.mean()["MAP"],scores_l2r_val_error_only.mean()["MAP"],scores_l2r_meta_only.mean()["MAP"],scores_l2r_no_features.mean()["MAP"], \
		# 	scores_h1.mean()["MAP"],scores_h1_error_only.mean()["MAP"],scores_h1_val_error_only.mean()["MAP"],scores_h1_meta_only.mean()["MAP"],scores_h1_oracle.mean()["MAP"],\
		# 	scores_h2.mean()["MAP"],scores_h2_error_only.mean()["MAP"],scores_h2_val_error_only.mean()["MAP"],scores_h2_meta_only.mean()["MAP"],scores_h2_no_meta_features.mean()["MAP"],\
		# 	scores_fwls_baseline.mean()["MAP"],scores_fwls_error_only.mean()["MAP"],scores_fwls_val_error_only.mean()["MAP"],scores_fwls_meta_only.mean()["MAP"],
		# 	scores_h1_fixed.mean()["MAP"],scores_h1_error_only_fixed.mean()["MAP"],scores_h1_val_error_only_fixed.mean()["MAP"],scores_h1_meta_only_fixed.mean()["MAP"],\
		# 	scores_h2_fixed.mean()["MAP"],scores_h2_error_only_fixed.mean()["MAP"],scores_h2_val_error_only_fixed.mean()["MAP"],scores_h2_meta_only_fixed.mean()["MAP"],scores_h2_no_meta_features_fixed.mean()["MAP"]]

		# for rs in rs_list:
		# 	avg_metrics_line_map.append(rs_scores.mean()["MAP_"+rs])

		# # NDCG		
		# avg_metrics_line_ndcg = [dataset_name,
		# 	scores_borda_count.mean()["NDCG"], \
		# 	scores_l2r_all.mean()["NDCG"],scores_l2r_error_only.mean()["NDCG"],scores_l2r_val_error_only.mean()["NDCG"],scores_l2r_meta_only.mean()["NDCG"],scores_l2r_no_features.mean()["NDCG"], \
		# 	scores_h1.mean()["NDCG"],scores_h1_error_only.mean()["NDCG"],scores_h1_val_error_only.mean()["NDCG"],scores_h1_meta_only.mean()["NDCG"],scores_h1_oracle.mean()["NDCG"],\
		# 	scores_h2.mean()["NDCG"],scores_h2_error_only.mean()["NDCG"],scores_h2_val_error_only.mean()["NDCG"],scores_h2_meta_only.mean()["NDCG"],scores_h2_no_meta_features.mean()["NDCG"],\
		# 	scores_fwls_baseline.mean()["NDCG"],scores_fwls_error_only.mean()["NDCG"],scores_fwls_val_error_only.mean()["NDCG"],scores_fwls_meta_only.mean()["NDCG"],
		# 	scores_h1_fixed.mean()["NDCG"],scores_h1_error_only_fixed.mean()["NDCG"],scores_h1_val_error_only_fixed.mean()["NDCG"],scores_h1_meta_only_fixed.mean()["NDCG"],\
		# 	scores_h2_fixed.mean()["NDCG"],scores_h2_error_only_fixed.mean()["NDCG"],scores_h2_val_error_only_fixed.mean()["NDCG"],scores_h2_meta_only_fixed.mean()["NDCG"],scores_h2_no_meta_features_fixed.mean()["NDCG"]]

		# for rs in rs_list:
		# 	avg_metrics_line_ndcg.append(rs_scores.mean()["NDCG_"+rs])

		# # precision		
		# avg_metrics_line_precision = [[] for i in range(0,len(precision_levels))]
		# for i in range(0,len(precision_levels)):
		# 	avg_metrics_line_precision[i] = [dataset_name, 
		# 		scores_borda_count.mean()["Precision@"+str(precision_levels[i])], \
		# 		scores_l2r_all.mean()["Precision@"+str(precision_levels[i])],scores_l2r_error_only.mean()["Precision@"+str(precision_levels[i])],scores_l2r_val_error_only.mean()["Precision@"+str(precision_levels[i])],scores_l2r_meta_only.mean()["Precision@"+str(precision_levels[i])],scores_l2r_no_features.mean()["Precision@"+str(precision_levels[i])],\
		# 		scores_h1.mean()["Precision@"+str(precision_levels[i])],scores_h1_error_only.mean()["Precision@"+str(precision_levels[i])],scores_h1_val_error_only.mean()["Precision@"+str(precision_levels[i])],scores_h1_meta_only.mean()["Precision@"+str(precision_levels[i])],scores_h1_oracle.mean()["Precision@"+str(precision_levels[i])],\
		# 		scores_h2.mean()["Precision@"+str(precision_levels[i])],scores_h2_error_only.mean()["Precision@"+str(precision_levels[i])],scores_h2_val_error_only.mean()["Precision@"+str(precision_levels[i])],scores_h2_meta_only.mean()["Precision@"+str(precision_levels[i])],scores_h2_no_meta_features.mean()["Precision@"+str(precision_levels[i])],\
		# 		scores_fwls_baseline.mean()["Precision@"+str(precision_levels[i])],scores_fwls_error_only.mean()["Precision@"+str(precision_levels[i])],scores_fwls_val_error_only.mean()["Precision@"+str(precision_levels[i])],scores_fwls_meta_only.mean()["Precision@"+str(precision_levels[i])],
		# 		scores_h1_fixed.mean()["Precision@"+str(precision_levels[i])],scores_h1_error_only_fixed.mean()["Precision@"+str(precision_levels[i])],scores_h1_val_error_only_fixed.mean()["Precision@"+str(precision_levels[i])],scores_h1_meta_only_fixed.mean()["Precision@"+str(precision_levels[i])],\
		# 		scores_h2_fixed.mean()["Precision@"+str(precision_levels[i])],scores_h2_error_only_fixed.mean()["Precision@"+str(precision_levels[i])],scores_h2_val_error_only_fixed.mean()["Precision@"+str(precision_levels[i])],scores_h2_meta_only_fixed.mean()["Precision@"+str(precision_levels[i])],scores_h2_no_meta_features_fixed.mean()["Precision@"+str(precision_levels[i])]]
				
		# 	for rs in rs_list:
		# 		avg_metrics_line_precision[i].append(rs_scores.mean()["Precision_"+rs+"@"+str(precision_levels[i])])
	
		# #RMSE
		# avg_metrics_line_rmse = [dataset_name,		
		# 	"-","-","-","-","-","-",#rmse_borda_count, rmse__l2r_all,rmse__l2r_error_only,rmse__l2r_meta_only,rmse__l2r_no_features,
		# 	rmse__h1,rmse__h1_error_only,rmse__h1_val_error_only,rmse__h1_meta_only,rmse__h1_oracle,
		# 	rmse__h2,rmse__h2_error_only,rmse__h2_val_error_only,rmse__h2_meta_only,rmse__h2_no_meta_features,
		# 	rmse__fwls_baseline,rmse__fwls_error_only,rmse__fwls_val_error_only,rmse__fwls_meta_only,
		# 	rmse__h1_fixed,rmse__h1_error_only_fixed,rmse__h1_val_error_only_fixed,rmse__h1_meta_only_fixed,
		# 	rmse__h2_fixed,rmse__h2_error_only_fixed,rmse__h2_val_error_only_fixed,rmse__h2_meta_only_fixed,rmse__h2_no_meta_features_fixed,
		# 	] + RMSEs

		# for model in [rs["name"] for rs in RS if "KNN" in rs["name"]]:
		# 	if(model not in rs_list):
		# 		avg_metrics_line_ndcg.append("-")
		# 		for i in range(0,len(precision_levels)):
		# 			avg_metrics_line_precision[i].append("-")
		# 		avg_metrics_line_rmse.append("-")
		# 		avg_metrics_line_map.append("-")

		# avg_metrics_ndcg_df.append([avg_metrics_line_ndcg[0]] + [round(v,3) if type(v) != str else "-" for v in avg_metrics_line_ndcg[1:]])		
		# for i in range(0,len(precision_levels)):
		# 	avg_metrics_precision_df[i].append([avg_metrics_line_precision[i][0]] + [round(v,3) if type(v) != str else "-" for v in avg_metrics_line_precision[i][1:]])
		# avg_metrics_rmse.append([avg_metrics_line_rmse[0]] + [round(v,3) if type(v) != str else "-" for v in avg_metrics_line_rmse[1:]])
		# avg_metrics_map.append([avg_metrics_line_map[0]] + [round(v,3) if type(v) != str else "-" for v in avg_metrics_line_map[1:]])

		# """
	 # 	   NDCG, PRECISION and RMSE statistical tests
		# """
		# print("calculating statistical tests")		
		# #MAP
		# eval_dfs = [
		# 	scores_borda_count,\
		# 	scores_l2r_all,scores_l2r_error_only,scores_l2r_val_error_only,scores_l2r_meta_only,scores_l2r_no_features,\
		# 	scores_h1,scores_h1_error_only,scores_h1_val_error_only,scores_h1_meta_only,scores_h1_oracle,\
		# 	scores_h2,scores_h2_error_only,scores_h2_val_error_only,scores_h2_meta_only,scores_h2_no_meta_features,\
		# 	scores_fwls_baseline,scores_fwls_error_only,scores_fwls_val_error_only,scores_fwls_meta_only,
		# 	scores_h1_fixed,scores_h1_error_only_fixed,scores_h1_val_error_only_fixed,scores_h1_meta_only_fixed,\
		# 	scores_h2_fixed,scores_h2_error_only_fixed,scores_h2_val_error_only_fixed,scores_h2_meta_only_fixed,scores_h2_no_meta_features_fixed]

		# names = ["borda-count","l2r-all","l2r-E","l2r-E-val","l2r-M","l2r-0","H1","H1-E","H1-E-val","H1-M",\
		# 		 "H1 oracle","STREAM","S-E","S-E-val","S-M","Stacking","FWLS", "F-E","F-E-val","F-M",
		# 		 "H1>fixed","H1-E>fixed","H1-E-val>fixed","H1-M>fixed",
		# 		 "STREAM>fixed","S-E>fixed","S-E-val>fixed","S-M>fixed","Stacking>fixed"]
		# names = names + rs_list
		# for rs in rs_list:
		# 	df = rs_scores
		# 	df["MAP"] = df["MAP_"+rs]
		# 	df = df[["MAP"]]
		# 	eval_dfs.append(df)
		
		# t_tests_map = statistical_evaluation(eval_dfs,"MAP",names)		
		# overall_t_tests[dataset_name]["MAP"] = t_tests_map
		# t_tests_df_map = map_to_table(eval_dfs,names,t_tests_map)
		# t_tests_df_map.to_csv("./created_data/results/t_tests_map_"+dataset_name+".csv",index=False)

		# #NDCG
		# eval_dfs = [
		# 	scores_borda_count,\
		# 	scores_l2r_all,scores_l2r_error_only,scores_l2r_val_error_only,scores_l2r_meta_only,scores_l2r_no_features,\
		# 	scores_h1,scores_h1_error_only,scores_h1_val_error_only,scores_h1_meta_only,scores_h1_oracle,\
		# 	scores_h2,scores_h2_error_only,scores_h2_val_error_only,scores_h2_meta_only,scores_h2_no_meta_features,\
		# 	scores_fwls_baseline,scores_fwls_error_only,scores_fwls_val_error_only,scores_fwls_meta_only,
		# 	scores_h1_fixed,scores_h1_error_only_fixed,scores_h1_val_error_only_fixed,scores_h1_meta_only_fixed,\
		# 	scores_h2_fixed,scores_h2_error_only_fixed,scores_h2_val_error_only_fixed,scores_h2_meta_only_fixed,scores_h2_no_meta_features_fixed]

		# names = ["borda-count","l2r-all","l2r-E","l2r-E-val","l2r-M","l2r-0","H1","H1-E","H1-E-val","H1-M",\
		# 	 "H1 oracle","STREAM","S-E","S-E-val","S-M","Stacking","FWLS", "F-E","F-E-val","F-M",
		# 	 "H1>fixed","H1-E>fixed","H1-E-val>fixed","H1-M>fixed",
		# 	 "STREAM>fixed","S-E>fixed","S-E-val>fixed","S-M>fixed","Stacking>fixed"]
		# names = names + rs_list
		# for rs in rs_list:
		# 	df = rs_scores
		# 	df["NDCG"] = df["NDCG_"+rs]
		# 	df = df[["NDCG"]]
		# 	eval_dfs.append(df)
		
		# t_tests_ndcg = statistical_evaluation(eval_dfs,"NDCG",names)
		# overall_t_tests[dataset_name]["NDCG"] = t_tests_ndcg
		# t_tests_df_ndcg = map_to_table(eval_dfs,names,t_tests_ndcg)
		# t_tests_df_ndcg.to_csv("./created_data/results/t_tests_ndcg_"+dataset_name+".csv",index=False)

		# #precision
		# for i in range(0,len(precision_levels)):
		# 	eval_dfs = [
		# 		scores_borda_count,\
		# 		scores_l2r_all,scores_l2r_error_only,scores_l2r_val_error_only,scores_l2r_meta_only,scores_l2r_no_features,\
		# 		scores_h1,scores_h1_error_only,scores_h1_val_error_only,scores_h1_meta_only,scores_h1_oracle,\
		# 		scores_h2,scores_h2_error_only,scores_h2_val_error_only,scores_h2_meta_only,scores_h2_no_meta_features,\
		# 		scores_fwls_baseline,scores_fwls_error_only,scores_fwls_val_error_only,scores_fwls_meta_only,
		# 		scores_h1_fixed,scores_h1_error_only_fixed,scores_h1_val_error_only_fixed,scores_h1_meta_only_fixed,\
		# 		scores_h2_fixed,scores_h2_error_only_fixed,scores_h2_val_error_only_fixed,scores_h2_meta_only_fixed,scores_h2_no_meta_features_fixed]
		# 	for rs in rs_list:
		# 		df = rs_scores
		# 		df["Precision@"+str(precision_levels[i])] = df["Precision_"+rs+ "@"+str(precision_levels[i])]
		# 		df = df[["Precision@"+str(precision_levels[i])]]
		# 		eval_dfs.append(df)

		# 	t_tests_precision = statistical_evaluation(eval_dfs,"Precision@"+str(precision_levels[i]),names)
		# 	overall_t_tests[dataset_name]["Precision@"+str(precision_levels[i])] = t_tests_precision
		# 	t_tests_df_precision = map_to_table(eval_dfs,names,t_tests_precision)
		# 	if(precision_levels[i]<10):
		# 		t_tests_df_precision.to_csv("./created_data/results/t_tests_precision@0"+str(precision_levels[i])+ "_"+dataset_name+".csv",index=False)			
		# 	else:
		# 		t_tests_df_precision.to_csv("./created_data/results/t_tests_precision@"+str(precision_levels[i])+ "_"+dataset_name+".csv",index=False)		
	
	# for model in [rs["name"] for rs in RS if "KNN" in rs["name"]]:
	# 	if(model not in rs_list):
	# 		names = names + [model]

	robustness_dfs = pd.concat(robustness_dfs)
	robustness_dfs.to_csv("./created_data/results/robustness_analysis.csv",header=True,index=False)

	all_scores_raw_df = pd.concat(all_scores_raw_df)
	all_scores_raw_df.to_csv("./created_data/results/raw_eval.csv")

	# avg_metrics_map_df = pd.DataFrame(avg_metrics_map, columns = ["dataset"] + names )
	# avg_metrics_map_df.to_csv("./created_data/results/avg_metrics_map.csv",index=False)
			
	# avg_metrics_ndcg_df = pd.DataFrame(avg_metrics_ndcg_df, columns = ["dataset"] + names )
	# avg_metrics_ndcg_df.to_csv("./created_data/results/avg_metrics_ndcg.csv",index=False)

	# for i in range(0,len(precision_levels)):
	# 	avg_metrics_precision_df_at = pd.DataFrame(avg_metrics_precision_df[i], columns = ["dataset"] +  names )
	# 	if(precision_levels[i]<10):
	# 		avg_metrics_precision_df_at.to_csv("./created_data/results/avg_metrics_precision@0"+str(precision_levels[i])+".csv",index=False)
	# 	else:
	# 		avg_metrics_precision_df_at.to_csv("./created_data/results/avg_metrics_precision@"+str(precision_levels[i])+".csv",index=False)

	# avg_metrics_rmse = pd.DataFrame(avg_metrics_rmse, columns = ["dataset"] +  names )
	# avg_metrics_rmse.to_csv("./created_data/results/avg_metrics_rmse.csv",index=False)

	# print("Results written to files in ./created_data/results")

	# table_legends = []
	# for name in datasets:
	# 	line = "|map "+name+"|ndcg "+name+''.join(["|precision@"+ str(k)+" "+name for k in precision_levels])
	#  	table_legends.append(line)

	# os.system("python tably.py ./created_data/results/*.csv > ./created_data/results/tables_tex.tex -p -c \""+ "MAP@20|NDCG@20|"+ '|'.join(["Precision@"+str(k) for k in precision_levels])+ "|RMSE"+''.join(table_legends)+"\"")
	# os.system("pdflatex -interaction nonstopmode -output-directory ./created_data/results/ ./created_data/results/tables_tex.tex > /dev/null")

	# create_ef_vs_mf_table(overall_t_tests,datasets)	
	# create_final_results_table(overall_t_tests,datasets)

if __name__ == "__main__":
	main()