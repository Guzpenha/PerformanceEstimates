import optparse

from config import *
from IPython import embed

import os.path

import pandas as pd
import numpy as np

# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import confusion_matrix,f1_score,roc_curve

from surprise import dump
from surprise import Dataset, Reader

from calculate_user_features import UserFeatures
from create_hypothesis_dataset import *
from evaluate_ensembles import *

import warnings

warnings.filterwarnings('ignore')

random_state=42

models_clfs = [
		  # ("XGB",XGBClassifier(seed=random_state,objective = "multi:softmax")),
		  ("NaiveBayes",GaussianNB()),
		  ("RF",RandomForestClassifier(random_state=random_state)),
          ("AdaBoost",AdaBoostClassifier(random_state=random_state)),
          ("LR",LogisticRegression(random_state=random_state)),
          ("SVC",SVC(random_state=random_state,probability=True)),
          ("MLP",MLPClassifier(random_state=random_state)),
          ("KNN",KNeighborsClassifier())
          ]

hyperparameters_clfs = [	 
					 # [("max_depth",[15,20]),("n_estimators",[100,200])],
					 [],
					 [("max_depth",[5,10]),("n_estimators",[100,200])],
			  	     [("n_estimators",[100,100])],
			  	     [("C",[1.0,0.95,0.9])],
					 [],
					 [("hidden_layer_sizes",[(200,),(150,),(100,100),(300,)])],
			  	     [("n_neighbors",[5,6,7,10])]
			  	]

def model_selection(X,y, rep=10, random_state=42):
	""" Uses grid search and cross validation to choose the best clf for the task (X,y)"""

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

	# models = [
	# 		 ## ("XGB",XGBClassifier(seed=random_state,objective = "multi:softmax")),
	# 		  ("NaiveBayes",GaussianNB()),
	# 		  ("RF",RandomForestClassifier(random_state=random_state)),
	#           ("AdaBoost",AdaBoostClassifier(random_state=random_state)),
	#           ("LR",LogisticRegression(random_state=random_state)),
	#           ## ("linSVC",LinearSVC(multi_class="ovr")),
	#           ("SVC",SVC(random_state=random_state,probability=True)),
	#           ("MLP",MLPClassifier(random_state=random_state))
	#           # ("KNN",KNeighborsClassifier())
	#           ]
	
	# hyperparameters = [	 
	# 					 ## [("max_depth",[15,20]),("n_estimators",[100,200])],
	# 					 [],
	# 					 [("max_depth",[5,6,10])],
	# 			  	     [("n_estimators",[10,50,100])],
	# 			  	     [("C",[1.0,0.95,0.9])],
	# 					 ## [],
	# 					 [],
	# 					 [("hidden_layer_sizes",[(200,),(150,),(100,100),(300,)])]
	# 			  	     # [("n_neighbors",[5,6,7,10])]
	# 			  	]

	best_est = None
	best_score = 0.0
	results_summary = []
	all_models = []
	for model, hyperp_setting in zip(models_clfs,hyperparameters_clfs):
		print("Fitting "+model[0])
		pipeline = Pipeline([model])
		# pipeline = Pipeline([("scaling",StandardScaler()),model])
		param_grid = {}
		for param in hyperp_setting:
			param_grid[model[0]+"__"+param[0]] = param[1]
		grid_search = GridSearchCV(pipeline,param_grid=param_grid,verbose=True,scoring="f1_weighted",cv=3, n_jobs=5)
		grid_search.fit(X_train,y_train)

		clf = grid_search.best_estimator_
		scores = []
		np.random.seed(random_state)
		for i in range(0,rep):
			rows = np.random.randint(2, size=len(X_train)).astype('bool')									
			clf.fit(X_train[rows],y_train[rows])
			preds = clf.predict(X_test)
			scores.append(f1_score(y_test,preds,average='weighted'))

		results_summary.append([model,scores])
		print(results_summary[-1])
		avg_score = pd.DataFrame(scores).mean()[0]
		if(avg_score > best_score):
			best_score = avg_score
			best_est = clf
		clf.fit(X, y)
		all_models.append([model[0],clf])
				
	y_pred = best_est.predict(X_test)
	rocs = []
	preds_score = best_est.predict_proba(X_test)
	for i in range(0,len(best_est.classes_)):
		correct_class = best_est.classes_[i]
		fpr, tpr, _ = roc_curve(y_test.as_matrix(), [p[i] for p in preds_score], pos_label=correct_class)
		roc_df = pd.DataFrame(tpr,columns=["tpr"]).join(pd.DataFrame(fpr,columns=["fpr"])).join(pd.DataFrame([correct_class]*len(tpr),columns=["class"]))
		rocs.append(roc_df)
	rocs_df = pd.concat(rocs)

	#using whole data after cv
	best_est.fit(X,y)

	return best_est, best_score, confusion_matrix(y_test,y_pred), rocs_df, results_summary, all_models


class SwitchEnsemble():
	"""Class for making predictions using the classifier best RS user-wise"""

	def __init__(self,classifier,trained_RS_folder,dataset_name,rs_list):
		"""Constructor, expects the classifier for choosing between RS and the RS predictions folder"""
		self.classifier_ = classifier
		self.trained_RS_folder_ = trained_RS_folder
		self.dataset_name_ = dataset_name
		self.rs_list_ = rs_list
		self.surprise_rs_models_ = {}

	def fit(self):
		# for rs in self.rs_list_:
		# 	file_name = os.path.expanduser(self.trained_RS_folder_+'/dump_file_'+self.dataset_name_+'_'+rs)
		# 	_, loaded_algo = dump.load(file_name)
		# 	self.surprise_rs_models_[rs] = loaded_algo

		return self

	def oracle_predict(self,bestRS,surprise_dataset):					
		# bestRS = true_y.to_frame(name="RS").reset_index()["RS"]
		# bestRS = pd.DataFrame(bestRS).join(pd.DataFrame(user_ids))
		bestRS["userId"] = bestRS["userId"].astype(str)				
		predictions = {}
		for rs in self.rs_list_:
			file_name = os.path.expanduser(self.trained_RS_folder_+'/dump_file_'+self.dataset_name_+'_'+rs)
			_, loaded_algo = dump.load(file_name)			
			predictions[rs] = loaded_algo.test(surprise_dataset)
			# predictions[rs] = self.surprise_rs_models_[rs].test(surprise_dataset)

		self.algo_predictions_ = predictions

		final_predictions = self.choose_predictions_by_clf(self.algo_predictions_,bestRS)

		return final_predictions		

	def predict(self,user_features,user_ids,surprise_dataset):				
		bestRS = self.classifier_.predict(user_features)
		bestRS = pd.DataFrame(bestRS,columns = ["RS"])
		bestRS["userId"] = user_ids
		bestRS["userId"] = bestRS["userId"].astype(str)		
		predictions = {}
		for rs in self.rs_list_:
			file_name = os.path.expanduser(self.trained_RS_folder_+'/dump_file_'+self.dataset_name_+'_'+rs)
			_, loaded_algo = dump.load(file_name)			
			predictions[rs] = loaded_algo.test(surprise_dataset)
			# predictions[rs] = self.surprise_rs_models_[rs].test(surprise_dataset)

		self.algo_predictions_ = predictions		
		final_predictions = self.choose_predictions_by_clf(predictions,bestRS)

		return final_predictions

	def choose_predictions_by_clf(self,predictions,bestRS):		
		all_predictions = []
		for rs, preds in predictions.iteritems():
			predictions_df = pd.DataFrame(preds,columns = ["userId","movieId","rating","prediction_"+rs,"details"])
			predictions_df["RS"] = rs
			all_predictions.append(predictions_df)

		all_predictions_df = pd.concat(all_predictions)

		final_predictions = all_predictions_df.merge(bestRS,on=["userId","RS"])		

		def rec_prediction_col(r):		
			return r["prediction_"+r["RS"]]		
		final_predictions["prediction_ensemble"] = final_predictions.apply(lambda r,f=rec_prediction_col: f(r),axis=1)

		return final_predictions[["userId","movieId","prediction_ensemble","rating","details"]]

def get_oracle_labels_for_test_set(dataset_name,switch_ensemble):
	reader = Reader(line_format='user item rating timestamp', sep=',')
	train = Dataset.load_from_file("./created_data/"+dataset_name+"_train.csv", reader=reader)
	test_ensembles = Dataset.load_from_file("./created_data/"+dataset_name+"_test_ensembles.csv", reader=reader)
		
	uf = UserFeatures(pd.DataFrame(train.raw_ratings,columns = ["userId","movieId","rating","timestamp"]),False)
	all_features_df = uf.get_all_user_features()

	recs_avg_errors = []		
	for rs in RS:
		#Memory error for 16GB machine or float division error for lastfm
		if("KNN" in rs["name"] and dataset_name in datasets_knn_mem_error):
			continue
		file_name = os.path.expanduser('./created_data/trained_RS/dump_file_'+dataset_name+'_'+rs["name"])
		_, loaded_algo = dump.load(file_name)

		predictions = loaded_algo.test(test_ensembles.build_full_trainset().build_testset())
		predictions_df = pd.DataFrame(predictions,columns = ["userId","movieId","rating","prediction","details"])

		predictions_with_relevance = remove_dataset_bias(predictions_df, has_ns = True)		
		scores = predictions_with_relevance.groupby("userId").agg(lambda r,f = calculate_ndcg_score: f(r,"prediction"))
		scores = scores[[scores.columns[0]]].rename(index=str,columns={scores.columns[0]:"NDCG"}).reset_index()
		scores["RS"] = rs["name"]
		# this was used when mae was the criterea for creating the H1 dataset
		# predictions_df["error"] = abs(predictions_df["prediction"]-predictions_df["rating"])
		# avg_errors = predictions_df.groupby("userId")["error"].mean().rename("avg_error").to_frame().reset_index()
		# avg_errors["RS"] = rs["name"]

		recs_avg_errors.append(scores)

	all_avg_errors = pd.concat(recs_avg_errors).reset_index()	
	assert all_avg_errors.isnull().values.any() == False	

	Xy = create_best_RS_userwise_dataset(all_avg_errors,all_features_df)
	if("amazon" not in dataset_name):
		Xy["userId"] = Xy["userId"].astype(int)
	return Xy.sort_values("userId")[["userId","label"]]

def main():
	parser = optparse.OptionParser()
	parser.add_option('-d', '--datasets', 
						dest="datasets")

	options, remainder = parser.parse_args()	

	datasets = options.datasets.split(",")

	print(datasets)

	# run_h1_on_test_set = False
	run_h1_on_test_set = True

	reader = Reader(line_format='user item rating timestamp', sep=',')

	experiment_combination = ["error-features-val","meta-features","error-features","all"]	

	for dataset_name in datasets:			
		
		print("\n Dataset "+ dataset_name)

		classify_task_df = pd.read_csv("./created_data/hypothesis_data/H1_"+dataset_name+".csv")		
		
		for features in experiment_combination:				
			
			print("Feature combination: " +features)

			feature_cols = []
			if features == "all":
				feature_cols = [c for c in classify_task_df.columns if c!= "label"]
			elif features == "meta-features":
				feature_cols = [c for c in classify_task_df.columns if c!= "label" and "MAE" not in c and "RMSE" not in c and "MSE" not in c and "RR" not in c and "NDCG" not in c and "Precision" not in c and  "AP" not in c]
			elif features == "error-features":
				feature_cols = [c for c in classify_task_df.columns if c!= "label" and ("MAE" in c or "NDCG" in c or "Precision" in c or "AP" in c or "RMSE" in c or "MSE" in c or "RR" in c) and "val_set" not in c]
			elif features == "error-features-val":
				feature_cols = [c for c in classify_task_df.columns if c!= "label" and ("MAE" in c or "NDCG" in c or "Precision" in c or "AP" in c or "RMSE" in c or "MSE" in c or "RR" in c) and "val_set" in c]
			# print(feature_cols)

			X = classify_task_df[feature_cols]
			y = classify_task_df["label"]			

			best_clf,best_score,cm,rocs_df,results_summary, all_models = model_selection(X.as_matrix(),y,1)
			print("Best score : "+str(best_score))
			print("Classifier : "+str(best_clf))
			print("Confusion matrix:")
			print(cm)

			fixed_clf = RandomForestClassifier(random_state=42,n_estimators = 100)
			fixed_clf.fit(X,y)



			# rocs_df.to_csv("./created_data/tmp/SCB_"+features+"_"+dataset_name+"_ROCs.csv",index=False)
			# pd.DataFrame(cm,columns= best_clf.classes_).join(pd.DataFrame(best_clf.classes_)).to_csv("./created_data/tmp/SCB_"+features+"_"+dataset_name+"_CM.csv",index=False)
			# results_df = []
			# for row in [(r[0][0],r[1]) for r in results_summary]:
			# 	for result in row[1]:
			# 		results_df.append([row[0],result])
			# results_df = pd.DataFrame(results_df,columns = ["Classifier","f1"])
			# results_df.to_csv("./created_data/tmp/SCB_"+features+"_"+dataset_name+"_results.csv",index=False)

			# """

			if(run_h1_on_test_set):

				# if(not os.path.exists("./created_data/tmp/h1_"+dataset_name+"_user_features_df.csv")):
				train = Dataset.load_from_file("./created_data/"+dataset_name+"_train.csv", reader=reader)
				uf = UserFeatures(pd.DataFrame(train.raw_ratings,columns = ["userId","movieId","rating","timestamp"]),False)
				all_features_df = uf.get_all_user_features()
				all_features_df.to_csv("./created_data/tmp/h1_"+dataset_name+"_user_features_df.csv",index=False,header=True)

				test_ensemble_df = pd.read_csv("./created_data/"+dataset_name+"_test_ensembles.csv",names = ["userId","movieId","prediction","timestamp","is_negative_sample"])

				all_features_df = pd.read_csv("./created_data/tmp/h1_"+dataset_name+"_user_features_df.csv")
				test_user_features = all_features_df.merge(test_ensemble_df,on=["userId"],how="inner")[[c for c in all_features_df.columns if c not in ["Index"]]]
				test_user_features = test_user_features.groupby("userId").first().reset_index()
				user_ids = test_user_features["userId"]

				# Adding train error as features for H1
				user_train_time_features = pd.read_csv("./created_data/tmp/h2_"+dataset_name+"_user_train_time_features.csv")
				# user_train_time_features["userId"] = user_train_time_features["userId"].astype(str)
				
				user_train_time_val_features = pd.read_csv("./created_data/tmp/h2_"+dataset_name+"_user_train_time_features_val_set.csv")

				test_user_features = (test_user_features.merge(user_train_time_features,on="userId")).merge(user_train_time_val_features,on="userId",how="left")				
				test_user_features = test_user_features.fillna(0.0)

				test_user_features = test_user_features[[c for c in test_user_features.columns if c != "userId" and c in feature_cols]]

				surprise_test_set = Dataset.load_from_file("./created_data/"+dataset_name+"_test_ensembles.csv", reader=reader).build_full_trainset().build_testset()

				rs_list = [rs["name"] for rs in RS]
				if(dataset_name in datasets_knn_mem_error):
					rs_list = [rs["name"] for rs in RS if "KNN" not in rs["name"]]					

				switch_ensemble = SwitchEnsemble(best_clf,"./created_data/trained_RS",dataset_name,rs_list)
				# """
				print("fitting ensemble (loading rs trained models)")
				switch_ensemble.fit()
				predictions = switch_ensemble.predict(test_user_features.as_matrix(),user_ids,surprise_test_set)
				predictions.to_csv("./created_data/tmp/predictions_H1"+dataset_name+"_"+features+".csv",header=True,index=False)		
				
				print("fitting ensemble (loading rs trained models) for fixed model")
				switch_ensemble = SwitchEnsemble(fixed_clf,"./created_data/trained_RS",dataset_name,rs_list)
				switch_ensemble.fit()
				predictions = switch_ensemble.predict(test_user_features.as_matrix(),user_ids,surprise_test_set)
				predictions.to_csv("./created_data/tmp/predictions_H1"+dataset_name+"_"+features+"_fixed.csv",header=True,index=False)		
				
				print("fitting ensemble for all models for robustness analysis")
				for (model_name, clf) in all_models:
					switch_ensemble = SwitchEnsemble(clf, "./created_data/trained_RS", dataset_name, rs_list)
					print("		"+ model_name)
					switch_ensemble.fit()
					predictions = switch_ensemble.predict(test_user_features.as_matrix(),user_ids,surprise_test_set)
					predictions.to_csv("./created_data/tmp/predictions_H1"+dataset_name+"_"+features+"_"+model_name+"_robustness.csv",header=True,index=False)

				# """ 
				if(features == "all"):
					best_rs = get_oracle_labels_for_test_set(dataset_name,switch_ensemble)									
					best_rs["RS"] = best_rs["label"]					
					best_rs.to_csv("./created_data/tmp/predictions_H1"+dataset_name+"_oracle_labels.csv",header=True,index=False)
					oracle_predictions = switch_ensemble.oracle_predict(best_rs,surprise_test_set)										
					oracle_predictions.to_csv("./created_data/tmp/predictions_H1"+dataset_name+"_oracle.csv",header=True,index=False)

if __name__ == "__main__":
	main()