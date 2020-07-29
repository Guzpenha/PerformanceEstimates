from IPython import embed

import pandas as pd
import numpy as np

from xgboost import XGBClassifier
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

def create_best_RS_userwise_dataset(errors_df,user_features_df,RS):
	"""
	Creates dataset that for a classify task, where X are user features and y is 
	the best RS on average.

	Inputs
	---------
		errors_df: pandas DataFrame containg average errors (avg_error) and user ids (userId) columns
		user_features_df: pandas DataFrame containing user features as columns and user ids (userId)		
		RS: list str containing RS names to keep in dataset

	Returns
	--------
		Xy: pandas DataFrame containing user ids (userID), features and labels (label).

	"""	
	errors_df = errors_df[errors_df["RS"].isin(RS)]
	min_avg_errors = errors_df.loc[errors_df.groupby("userId")["avg_error"].idxmin()]
	df = min_avg_errors.merge(user_features_df,on="userId")

	Xy = df[[c for c in df.columns if c != "avg_error"]]
	Xy["label"]= Xy["RS"]
	Xy = Xy[[c for c in Xy.columns if c != "RS"]]

	return Xy

def model_selection(X,y, rep=10):
	""" Uses grid search and cross validation to choose the best clf for the task (X,y)"""

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	models = [("XGB",XGBClassifier(objective = "multi:softmax")),
			  ("NaiveBayes",GaussianNB()),
			  ("RF",RandomForestClassifier()),
	          ("AdaBoost",AdaBoostClassifier()),
	          ("LR",LogisticRegression()),
	          # ("linSVC",LinearSVC(multi_class="ovr")),
	          # ("SVC",SVC()),
	          ("MLP",MLPClassifier()),
	          ("KNN",KNeighborsClassifier())]
	# hyperparameters = [[] for m in models]	
	
	hyperparameters = [	 [("max_depth",[10]),("n_estimators",[100,200])],
						 [],
						 [("max_depth",[5,6,10])],
				  	     [("n_estimators",[10,50,100])],
				  	     [("C",[1.0,0.95,0.9])],
						 [("hidden_layer_sizes",[(200,),(150,),(100,100),(300,)])],
				  	     [("n_neighbors",[5,6,7,10])]]

	best_est = None
	best_score = 0.0
	results_summary = []
	for model, hyperp_setting in zip(models,hyperparameters):
		print("Fitting "+model[0])
		pipeline = Pipeline([model])
		# pipeline = Pipeline([("scaling",StandardScaler()),model])
		param_grid = {}
		for param in hyperp_setting:
			param_grid[model[0]+"__"+param[0]] = param[1]
		grid_search = GridSearchCV(pipeline,param_grid=param_grid,verbose=True,scoring="f1_weighted",cv=5)
		grid_search.fit(X_train,y_train)

		clf = grid_search.best_estimator_
		scores = []
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

	y_pred = best_est.predict(X_test)
	rocs = []
	preds_score = best_est.predict_proba(X_test)
	for i in range(0,len(best_est.classes_)):
		correct_class = best_est.classes_[i]
		fpr, tpr, _ = roc_curve(y_test.as_matrix(), [p[i] for p in preds_score], pos_label=correct_class)
		roc_df = pd.DataFrame(tpr,columns=["tpr"]).join(pd.DataFrame(fpr,columns=["fpr"])).join(pd.DataFrame([correct_class]*len(tpr),columns=["class"]))
		rocs.append(roc_df)
	rocs_df = pd.concat(rocs)

	return best_est, best_score, confusion_matrix(y_test,y_pred),rocs_df,results_summary


class SwitchEnsemble():
	"""Class for making predictions using the classifier best RS user-wise"""

	def __init__(self,classifier,recSysPredictionsFolder):
		"""Constructor, expects the classifier for choosing between RS and the RS predictions folder"""
		self.classifier_ = classifier
		self.predictionsFolder_ = recSysPredictionsFolder		
		self.RSPredictionsLoaded_ = None
		self.predictionsLoaded_ = None		

	def fit(self):		
		return self

	def predict(self,userId,user_features):
		"""
		Predicts all movies for userId in self.predictionsFolder_ considering the best RS for
		such user_features

		Inputs
		---------
		userId: int containing the id of the user to get predictions.
		user_features: list of int containing user features, same dimensionality 
					   as classifier model was trained.
	
		
		Return
		--------
		predictions: pandas DataFrame with two colums: itemId and prediction, containing
					 predictions for all movies for this user using the best RS predictions
					 in self.predictionsFold

		"""		
		bestRS = self.classifier_.predict(user_features)[0]

		return self.getRSPredFromFile(userId,bestRS)

	def getRSPredFromFile(self,userId,bestRS):
		if(self.RSPredictionsLoaded_ != bestRS):
			self.predictionsLoaded_ = pd.read_csv(self.predictionsFolder_+"predictions_"+bestRS+".csv",names = ["userId","movieId","prediction"])
			self.RSPredictionsLoaded_ = bestRS

		pred = self.predictionsLoaded_[(self.predictionsLoaded_["userId"]==userId)][["movieId","prediction"]]

		return pred

def main():
	# Loads datasets
	user_avg_errors = pd.read_csv("../../data/created/user_avg_errors.csv")	
	user_features_df = pd.read_csv("../../data/created/user_features.csv")	
	
	# Prepare classify task dataset
	#"BiasedMatrixFactorization"
	# "MatrixFactorization"
	RS = ["BiPolarSlopeOne","CoClustering", \
	"FactorWiseMatrixFactorization","GlobalAverage","ItemAverage", \
	"LatentFeatureLogLinearModel","SigmoidSVDPlusPlus",\
	"SlopeOne","TimeAwareBaseline","TimeAwareBaselineWithFrequencies","UserAverage","UserItemBaseline"]

	classify_task_df = create_best_RS_userwise_dataset(user_avg_errors,user_features_df,RS)
	classify_task_df.to_csv("../../data/created/hypothesis1_df.csv",index=False)
	classify_task_df = pd.read_csv("../../data/created/hypothesis1_df.csv")	

	X = classify_task_df[[c for c in classify_task_df.columns if c != "userId" and c != "label"]]
	y = classify_task_df["label"]

	# Finds best classifier using cross validation and grid_search on hyperparameters
	# best_clf,best_score,cm,rocs_df,results_summary = model_selection(X.as_matrix(),y)
	# print("Best score : "+str(best_score))
	# print("Classifier : "+str(best_clf))
	# print("Confusion matrix:")
	# print(cm)
	# pd.DataFrame(cm,columns= best_clf.classes_).join(pd.DataFrame(best_clf.classes_)).to_csv("../../data/created/switching_approach_cm.csv",index=False)
	# rocs_df.to_csv("../../data/created/switching_approach_rocs.csv",index=False)	
	# results_df = []
	# for row in [(r[0][0],r[1]) for r in results_summary]:
	# 	for result in row[1]:
	# 		results_df.append([row[0],result])
	# results_df = pd.DataFrame(results_df,columns = ["Classifier","f1"])
	# results_df.to_csv("../../data/created/switching_approach_results.csv",index=False)

	best_clf = XGBClassifier(max_depth=10,n_estimators=200)	
	best_clf.fit(X.as_matrix(),y)

	# Makes predictions using switching approach
	switchingRS_predictions = []
	ml20m_test = pd.read_csv("../../data/ml20m_test.csv",names = ["userId","movieId","prediction","timestamp"])
	switchingRS = SwitchEnsemble(classifier=best_clf,recSysPredictionsFolder="../../data/created_17/")		

	#make sure we ask for users with the same RS after each other, to reduce loading from files
	predictions = pd.DataFrame(best_clf.predict(X.as_matrix()),columns = ["prediction"])
	users_test = ml20m_test["userId"].drop_duplicates().reset_index()
	users_with_pred = users_test.join(predictions)
	users = users_with_pred.sort_values("prediction")["userId"].as_matrix()
	#makes predictions for each user
	i=0
	for user in users:
		if(i%100==0):
			print("User "+str(i))
		pred = switchingRS.predict(user, user_features_df[user_features_df["userId"]==user] \
			[[c for c in user_features_df.columns if c !="userId"]].as_matrix())
		
		pred["userId"] = user
		switchingRS_predictions.append(pred)
		i+=1
		
	predictionsDF = pd.concat(switchingRS_predictions)
	predictionsDF.to_csv("../../data/created/switching_predictions.csv",index=False)

	#makes oracle predictions for each user
	predictions = pd.DataFrame(y)
	predictions["prediction"] = predictions["label"]
	users_test = ml20m_test["userId"].drop_duplicates().reset_index()
	users_with_pred = users_test.join(predictions)
	users = users_with_pred.sort_values("prediction")	
	i=0
	for index, row in users.iterrows():
		user = row["userId"]
		bestRS = row["prediction"]
		if(i%100==0):
			print("User "+str(i))
								
		pred = switchingRS.getRSPredFromFile(user,bestRS)

		pred["userId"] = user
		switchingRS_predictions.append(pred)
		i+=1

	predictionsDF = pd.concat(switchingRS_predictions)
	predictionsDF.to_csv("../../data/created/switching_predictions_oracle.csv",index=False)

if __name__ == "__main__":
	main()