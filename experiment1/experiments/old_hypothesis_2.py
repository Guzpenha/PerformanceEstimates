from IPython import embed

import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV,train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def create_stacking_dataset(predictions_dict, user_features_dict = {}):	
	
	X = []
	y = []

	for k in predictions_dict.keys():
		user,movie = k
		features = []
		for rec in predictions_dict[k].keys():
			if(rec == "label"):
				y.append(predictions_dict[k][rec])
			else:					
				features.append(predictions_dict[k][rec])
		if(len(user_features_dict.keys())>0):
			features = features + user_features_dict[user]
		X.append(features)

	return X,y

def load_recommender_predictions(RS,sample_size = 1000000):
	predictions = {}

	for rec in RS:
		pred_df = pd.read_csv("../../data/created_17/predictions_"+rec+"_with_errors.csv")

		print("Rec: "+rec)
		for index, row in pred_df[0:sample_size].iterrows():			
			if (row["userId"],row["movieId"]) not in predictions:
				predictions[(row["userId"],row["movieId"])] = {}
				predictions[(row["userId"],row["movieId"])]["label"] = row["rating"]
			predictions[(row["userId"],row["movieId"])][rec] = row["prediction_"+rec]
	return predictions

def create_user_features_dict(df):
	user_dict = {}
	for index, row in df.iterrows():
		user_dict[row["userId"]] = [row[c] for c in df.columns if c != "userId"]
	return user_dict

def model_selection(X,y, rep=10):
	""" Uses grid search and cross validation to choose the best clf for the task (X,y)"""

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	models = [#("XGBRegressor",XGBRegressor(max_depth=5)),					
			  ("MLPRegressor",MLPRegressor(hidden_layer_sizes=(200,))),
			  # ("AdaBoostRegressor",AdaBoostRegressor(n_estimators = 100)),
			  ("SGDRegressor",SGDRegressor()),
			  ("GBoostingRegressor",GradientBoostingRegressor()),
			  ("RF",RandomForestRegressor(max_depth=10,n_estimators = 100)),
			  ("LinearReg",LinearRegression()),]
	hyperparameters = [[] for m in models]
	

	best_est = None
	best_score = 0.0
	results_summary = []
	for model, hyperp_setting in zip(models,hyperparameters):
		print("Fitting "+model[0])
		pipeline = Pipeline([("StdScaling",StandardScaler()),model])
		# pipeline = Pipeline([("scaling",StandardScaler()),model])
		param_grid = {}
		for param in hyperp_setting:
			param_grid[model[0]+"__"+param[0]] = param[1]
		grid_search = GridSearchCV(pipeline,param_grid=param_grid,verbose=True,scoring="neg_mean_absolute_error",cv=2)
		grid_search.fit(X_train,y_train)

		clf = grid_search.best_estimator_
		scores = []
		for i in range(0,rep):
			rows = np.random.randint(2, size=len(X_train)).astype('bool')									
			clf.fit(np.array(X_train)[rows],np.array(y_train)[rows])
			preds = clf.predict(X_test)
			scores.append(mean_absolute_error(y_test,preds))

		results_summary.append([model,scores])
		print(results_summary[-1])
		avg_score = pd.DataFrame(scores).mean()[0]
		if(avg_score > best_score):
			best_score = avg_score
			best_est = clf

	return best_est, best_score, results_summary


def main():
	RS = ["BiPolarSlopeOne","CoClustering", \
		"FactorWiseMatrixFactorization", \
		"LatentFeatureLogLinearModel","SigmoidSVDPlusPlus",\
		"SlopeOne"]

	print("Creating dataset")
	user_features_df = pd.read_csv("../../data/created/user_features.csv")
	user_features_dict = create_user_features_dict(user_features_df)	

	predictions = load_recommender_predictions(RS)
	user_item_tuples = predictions.keys()
	X,y = create_stacking_dataset(predictions,user_features_dict)

	pd.DataFrame(X).to_csv("../../data/created/hypothesis2_X.csv",index=False)	
	pd.DataFrame(y).to_csv("../../data/created/hypothesis2_y.csv",index=False)
	pd.DataFrame(user_item_tuples,columns = ["userId","movieId"]).to_csv("../../data/created/hypothesis2_useritems.csv",index=False)

	# print("Creating dataset with no user features")
	# X_no_features,y_no_features = create_stacking_dataset(predictions)	
	# pd.DataFrame(X_no_features).to_csv("../../data/created/hypothesis2_X_no_features.csv",index=False)	
	# pd.DataFrame(y_no_features).to_csv("../../data/created/hypothesis2_y_no_features.csv",index=False)

	embed()

	X = pd.read_csv("../../data/created/hypothesis2_X.csv")
	y = pd.read_csv("../../data/created/hypothesis2_y.csv")
	user_item_tuples = pd.read_csv("../../data/created/hypothesis2_useritems.csv")

	print("Finding best model")
	best_regressor,mse, results_summary = model_selection(X,y)
	
	print("Writing predictions to file")
	best_regressor.fit(X,y)
	user_item_tuples["predictions"] = best_regressor.predict(X)
	user_item_tuples.to_csv("../../data/created/weighted_predictions.csv",index=False)

	results_df = []
	for row in [(r[0][0],r[1]) for r in results_summary]:
		for result in row[1]:
			results_df.append([row[0],result])
	results_df = pd.DataFrame(results_df,columns = ["Model","mse"])
	results_df.to_csv("../../data/created/weighted_approach_results.csv",index=False)

	# print("Finding best model with no user features")
	# X_no_features = pd.read_csv("../../data/created/hypothesis2_X_no_features.csv")
	# y_no_features = pd.read_csv("../../data/created/hypothesis2_y_no_features.csv")

	# best_regressor,mse, results_summary = model_selection(X,y)
	
	# print("Writing predictions to file")
	# best_regressor.fit(X,y)
	# user_item_tuples["predictions"] = best_regressor.predict(X)
	# user_item_tuples.to_csv("../../data/created/weighted_predictions_no_features.csv",index=False)

	# results_df = []
	# for row in [(r[0][0],r[1]) for r in results_summary]:
	# 	for result in row[1]:
	# 		results_df.append([row[0],result])
	# results_df = pd.DataFrame(results_df,columns = ["Model","mse"])
	# results_df.to_csv("../../data/created/weighted_no_features_approach_results.csv",index=False)

	embed()

if __name__ == '__main__':
	main()
