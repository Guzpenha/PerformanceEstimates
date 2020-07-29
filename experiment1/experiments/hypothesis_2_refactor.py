from IPython import embed

import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.kernel_ridge import KernelRidge

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.model_selection import GridSearchCV,train_test_split

from sklearn.preprocessing import StandardScaler,MinMaxScaler,PolynomialFeatures
from sklearn.feature_selection import SelectKBest,chi2,VarianceThreshold
from sklearn.pipeline import Pipeline

def model_selection(X,y, rep=10):
	""" Uses grid search and cross validation to choose the best clf for the task (X,y)"""

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	models = [
	  		  # ("KNeighborsRegressor",KNeighborsRegressor(n_neighbors=3)),
			  # ("ExtraTreesRegressor",ExtraTreesRegressor(criterion='mae')),
			  # ("GBoostingRegressor",GradientBoostingRegressor()),
			  # ("RF",RandomForestRegressor()),
	  		  # ("SVR",SVR()),
	  		  # ("LinearSVR",LinearSVR()),
	  		  # ("NuSVR",NuSVR()),			  
			  # ("SGDRegressor",SGDRegressor()),
			  ("MLP",MLPRegressor(hidden_layer_sizes=(30,20,),random_state=2))#,
			  # ("LinearReg",LinearRegression()),
			  ]

	# hyperparameters = [[] for m in models]

	hyperparameters = [ #[],
					    # [("hidden_layer_sizes",[(30,),(40,),(50,),(30,10,)])],
					    #[("n_estimators",[25,50,100])],
						# [],
						[]						
						# [("alpha",[0.0001,0.0002,0.0005])],						
						# [("activation",['logistic', 'tanh', 'relu'])]
						]	

	best_est = None
	best_score = 100
	results_summary = []
	for model, hyperp_setting in zip(models,hyperparameters):
		print("Fitting "+model[0])
		# pipeline = Pipeline([("StdScaling",StandardScaler()),("PolynomialFeatures",PolynomialFeatures(interaction_only=True)),model])		
		pipeline = Pipeline([("StdScaling",StandardScaler()),model])		
		param_grid = {}
		for param in hyperp_setting:
			param_grid[model[0]+"__"+param[0]] = param[1]
		grid_search = GridSearchCV(pipeline,param_grid=param_grid,verbose=True,scoring="neg_mean_absolute_error",cv=5)
		grid_search.fit(X_train,y_train)

		clf = grid_search.best_estimator_
		scores = []
		for i in range(0,rep):
			rows = np.random.randint(2, size=len(X_train)).astype('bool')									
			clf.fit(np.array(X_train)[rows],np.array(y_train)[rows])
			preds = clf.predict(X_test)
			scores.append(mean_absolute_error(y_test,preds))

		results_summary.append([model,scores])		
		avg_score = pd.DataFrame(scores).mean()[0]
		if(avg_score < best_score):
			best_score = avg_score
			best_est = clf

	return best_est, best_score, results_summary

def main():
	# RS = ["BiPolarSlopeOne","CoClustering", \
	# 	"FactorWiseMatrixFactorization", \
	# 	"LatentFeatureLogLinearModel","SigmoidSVDPlusPlus",\
	# 	"SlopeOne"]

	RS = ["BiPolarSlopeOne", \
		"FactorWiseMatrixFactorization", \
		"SigmoidSVDPlusPlus",
		"BiasedMatrixFactorization",
		"MatrixFactorization"]

	user_features_df = pd.read_csv("../../data/created/user_features.csv")
	
	# print("Creating dataset")
	# Xy = pd.read_csv("../../data/created_17/predictions_"+RS[0]+"_with_errors.csv")[["userId","movieId","prediction_"+RS[0],"rating"]]
	# for rec in RS[1:]:		
	# 	Xy = Xy.merge(pd.read_csv("../../data/created_17/predictions_"+rec+"_with_errors.csv") \
	# 		[["userId","movieId","prediction_"+rec]],on=["userId","movieId"])	
	# Xy = Xy.merge(user_features_df,on=["userId"])
	
	# Xy.to_csv("../../data/created/Xy_second_hypothesis.csv",index=False)
	Xy = pd.read_csv("../../data/created/Xy_second_hypothesis.csv")

	X_cols = ["prediction_"+rec for rec in RS]
	y_col = ["rating"]
	user_cols = [c for c in user_features_df.columns if c not in ["userId","moviesPopularity","support"] and "SVD" not in c]
		
	X = Xy[X_cols+ user_cols].as_matrix()
	y = Xy[y_col].as_matrix()

	sample_division_size = 10
	reps = 1

	print("Finding best model")
	# best_regressor = Pipeline([("StdScaling",StandardScaler()),("FeatureSel",SelectKBest(chi2, k=5)),("MLP",MLPRegressor(hidden_layer_sizes=(30,10,),random_state=3))])
	best_regressor = Pipeline([("StdScaling",StandardScaler()), \
		("MLP",MLPRegressor(hidden_layer_sizes=(30,10,),random_state=3,activation='tanh'))])		
	# best_regressor,mae, results_summary = model_selection(X[0:len(X)/sample_division_size],y[0:len(X)/sample_division_size],rep=reps)
	# print("Best model: ")
	# print(best_regressor)
	# print(pd.DataFrame(results_summary[0][1]).mean())	


	print("Fitting model")
	best_regressor.fit(X[0:len(X)/sample_division_size],y[0:len(X)/sample_division_size])
	preds = best_regressor.predict(X)
	print(mean_absolute_error(preds,y))
	print(np.sqrt(mean_squared_error(preds,y)))

	print("Best Single model MAE  : 0.6055113")
	print("Best Single model RMSE : 0.7937833")
	# 0.6030308	for 30-10 mlp 3 RS and no constants	and no moviesPopularity or support and 10% of data and activation = tanh
  	# 0.5957313   for 30-10 mlp 5 rs = 
	# 0.6024157	for switching

	# print("Writing predictions to file")
	# Xy["predictions"] = best_regressor.predict(X)
	# Xy[["userId","movieId","predictions"]].to_csv("../../data/created/weighted_predictions.csv",index=False)

	# results_df = []
	# for row in [(r[0][0],r[1]) for r in results_summary]:
	# 	for result in row[1]:
	# 		results_df.append([row[0],result])
	# results_df = pd.DataFrame(results_df,columns = ["Model","mae"])
	# results_df.to_csv("../../data/created/weighted_approach_results.csv",index=False)


	###############################################
	# no user_features model
	###############################################

	# X = Xy[X_cols+ constants].as_matrix()
	# y = Xy[y_col].as_matrix()

	# print("Finding best model no user_features")
	# best_regressor, mae, results_summary = model_selection(X[0:len(X)/sample_division_size],y[0:len(X)/sample_division_size],rep=reps)
	# print("Best model: ")
	# print(best_regressor)

	# print("Writing predictions to file")
	# best_regressor.fit(X[0:len(X)/sample_division_size],y[0:len(X)/sample_division_size])
	# Xy["predictions"] = best_regressor.predict(X)
	# Xy[["userId","movieId","predictions"]].to_csv("../../data/created/weighted_no_features_predictions.csv",index=False)

	# results_df = []
	# for row in [(r[0][0],r[1]) for r in results_summary]:
	# 	for result in row[1]:
	# 		results_df.append([row[0],result])
	# results_df = pd.DataFrame(results_df,columns = ["Model","mae"])
	# results_df.to_csv("../../data/created/weighted_no_features_approach_results.csv",index=False)

if __name__ == '__main__':
	main()
