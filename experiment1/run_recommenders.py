from IPython import embed

import pandas as pd
import os

from functools import reduce

def run_methods(recommenders, train_file = "../data/ml20m_train.csv",test_file = "../data/ml20m_test.csv"):
	
	"""
	Runs recommenders methods on train and test files and returns predictions and errors for them.

	Inputs
	---------
		recommenders: list of str containing recommender names from MyMediaLite
		train_file: str indicating train file to be used (user,movie,rating,timestamp) format expected
		test_file: str indicating train file to be used (user,movie,rating,timestamp) format expected

	Returns
	--------
		nothing, it writes results to "../data/created/"

	"""	

	#run recommenders writting results to files
	for rec in recommenders:
		predictions_file = "../data/created/predictions_"+rec+".csv"		
		if(rec=="Constant5"):	
			os.system("~/MyMediaLite-3.11/bin/rating_prediction --training-file="+train_file+" --test-file="+test_file+" --prediction-line=\"{0},{1},{2}\" --prediction-file="+predictions_file+ " --recommender="+rec[0:-1] + " --recommender-options=\"constant_rating=5\"")
		else:
			os.system("~/MyMediaLite-3.11/bin/rating_prediction --training-file="+train_file+" --test-file="+test_file+" --prediction-line=\"{0},{1},{2}\" --prediction-file="+predictions_file+ " --recommender="+rec)	


def append_errors(recommenders, test_file = "../data/ml20m_test.csv"):
	"""
	Joins predictions with true ratings, calculating error on each prediction

	Inputs
	---------
		recommenders: list of str containing recommender names from MyMediaLite
		train_file: str indicating train file to be used (user,movie,rating,timestamp) format expected
		test_file: str indicating train file to be used (user,movie,rating,timestamp) format expected

	Returns
	--------
		nothing, it writes results to "../data/created/"

	"""	

	test_data = pd.read_csv(test_file,names = ["userId","movieId","rating","timestamp"])
	for rec in recommenders:
		print("Appending true ratings and errors to recommender "+ rec)
		predictions = pd.read_csv("../data/created/predictions_"+rec+".csv", names = ["userId","movieId","prediction_"+rec])
		predictions_with_errors = predictions.merge(test_data,on=["userId","movieId"])
		predictions_with_errors["error_"+rec] = abs(predictions_with_errors["rating"] - predictions_with_errors["prediction_"+rec])
		predictions_with_errors.to_csv("../data/created/predictions_"+rec+"_with_errors.csv",header=True,sep = ",", index=False)

def aggregate_errors_by_user(recommenders):
	"""
	Uses files created by append_errors to get user average errors of each RSs 

	Inputs
	---------
		recommenders: list of str containing recommender names from MyMediaLite

	Returns
	--------
		all user erros in a dataframe

	"""	
	merged = []
	for rec in recommenders:
		predictions_with_errors = pd.read_csv("../data/created/predictions_"+rec+"_with_errors.csv")
		user_errors = predictions_with_errors.groupby("userId")["error_"+rec].mean().rename("avg_error").to_frame()
		user_errors["RS"] = rec		
		merged.append(user_errors)

	return pd.concat(merged)

def main():
	recommenders = ["BiasedMatrixFactorization","BiPolarSlopeOne","CoClustering", \
	"Constant","Constant5","FactorWiseMatrixFactorization","GlobalAverage","ItemAverage", \
	"LatentFeatureLogLinearModel","MatrixFactorization","Random","SigmoidSVDPlusPlus",\
	"SlopeOne","TimeAwareBaseline","TimeAwareBaselineWithFrequencies","UserAverage","UserItemBaseline"]

	run_methods(recommenders)
	append_errors(recommenders)	
	aggregate_errors_by_user(recommenders).to_csv("../data/created/user_avg_errors.csv",header=True,sep = ",")

if __name__ == "__main__":
	main()