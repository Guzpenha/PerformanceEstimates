from IPython import embed

import pandas as pd
from random import shuffle


def split_dataset_by_user(ratings,train_percentage=0.6):
	"""
	This function splits a rating matrix user-wise into train and test by user

	Inputs
	--------
		ratings: pandas DataFrame, with following columns: ["userId,movieId,rating,timestamp"]
		train_percentage: float64 indicating the percentage of data used for train data
	
	Returns
	-------
		(train_data,test_data): pandas DataFrames tuple containing train and test data.

	"""

	#groupby user select list
	ratings_by_users = ratings.groupby("userId").apply(lambda group: list(zip(group['rating'],group['movieId'],group["timestamp"]))).to_frame("mv_rat_list")

	#shuffles list
	def shuffled(x):
		import random
		y = x[:]
		random.shuffle(y)
		return y

	ratings_by_users["mv_rat_list_shuffled"] =  ratings_by_users["mv_rat_list"].apply(shuffled)

	#divide lists by 60 and 40 %
	train_percentage = 0.6
	ratings_by_users["train_by_user"] = ratings_by_users["mv_rat_list_shuffled"].apply(lambda r: r[0:int(len(r)*0.6)])
	ratings_by_users["test_by_user"] = ratings_by_users["mv_rat_list_shuffled"].apply(lambda r: r[int(len(r)*0.6)+1:])

	#adds lists sizes as columns
	ratings_by_users["train_size"] = ratings_by_users["train_by_user"].apply(lambda r: len(r))
	ratings_by_users["test_size"] = ratings_by_users["test_by_user"].apply(lambda r: len(r))
	
	#extract ratings from lists
	train_data = []
	test_data = []

	ratings_by_users = ratings_by_users.reset_index()

	for index, row in ratings_by_users.iterrows():
		for rating in row.train_by_user:
			train_data.append([row.userId,rating[1],rating[0],rating[2]])
		for rating in row.test_by_user:
			test_data.append([row.userId,rating[1],rating[0],rating[2]])

	train_data_df = pd.DataFrame(train_data,columns=["userId","movieId","rating","timestamp"])
	test_data_df = pd.DataFrame(test_data,columns=["userId","movieId","rating","timestamp"])

	return train_data_df,test_data_df

def main():
	ratings = pd.read_csv("../data/ml-20m/ratings.csv",sep=",")	
	train,test = split_dataset_by_user(ratings)
	
	train.to_csv("../data/ml20m_train.csv",sep=",",index=False,header=False)
	test.to_csv("../data/ml20m_test.csv",sep=",",index=False,header=False)
	

if __name__ == "__main__":
	main()