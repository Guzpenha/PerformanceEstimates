from IPython import embed
from datetime import timedelta

import pandas as pd
import numpy as np
import random

from config import *

from datetime import datetime
import warnings

import optparse

warnings.filterwarnings('ignore')

def load_ml_100k(path):
	"""Loads ml100k from path to a pandas DataFrame format"""
	ratings = pd.read_csv(path,sep="\t", names = ["userId","movieId","rating","timestamp"])
	ratings.loc[:,"timestamp"] = pd.to_datetime(ratings["timestamp"],unit='s')
	return ratings

def load_ml_1m(path):
	"""Loads ml1m from path to a pandas DataFrame format"""
	ratings = pd.read_csv(path,sep="::", engine='python', names = ["userId","movieId","rating","timestamp"])
	ratings.loc[:,"timestamp"] = pd.to_datetime(ratings["timestamp"],unit='s')
	return ratings

def load_ml_20m(path):
	"""Loads ml20m from path to a pandas DataFrame format"""
	ratings = pd.read_csv(path ,sep=",")		
	ratings.loc[:,"timestamp"] = pd.to_datetime(ratings["timestamp"],unit='s')
	return ratings

def load_netflix(path):
	ratings = pd.read_csv(path, sep=",")
	ratings.loc[:,"timestamp"] = pd.to_datetime(ratings["timestamp"])
	return ratings

def load_yelp(path):
	ratings = pd.read_csv(path,sep=" ", names = ["userId","movieId","rating","timestamp"])
	#same as movielens datasets
	threshold_user = 20
	
	users_r_count = ratings.groupby("userId")["rating"].count().rename("count").reset_index()
	users_to_remove = users_r_count[users_r_count["count"] < threshold_user]["userId"].as_matrix()
	ratings = ratings[~ratings["userId"].isin(users_to_remove)]

	threshold_item = 50
	items_r_count = ratings.groupby("movieId")["rating"].count().rename("count").reset_index()
	items_to_remove = items_r_count[items_r_count["count"] < threshold_item]["movieId"].as_matrix()
	ratings = ratings[~ratings["movieId"].isin(items_to_remove)]

	ratings.loc[:,"timestamp"] = pd.to_datetime(ratings["timestamp"],unit='s')	
	return ratings

def load_amazon_books(path):
	ratings = pd.read_csv(path,sep=",", names = ["userId","movieId","rating","timestamp"])
	ratings.loc[:,"timestamp"] = pd.to_datetime(ratings["timestamp"],unit='s')
	return ratings

def filter_users(train,test,ratings_count = 10):
	"""
	Filter users from a train and test sets, avoiding cold-start (< than ratings_count) and
	avoiding users that are not in train set.

	Inputs
	---------
		train: pandas DataFrame, containing train set with following columns: ["userId,movieId,rating,timestamp"]
		test: pandas DataFrame, containing test set with following columns: ["userId,movieId,rating,timestamp"]

	Returns
	--------
		(train,test): tuple of filtered train and test sets, in pandas DataFrame format.

	"""
	users_r_count = train.groupby("userId")["rating"].count().rename("count").reset_index()
	users_to_remove = users_r_count[users_r_count["count"] < ratings_count]["userId"].as_matrix()

	# removing users with less than ratings_count in train from train and test
	train_filtered = train[~train["userId"].isin(users_to_remove)]
	test_filtered = test[~test["userId"].isin(users_to_remove)]
	print("Removed "+ str(len(train.userId.unique())-len(train_filtered.userId.unique())) + " users with less than 10 ratings before split date.")

	# removing users that only appear in test
	users_in_train = set(train.userId.unique())
	users_in_test = set(test.userId.unique())
	users_not_in_train = users_in_test - users_in_train
	print("Removed "+ str(len(users_not_in_train))+ " users that only appear in test.")
	test_final = test_filtered[~test_filtered["userId"].isin(list(users_not_in_train))]

	return (train_filtered,test_final)

def filter_items(train,test, remove_ns_only = True):
	"""
	Filter items from a train and test sets, avoiding items that are not in train set.

	Inputs
	---------
		train: pandas DataFrame, containing train set with following columns: ["userId,movieId,rating,timestamp"]
		test: pandas DataFrame, containing test set with following columns: ["userId,movieId,rating,timestamp"]

	Returns
	--------
		(train,test): tuple of filtered train and test sets, in pandas DataFrame format.

	"""

	# removing items that only appear in test
	items_in_train = set(train.movieId.unique())
	items_in_test = set(test.movieId.unique())
	items_not_in_train = items_in_test - items_in_train
	print("Removed "+ str(len(items_not_in_train))+ " items that only appear in test.")
	test_final = test[~test["movieId"].isin(list(items_not_in_train))]
	
	def all_negative(l):
		return sum(l) == -(len(l))

	user_ratings = test_final.groupby("userId")["rating"].apply(list).reset_index()
	user_ratings["all_negative_sampling"] = user_ratings.apply(lambda r,f= all_negative: f(r["rating"]),axis=1)
	users_with_no_real_test_ratings = user_ratings[user_ratings.all_negative_sampling == True]["userId"].unique().tolist()

	print("Removed " + str(len(users_with_no_real_test_ratings)) + " users with only negative sample in test")

	test_final = test_final[~test_final.userId.isin(users_with_no_real_test_ratings)]
	return (train,test_final)

def get_negative_sample(train_dataset,ns_rate):	
	all_movies = set(train_dataset.movieId.unique())
	
	def to_set(x):
		return set(x)

	user_negative_sample = train_dataset.groupby("userId")["movieId"].agg({"seen_movies": to_set})
	users = train_dataset.userId.unique()
	del(train_dataset)
	random.seed(42)

	def negative_sample(r,am,ns_rate):
		unseen = list((am - r["seen_movies"]))
		return random.sample(unseen,min(ns_rate,len(unseen)))

	user_negative_sample.loc[:,"ns"] = user_negative_sample.apply(lambda r,am = all_movies, ns = ns_rate: negative_sample(r,am,ns_rate),axis=1)

	users_ns_map = user_negative_sample.to_dict(orient="index")
	del user_negative_sample
	negative_sampling = []
	for user in users:
		ns = users_ns_map[user]["ns"]
		for movie in ns:
			negative_sampling.append([user,movie,-1,datetime.now(),True])
	del(users_ns_map)
	del(users)
	negative_sampled_df = pd.DataFrame(negative_sampling, columns = ["userId","movieId","rating","timestamp","is_negative_sample"])
	negative_sampled_df.loc[:,"timestamp"] = "0"
	return negative_sampled_df

def add_negative_sampling(train_dataset,test_set,ns_rate):
	train_dataset = pd.concat([train_dataset,test_set])
	all_movies = set(train_dataset.movieId.unique())
	
	def to_set(x):
		return set(x)

	user_negative_sample = train_dataset.groupby("userId")["movieId"].agg({"seen_movies": to_set})
	random.seed(42)

	def negative_sample(r,am,ns_rate):
		unseen = list((am - r["seen_movies"]))
		return random.sample(unseen,min(ns_rate,len(unseen)))

	user_negative_sample.loc[:,"ns"] = user_negative_sample.apply(lambda r,am = all_movies, ns = ns_rate: negative_sample(r,am,ns_rate),axis=1)

	users_ns_map = user_negative_sample.to_dict(orient="index")
	del user_negative_sample
	negative_sampling = []
	for user in test_set.userId.unique():
		ns = users_ns_map[user]["ns"]
		for movie in ns:
			negative_sampling.append([user,movie,-1,datetime.now(),True])
	del(users_ns_map)
	negative_sampled_df = pd.DataFrame(negative_sampling, columns = ["userId","movieId","rating","timestamp","is_negative_sample"])
	del(negative_sampling)
	test_set.loc[:,"is_negative_sample"] = False
	return pd.concat([test_set,negative_sampled_df])

def split_by_timestamp(ratings, train_dataset, split_point=0.6, verbose = True, filter_user_and_items = True, first_split = False, ns_rate = 1000):
	"""

	Inputs
	---------
		ratings: pandas DataFrame, with following columns: ["userId,movieId,rating,timestamp"]
		split_point: float indicating days percentage to keep in train set

	Returns
	--------
		(train,test): tuple of train and test sets in pandas DataFrame format.

	"""
	# ratings = ratings.sort_values("timestamp")	

	train_set = ratings[0 : int(split_point * ratings.shape[0])]
	test_set = ratings[int(split_point * ratings.shape[0]):]
		
	del(ratings)

	if(filter_user_and_items):
		train_set,test_set = filter_users(train_set,test_set)
		train_set,test_set = filter_items(train_set,test_set)

	negative_sample = None
	val = None
	if(first_split):
		negative_sample = get_negative_sample(pd.concat([train_set,test_set]),ns_rate)
	else:
		val = train_set
		train_set = test_set[0: int(test_set.shape[0] * 0.75)][["userId","movieId","rating","timestamp"]]
		test_set = test_set[int(test_set.shape[0] * 0.75):][["userId","movieId","rating","timestamp"]]
		# val = test_set[0:(test_set.shape[0]/2)][["userId","movieId","rating","timestamp"]]
		full_start = pd.concat([train_dataset,train_set,val])
		del(train_dataset)
		test_set = add_negative_sampling(full_start,test_set,ns_rate)
	
	assert(train_set.timestamp.max() <= test_set.timestamp.min())
	return (train_set,test_set,negative_sample,val)

def sample_user_wise(dataset,sample_percentage):
	users = dataset["userId"].unique()	
	users_to_filter = np.random.choice(users,int(sample_percentage* len(users)))
	dataset = dataset[~dataset["userId"].isin(users_to_filter)]
	return dataset

def main():
	
	"""

	This script runs split by timestamp twice on movielens datasets,
	storing them to files. 16GB machine can run each dataset preparation in isolation.

	"""

	parser = optparse.OptionParser()
	parser.add_option('-d', '--datasets', 
						dest="datasets")

	options, remainder = parser.parse_args()	

	datasets = options.datasets.split(",")

	print(datasets)

	datasets_dfs = []

	if("ml100k" in datasets):
		ml100k = load_ml_100k("../data/ml-100k/u.data")
		datasets_dfs.append(ml100k)
		ml100k.to_csv("./created_data/ml100k.csv")
	if("ml1m" in datasets):
		ml1m  = load_ml_1m("../data/ml-1m/ratings.dat")
		datasets_dfs.append(ml1m)
		ml1m.to_csv("./created_data/ml1m.csv")	
	if("ml20m" in datasets):
		ml20m  = load_ml_20m("../data/ml-20m/ratings.csv")
		datasets_dfs.append(ml20m)
		ml20m.to_csv("./created_data/ml20m.csv")	
	if("netflix" in datasets):
		netflix = load_netflix("../data/netflix/prepared_df.csv")
		datasets_dfs.append(netflix)
		netflix.to_csv("./created_data/netflix.csv")	
	if("lastfm1k" in datasets):
		lastfm1k = load_lastfm_1k("../data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv")
		datasets_dfs.append(lastfm1k)
		lastfm1k.to_csv("./created_data/lastfm1k.csv",header=True,index=False)
	if("yelp" in datasets):
		yelp = load_yelp("../data/yelp/ratings.txt")
		datasets_dfs.append(yelp)
		yelp.to_csv("./created_data/yelp.csv",header=True,index=False)

	if("amazon_books" in datasets):
		amazon_books = load_amazon_books("../data/amazon_books/ratings_Books.csv")		
		datasets_dfs.append(amazon_books)
		amazon_books.to_csv("./created_data/amazon_books.csv",header=True,index=False)
	
	if("amazon_movies" in datasets):
		amazon_books = load_amazon_books("../data/amazon_movies/ratings_Movies_and_TV.csv")		
		datasets_dfs.append(amazon_books)
		amazon_books.to_csv("./created_data/amazon_movies.csv",header=True,index=False)
	
	if("amazon_electronics" in datasets):
		amazon_books = load_amazon_books("../data/amazon_electronics/ratings_Electronics.csv")		
		datasets_dfs.append(amazon_books)
		amazon_books.to_csv("./created_data/amazon_electronics.csv",header=True,index=False)

	for (name,dataset) in zip(datasets,datasets_dfs):
		ns_rate = 20
		print("======================================")
		print(" Preparing dataset "+str(name))
		print("======================================\n")

		dataset = dataset.sort_values("timestamp")		
		if(name in datesets_sample):
			users_in_all_sets = (set(dataset[0:int(0.3 * dataset.shape[0])].userId.unique()) &\
								 set(dataset[int(0.31 * dataset.shape[0]):int(0.60 * dataset.shape[0])].userId.unique()) &\
								 set(dataset[int(0.61 * dataset.shape[0]):int(0.90 * dataset.shape[0])].userId.unique()) & \
								 set(dataset[int(0.91 * dataset.shape[0]):].userId.unique()))
			dataset = dataset[dataset["userId"].isin(users_in_all_sets)]
			# print("Sampling "+str(sample_percentage) + " %")
			# dataset = sample_user_wise(dataset,sample_percentage)
			print(dataset.userId.unique().shape)

		train, test, ns, _ = split_by_timestamp(dataset, None, split_point = 0.3, first_split= True, filter_user_and_items = False, ns_rate = ns_rate)
		del(dataset)		
		columns_order = ["userId","movieId","rating","timestamp"]		
		train_ensembles, test_ensembles, _, val = split_by_timestamp(test, train, split_point = (3.0/7.0), first_split = False, filter_user_and_items = False, ns_rate = ns_rate)	

		print("Before filtering items and users in all sets: ")		
		print("Train RS # ratings: "+ str(train.shape[0]))
		print("Val # ratings : "+str(val.shape[0]))
		print("Train ensembles # ratings : "+str(train_ensembles.shape[0]))
		print("Test # ratings : "+str(test_ensembles[test_ensembles["is_negative_sample"]==False].shape[0]))
		print("\n")
		

		users_count = (set(train.userId.unique()) | set(train_ensembles.userId.unique()) | set(val.userId.unique()) | set(test_ensembles.userId.unique()))
		items_count = (set(train.movieId.unique()) | set(train_ensembles.movieId.unique()) | set(val.movieId.unique()) | set(test_ensembles.movieId.unique()))
		ratings_count = (train.shape[0] + val.shape[0] + train_ensembles.shape[0] + test_ensembles.shape[0])
		density = float(ratings_count)/ (len(users_count) * len(items_count))
		stats_line = [name, len(users_count), len(items_count), ratings_count, density * 100]
		print(stats_line)

		users_in_all_sets = (set(train.userId.unique()) & set(train_ensembles.userId.unique()) & set(val.userId.unique()) & set(test_ensembles.userId.unique()))
		#Remove users that are not in all train sets
		train = train[train["userId"].isin(users_in_all_sets)]		
		val = val[val["userId"].isin(users_in_all_sets)]		
		train_ensembles = train_ensembles[train_ensembles["userId"].isin(users_in_all_sets)]		
		test_ensembles = test_ensembles[test_ensembles["userId"].isin(users_in_all_sets)]		
		ns = ns[ns["userId"].isin(users_in_all_sets)]

		#Avoid items we have not seen before in test
		train, test_ensembles = filter_items(train,test_ensembles)
		
		users_count = (set(train.userId.unique()) | set(train_ensembles.userId.unique()) | set(val.userId.unique()) | set(test_ensembles.userId.unique()))
		items_count = (set(train.movieId.unique()) | set(train_ensembles.movieId.unique()) | set(val.movieId.unique()) | set(test_ensembles.movieId.unique()))
		ratings_count = (train.shape[0] + val.shape[0] + train_ensembles.shape[0] + test_ensembles.shape[0])
		density = float(ratings_count)/ (len(users_count) * len(items_count))
		stats_line = [name, len(users_count), len(items_count), ratings_count, density]
		print(stats_line)

		users_sample_size = 3174
		if(name in datesets_sample and len(set(train.userId.unique()) & set(train_ensembles.userId.unique()) & set(val.userId.unique()) & set(test_ensembles.userId.unique())) > users_sample_size):
			users_sample = random.sample(set(train.userId.unique()) & set(train_ensembles.userId.unique()) & set(val.userId.unique()) & set(test_ensembles.userId.unique()), users_sample_size)
			train = train[train["userId"].isin(users_sample)]
			val = val[val["userId"].isin(users_sample)]		
			train_ensembles = train_ensembles[train_ensembles["userId"].isin(users_sample)]		
			test_ensembles = test_ensembles[test_ensembles["userId"].isin(users_sample)]		
			ns = ns[ns["userId"].isin(users_sample)]

		print("\nMax dates: ")
		print(train.timestamp.max())
		print(val.timestamp.max())
		print(train_ensembles.timestamp.max())
		print(test_ensembles[test_ensembles["is_negative_sample"]==False].timestamp.max())

		ns[[c for c in ns.columns if c!= "is_negative_sample"]].to_csv("./created_data/"+name+"_train_negative_sample.csv",index=False,header=False)
		train[columns_order].to_csv("./created_data/"+name+"_train.csv",index=False,header=False)
		val[columns_order].to_csv("./created_data/"+name+"_validation_set.csv",index=False,header=False)
		train_ensembles[columns_order].to_csv("./created_data/"+name+"_train_ensembles.csv",index=False,header=False)
		test_ensembles[columns_order + ["is_negative_sample"]].to_csv("./created_data/"+name+"_test_ensembles.csv",index=False,header=False)

		print("\n")
		total_size = train.shape[0] + val.shape[0] + train_ensembles.shape[0] + test_ensembles[test_ensembles["is_negative_sample"]==False].shape[0]
		print("After filtering items and users in all sets: ")
		print("Train RS # ratings: "+ str(train.shape[0]) + "(" + str(train.shape[0]/float(total_size)) + ")")
		print("Val # ratings : "+str(val.shape[0])+ "(" + str(val.shape[0]/float(total_size)) + ")")
		print("Train ensembles # ratings : "+str(train_ensembles.shape[0])+ "(" + str(train_ensembles.shape[0]/float(total_size)) + ")")
		print("Test # ratings : "+str(test_ensembles[test_ensembles["is_negative_sample"]==False].shape[0])+ "(" + str(test_ensembles[test_ensembles["is_negative_sample"]==False].shape[0]/float(total_size)) + ")")
		print("\n")
		print("Users that appear in all sets: " + str(len(set(train.userId.unique()) & set(train_ensembles.userId.unique()) & set(val.userId.unique()) & set(test_ensembles.userId.unique()))))	
		# print("Users total: "+str(len(dataset.userId.unique())))		

if __name__ == "__main__":
	main()
