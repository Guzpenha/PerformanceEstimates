from IPython import embed

import pandas as pd
import numpy as np

import inspect
from functools import reduce

class ItemFeatures():
	""" This class is used to calculate item features based on the rating matrix """
	
	def __init__(self, ratings,date_time_convert = True):
		self.r = ratings
		if(date_time_convert):
			self.r["timestamp"] = pd.to_datetime(self.r["timestamp"],unit='s')

	def get_all_item_features(self):
		""" Applies all functions to self.r and return all data joined """		
		return reduce(lambda x,y: x.join(y,how="left"), 
			[f[1]() for f in inspect.getmembers(self, predicate=inspect.ismethod) \
								if f[0] not in ['__init__','get_all_item_features']]).reset_index().fillna(0.0)

	def support(self):
		"""Item version of Number of ratings ('Combining Predictions for Accurate Recommender Systems' 2010) """
		return self.r.groupby("movieId")["rating"].count().rename("item_"+inspect.stack()[0][3])

	def avgRatingValue(self):
		""" Item version of User avg rating value ('Investigations into User Rating Information and Predictive Accuracy in CF Domain',2012) """
		return self.r.groupby("movieId")["rating"].mean().rename("item_"+inspect.stack()[0][3]).to_frame()

	def ratingStdDeviation(self):
		"""Item version of User std of rating values ('Investigations into User Rating Information and Predictive Accuracy in CF Domain',2012) """
		return self.r.groupby("movieId")["rating"].std().fillna(0.0).rename("item_"+inspect.stack()[0][3]).to_frame()

# def main():
# 	ratings = pd.read_csv("../data/ml-20m/ratings_sample.csv",sep=",")
# 	# ratings = pd.read_csv("../data/ml20m_train.csv",sep=",",names=["userId","movieId","rating","timestamp"])
	
# 	uf = ItemFeatures(ratings,False)

# 	all_features = uf.get_all_item_features()		
# 	# embed()	

# if __name__ == "__main__":
# 	main()