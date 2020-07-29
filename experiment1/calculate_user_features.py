from IPython import embed

import pandas as pd
import numpy as np

import inspect
from functools import reduce

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import evaluate, print_perf

class UserFeatures():
	""" This class is used to calculate user features based on the rating matrix """
	
	def __init__(self, ratings):		
		self.r = ratings
		self.r["timestamp"] = pd.to_datetime(self.r["timestamp"],unit='s')

	def get_all_user_features(self):
		""" Applies all functions to self.r and return all data joined """

		return reduce(lambda x,y: x.join(y,how="left"), 
			[f[1]() for f in inspect.getmembers(self, predicate=inspect.ismethod) \
								if f[0] not in ['__init__','get_all_user_features']]).fillna(0.0)

	def fwls_feature_4(self):
		""" The log of the number of distinct dates on which a user has rated movies"""		

		unique_ts = self.r.groupby("userId")["timestamp"].unique().reset_index()
		unique_ts[inspect.stack()[0][3]] = unique_ts.apply(lambda r,np=np: np.log(len(r.timestamp)),axis=1)
		
		return unique_ts[["userId",inspect.stack()[0][3]]].set_index("userId")

	def fwls_feature_6(self):
		""" The log of the number of user ratings"""

		return self.r.groupby("userId")["movieId"].apply(len).apply(np.log).rename(inspect.stack()[0][3])

	def fwls_feature16(self):
		""" The standard deviation of the user ratings"""

		return self.r.groupby("userId")["rating"].std().rename(inspect.stack()[0][3]).to_frame()
	
	def fwls_feature24(self):
		""" The (regularized) average number of movie ratings for the movies rated by the user"""
		movie_counts = self.r.groupby("movieId")["rating"].count().rename("numMovieRatings").reset_index()
		
		#regularize using min max (article is not clear about on which normalization is used)
		mini = movie_counts["numMovieRatings"].min()
		maxi = movie_counts["numMovieRatings"].max()
		movie_counts["numMovieRatingsRegularized"] = movie_counts.apply(lambda r, mini = mini, maxi = maxi : \
															 (r.numMovieRatings -mini)/ (maxi - mini) ,axis=1)

		r_with_mv_counts = self.r.merge(movie_counts,on = ["movieId"])
		return r_with_mv_counts.groupby("userId")["numMovieRatingsRegularized"].mean().rename(inspect.stack()[0][3]).to_frame()

	def support(self):
		"""Number of ratings ('Combining Predictions for Accurate Recommender Systems' 2010) """
		return self.r.groupby("userId")["movieId"].apply(len).rename(inspect.stack()[0][3])

	def abnormality(self):
		""" Average distance of user ratings to global mean of movies ('A mobile recommender service', 2010) """
		movies_avg = self.r.groupby("movieId")["rating"].mean().rename("movie_avg_rating").reset_index()
		ratings_with_movies_avg = self.r.merge(movies_avg,on=["movieId"])
		ratings_with_movies_avg["diff"] = abs(ratings_with_movies_avg["movie_avg_rating"] - ratings_with_movies_avg["rating"])
		return ratings_with_movies_avg.groupby("userId")["diff"].mean().rename(inspect.stack()[0][3]).to_frame()

	def abnormalityCR(self):
		""" ('Identifying Users with Atypical Preferences to Anticipate Inaccurate Recommendations', 2015)"""

 		movies_avg = self.r.groupby("movieId")["rating"].mean().rename("movie_avg_rating").reset_index()
		movies_std = self.r.groupby("movieId")["rating"].std().rename("movie_rating_std").reset_index()

		mini = movies_std["movie_rating_std"].min()
		maxi = movies_std["movie_rating_std"].max()
		movies_std["scaled_std"] = movies_std.apply(lambda r, mini = mini, maxi = maxi : \
															 (r["movie_rating_std"] -mini)/ (maxi - mini) ,axis=1)
		ratings_with_movies_info = self.r.merge(movies_avg,on = ["movieId"])
		ratings_with_movies_info = ratings_with_movies_info.merge(movies_std,on = ["movieId"])
		ratings_with_movies_info["diff"] = abs(ratings_with_movies_info["movie_avg_rating"] - ratings_with_movies_info["rating"])
		ratings_with_movies_info["measure"]  = (ratings_with_movies_info["diff"] * ratings_with_movies_info["scaled_std"])**2

		return ratings_with_movies_info.groupby("userId")["measure"].mean().rename(inspect.stack()[0][3]).to_frame()

	def avgRatingValue(self):
		""" User avg rating value ('Investigations into User Rating Information and Predictive Accuracy in CF Domain',2012) """
		return self.r.groupby("userId")["rating"].mean().rename(inspect.stack()[0][3]).to_frame()

	def ratingStdDeviation(self):
		""" User std of rating values ('Investigations into User Rating Information and Predictive Accuracy in CF Domain',2012) """
		return self.r.groupby("userId")["rating"].std().rename(inspect.stack()[0][3]).to_frame()

	def moviesPopularity(self):		
		""" 
		The  average number of movie ratings for the movies rated by the user 
		('Investigations into User Rating Information and Predictive Accuracy in CF Domain',2012) 
		"""
		movie_counts = self.r.groupby("movieId")["rating"].count().rename("numMovieRatings").reset_index()
		r_with_mv_counts = self.r.merge(movie_counts,on = ["movieId"])
		return r_with_mv_counts.groupby("userId")["numMovieRatings"].mean().rename(inspect.stack()[0][3]).to_frame()

	def moviesAvgRatings(self):
		""" Avg of ratings from movies rated by the user ('Investigations into User Rating Information and Predictive Accuracy in CF Domain',2012) """
		movies_avg = self.r.groupby("movieId")["rating"].mean().rename("movie_avg_rating").reset_index()
		ratings_with_movies_avg = self.r.merge(movies_avg,on=["movieId"])
		return ratings_with_movies_avg.groupby("userId")["movie_avg_rating"].mean().rename(inspect.stack()[0][3]).to_frame()

	def latentFeatures(self,):
		# Load the movielens-100k dataset (download it if needed),
		# and split it into 3 folds for cross-validation.
		reader = Reader(line_format='user item rating timestamp', sep=',')
		data = Dataset.load_from_file("../data/ml20m_train.csv", reader=reader)
		trainset = data.build_full_trainset()
		algo = SVD(n_factors=10)
		algo.train(trainset)
		
		userLatentFeatures = pd.DataFrame(algo.pu,columns = ["SVD_user_feature_"+str(i) for i in range(0,10)])
		userLatentFeatures["userId"] = self.r.userId.unique()
				
		return userLatentFeatures.set_index("userId")

def main():
	# ratings = pd.read_csv("../data/ml-20m/ratings_sample.csv",sep=",")
	ratings = pd.read_csv("../data/ml20m_train.csv",sep=",",names=["userId","movieId","rating","timestamp"])
	
	uf = UserFeatures(ratings)

	all_features = uf.get_all_user_features()		
	# embed()
	all_features.to_csv("../data/created/user_features.csv",header=True,sep = ",")

if __name__ == "__main__":
	main()