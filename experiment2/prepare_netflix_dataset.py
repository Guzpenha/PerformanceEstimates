import pandas as pd
from IPython import embed
from os import listdir
from os.path import isfile, join


def prepare_netflix(path):
	files = [f for f in listdir(path) if isfile(join(path, f))]	
	ratings = pd.DataFrame()
	dfs = []
	i=0
	for movie_file in files:
		if(i%100 == 0):
			print(i)
		i+=1
		df = pd.read_csv(path+movie_file,sep=",",names = ["userId","rating","timestamp"])
		df["movieId"] = movie_file.split("mv_")[1].split(".")[0]
		df = df.iloc[1:]
		dfs.append(df)

	return pd.concat(dfs)

def main():
	df = prepare_netflix("../data/netflix/download/training_set/")
	df.to_csv("../data/netflix/prepared_df.csv",index=False,header=True)

if __name__ == "__main__":
	main()