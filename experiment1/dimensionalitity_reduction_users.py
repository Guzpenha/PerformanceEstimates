from IPython import embed

import pandas as pd
import numpy as np

from sklearn.manifold import TSNE

def main():
	final_data = []

	for perplexity in range(5,51):
		print("Calculating reduction using perplexity = "+str(perplexity))
		users = pd.read_csv("../data/created/user_features.csv")
		tsne = TSNE(n_components =2,random_state=0,perplexity=perplexity)
		transformed_users = pd.DataFrame(tsne.fit_transform(users[0:5000][[c for c in users.columns if c != "userId"]].as_matrix()),columns=["TSNE_0","TSNE_1"])
		transformed_users -= transformed_users.min()
		transformed_users /= transformed_users.max()
		transformed_users = transformed_users.join(users)

		user_errors = pd.read_csv("../data/created/user_avg_errors.csv")
		user_errors = user_errors.set_index("userId")
		transformed_users = transformed_users.set_index("userId")
		transformed_users_with_errors = transformed_users.join(user_errors,how="inner")

		transformed_users_with_errors["label"] = transformed_users_with_errors.apply(lambda r: "1" if float(r.avg_error)>1.0 else 0,axis=1)		
		transformed_users_with_errors["perplexity"] = perplexity
		final_data.append(transformed_users_with_errors)

	df_by_perplexity = pd.concat(final_data)
	df_by_perplexity.reset_index().to_csv("../data/created/user_2d.csv",header=True,index=False)


if __name__ == "__main__":
	main()
