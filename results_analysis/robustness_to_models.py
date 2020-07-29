import optparse
from IPython import embed
import pandas as pd
import numpy as np
import sys
sys.path.append("../experiment2/")
from config import *

from scipy import stats
import itertools
import os
from sklearn.manifold import TSNE

def multiple_testing_correction(pvalues, correction_type="Bonferroni"):
    """
    Consistent with R - print
    correct_pvalues_for_multiple_testing([0.0, 0.01, 0.029, 0.03, 0.031, 0.05,
                                          0.069, 0.07, 0.071, 0.09, 0.1])
    """
    from numpy import array, empty
    pvalues = array(pvalues)
    sample_size = pvalues.shape[0]
    qvalues = empty(sample_size)
    if correction_type == "Bonferroni":
        # Bonferroni correction
        qvalues = sample_size * pvalues
    elif correction_type == "Bonferroni-Holm":
        # Bonferroni-Holm correction
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        for rank, vals in enumerate(values):
            pvalue, i = vals
            qvalues[i] = (sample_size-rank) * pvalue
    elif correction_type == "FDR":
        # Benjamini-Hochberg, AKA - FDR test
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = sample_size - i
            pvalue, index = vals
            new_values.append((sample_size/rank) * pvalue)
        for i in range(0, int(sample_size)-1):
            if new_values[i] < new_values[i+1]:
                new_values[i+1] = new_values[i]
        for i, vals in enumerate(values):
            pvalue, index = vals
            qvalues[index] = new_values[i]
    return qvalues

def main():			
	parser = optparse.OptionParser()
	parser.add_option('-d', '--datasets', 
						dest="datasets")

	options, remainder = parser.parse_args()	

	datasets = options.datasets.split(",")

	print(datasets)

	raw_eval = pd.read_csv("../experiment2/created_data/results/raw_eval_yelp.csv")
	winning_rs_models = {}
	for dataset in datasets:
		rs_scores = pd.read_csv("../experiment2/created_data/tmp/h2_"+dataset+"_user_train_time_features_test_set.csv")
		rs_scores[[c for c in rs_scores.columns if "NDCG" in c]].mean().idxmax()
		winning_rs_models[dataset] = rs_scores[[c for c in rs_scores.columns if "NDCG" in c]].mean().idxmax()

	df = pd.read_csv("../experiment2/created_data/results/robustness_analysis_yelp.csv")
	df["performance_estimate"] = df["performance_estimate"].fillna("")
	df = df[df["dataset"].isin(datasets)]
	
	df["is_all"] = df.apply(lambda r: "_with_MF" if r["is_all"] else "",axis=1)

	df["features"] = df["features"] + "_" + df["performance_estimate"]
	df["features"] = df["features"] +"_" + df["is_all"]
	
	ndcg_lists = pd.DataFrame(df.groupby(["model","ensemble","dataset","features"])["NDCG"].apply(list)).to_dict(orient='dict')["NDCG"]
	means_df_for_tsne = pd.DataFrame(df.groupby(["model","ensemble","dataset","features"])["NDCG"].mean()).to_dict(orient='dict')["NDCG"]

	df_to_dr = pd.DataFrame.from_dict(ndcg_lists,orient="index").reset_index()
	means_df_for_tsne_df = pd.DataFrame.from_dict(means_df_for_tsne,orient="index").reset_index()
	tsne = TSNE(n_components =2,random_state=42,perplexity = 25)
	reduced = pd.DataFrame(tsne.fit_transform(df_to_dr[[c for c in df_to_dr.columns if c not in ["index"]]].as_matrix()),columns=["TSNE_0","TSNE_1"])
	reduced_with_info = reduced.join(df_to_dr[["index"]])
	reduced_with_info["meta-learner"] = reduced_with_info.apply(lambda r: r["index"][0],axis=1)
	reduced_with_info["ensemble"] = reduced_with_info.apply(lambda r: r["index"][1],axis=1)
	reduced_with_info["dataset"] = reduced_with_info.apply(lambda r: r["index"][2],axis=1)
	reduced_with_info["features"] = reduced_with_info.apply(lambda r: r["index"][3].replace(")",""),axis=1)
	reduced_with_info = reduced_with_info[reduced_with_info["meta-learner"]!="LinearReg"]
	reduced_with_info = reduced_with_info[reduced_with_info["features"].isin(["none__","MF__","EF-val-pondered_RMSE_","EF-val-raw_RMSE_","EF-val-pondered_NDCG_","EF-val-raw_NDCG_"])]
	reduced_with_info = reduced_with_info.merge(means_df_for_tsne_df,on="index")
	reduced_with_info = reduced_with_info.replace({'none__': 'RP only', "MF__": "PP", "EF-val-pondered_RMSE_": "PE_weighted_rmse", "EF-val-pondered_NDCG_": "PE_weighted_ndcg","EF-val-raw_RMSE_":"PE_unweighted_rmse", "EF-val-raw_NDCG_": "PE_unweighted_ndcg","LinearSVR":"SVM","MLP":"NeuralNetwork","RF":"RandomForest","GBoostingRegressor":"GradientBoosting"}, regex=True)
	reduced_with_info.to_csv("../experiment2/created_data/tmp/reduced_eval.csv",index=False)
	# embed()

	means_df = df.groupby(["ensemble","features","dataset","model"])["NDCG"].mean().reset_index()
	means_df["NDCG"] = means_df.apply(lambda r: round(r["NDCG"],3),axis=1)
	

	pivoted_df = pd.pivot_table(means_df,index=["model","ensemble","dataset"], columns= ["features"],values="NDCG").reset_index().sort_values(["dataset","ensemble"])

	pivoted_df = pivoted_df.fillna(-1)

	def get_max(r):
		return (["EF-train-pondered_NDCG_","EF-train-pondered_NDCG__with_MF","EF-train-pondered_RMSE_","EF-train-pondered_RMSE__with_MF","EF-train-raw_NDCG_","EF-train-raw_NDCG__with_MF","EF-train-raw_RMSE_","EF-train-raw_RMSE__with_MF","EF-val-pondered_NDCG_","EF-val-pondered_NDCG__with_MF","EF-val-pondered_RMSE_","EF-val-pondered_RMSE__with_MF","EF-val-raw_NDCG_","EF-val-raw_NDCG__with_MF","EF-val-raw_RMSE_","EF-val-raw_RMSE__with_MF","MF__","none__"][r[["EF-train-pondered_NDCG_","EF-train-pondered_NDCG__with_MF","EF-train-pondered_RMSE_","EF-train-pondered_RMSE__with_MF","EF-train-raw_NDCG_","EF-train-raw_NDCG__with_MF","EF-train-raw_RMSE_","EF-train-raw_RMSE__with_MF","EF-val-pondered_NDCG_","EF-val-pondered_NDCG__with_MF","EF-val-pondered_RMSE_","EF-val-pondered_RMSE__with_MF","EF-val-raw_NDCG_","EF-val-raw_NDCG__with_MF","EF-val-raw_RMSE_","EF-val-raw_RMSE__with_MF","MF__","none__"]].as_matrix().argmax()])

	pivoted_df["best_comb"] = pivoted_df.apply(lambda r,f=get_max: f(r),axis=1)
	pivoted_df.to_csv("../experiment2/results_scripts/robustness_analysis_table.csv",index=False)


	def significance_tests(r,dict,col,codes,dataset,to_compare = []):
		models = "^{"	
		if(len(to_compare) == 0):
			to_compare = [c for c in r.index if c != "Ensemble" and "_stat" not in c]
		model_1 = (r["Ensemble"].split("-")[1], r["Ensemble"].split("-")[0], dataset, col)
		p_values = []
		t_values = []
		for idx,col_2 in enumerate(to_compare):
			model_2 = (r["Ensemble"].split("-")[1], r["Ensemble"].split("-")[0], dataset, col_2)
			if(model_1 in dict and model_2 in dict):
				t_value, p_value = stats.ttest_rel(dict[(model_1)],dict[model_2])
				p_values.append(p_value)		
				t_values.append(t_value)		
		p_values_corrected = multiple_testing_correction(p_values)
		for i,v in enumerate(p_values_corrected):
			p_value = p_values_corrected[i]
			t_value = t_values[i]
			if(p_value<0.05 and t_value>0):
				models+=codes[i]

		return models + "}"	

	def significance_tests_h2_table(r,dict,col,codes,dataset):
		models = "^{"	
		to_compare = [c for c in r.index if c != "Dataset" and "_stat" not in c and "best" not in c]
		print(to_compare)
		model_1 = col
		p_values = []
		t_values = []
		for idx,col_2 in enumerate(to_compare):
			model_2 = col_2
			if(model_1 in dict and model_2 in dict):
				t_value, p_value = stats.ttest_rel(dict[(model_1)],dict[model_2])
				p_values.append(p_value)		
				t_values.append(t_value)		
		p_values_corrected = multiple_testing_correction(p_values)
		for i,v in enumerate(p_values_corrected):
			p_value = p_values_corrected[i]
			t_value = t_values[i]
			if(p_value<0.05 and t_value>0):
				models+=codes[i]

		return models + "}"	

	def is_complementary(r,dict,col_1,col_2,dataset):
		is_complementary = ""
		model_1 = (r["Ensemble"].split("-")[1], r["Ensemble"].split("-")[0], dataset, col_1)
		p_values = []
		t_values = []
		model_2 = (r["Ensemble"].split("-")[1], r["Ensemble"].split("-")[0], dataset, col_2)
		if(model_1 in dict and model_2 in dict):
			t_value, p_value = stats.ttest_rel(dict[(model_1)],dict[model_2])
			if(p_value<0.05 and t_value>0):
				is_complementary = "Yes (p=" + ("%.3f" % p_value) + ", t-stat=" +("%.3f" % t_value)+ ")"
			else:
				is_complementary = "No (p=" + ("%.3f" % p_value) + ", t-stat=" +("%.3f" % t_value)+ ")"
				
		return is_complementary

	pivoted_df["Ensemble"] = pivoted_df["ensemble"] + "-" + pivoted_df["model"]
	
	## Table H2
	df_borda_count = raw_eval[raw_eval["model"]=="borda-count"]	
	df_borda_count[df_borda_count["dataset"] == dataset]

	max_values_stream = []
	stream_info = []
	max_values_LTRERS = []
	ltrers_info = []
	for c in pivoted_df.columns:
		if(c!= "dataset" and c != "model" and c!= "ensemble" and c != "Ensemble" and c!= "best_comb" and c !="none__"):
			max_values_stream.append(pivoted_df[pivoted_df["ensemble"] == "STREAM"][c].max())
			stream_info.append((pivoted_df[pivoted_df["ensemble"] == "STREAM"][c].argmax(),c))
			max_values_LTRERS.append(pivoted_df[pivoted_df["ensemble"] == "LTRERS"][c].max())
			ltrers_info.append((pivoted_df[pivoted_df["ensemble"] == "LTRERS"][c].argmax(),c))

	raw_values_dict = {}
	raw_values_dict["Best Solo RS"] = rs_scores[winning_rs_models["yelp"]]
	raw_values_dict["Borda-count"] = df_borda_count["NDCG"]
	# embed()
	stacking_max = pivoted_df[pivoted_df["ensemble"]=="STREAM"]["none__"].argmax()
	# if(stacking_max == 0):
		# stacking_max +=1
	max_line = pivoted_df[pivoted_df["ensemble"]=="STREAM"].loc[stacking_max]
	raw_values_dict["Best Stacking"] = ndcg_lists[( max_line["model"],"STREAM",dataset,"none__")]

	stream_max = stream_info[np.argmax((max_values_stream))][0]
	# if(stream_max == 0):
	# 	stream_max +=1
	max_line = pivoted_df[pivoted_df["ensemble"]=="STREAM"].loc[stream_max]
	raw_values_dict["Best STREAM"] = ndcg_lists[(max_line["model"],"STREAM",dataset,stream_info[np.argmax((max_values_stream))][1])]
	
	ltrers_max = ltrers_info[np.argmax((max_values_LTRERS))][0]
	# if(ltrers_max == 0):
	# 	ltrers_max+=1
	max_line = pivoted_df[pivoted_df["ensemble"]=="LTRERS"].loc[ltrers_max]
	raw_values_dict["Best LTRERS"] = ndcg_lists[(max_line["model"],"LTRERS",dataset,ltrers_info[np.argmax((max_values_LTRERS))][1])]


	table_h2 = [dataset,("%.3f" % rs_scores[winning_rs_models["yelp"]].mean()), ("%.3f" % df_borda_count["NDCG"].mean()), \
				("%.3f" % pivoted_df[pivoted_df["ensemble"]=="STREAM"]["none__"].max()), ("%.3f" % max(max_values_stream)), ("%.3f" % max(max_values_LTRERS))]


	table_h2 = pd.DataFrame([table_h2], columns = ["Dataset","Best Solo RS","Borda-count","Best Stacking","Best STREAM","Best LTRERS"])	
	table_h2["Best Solo RS_stat"] = table_h2.apply(lambda r, dict = raw_values_dict, f = significance_tests_h2_table: f(r,dict,"Best Solo RS","abcde","yelp"),axis=1)
	table_h2["Borda-count_stat"] = table_h2.apply(lambda r, dict = raw_values_dict, f = significance_tests_h2_table: f(r,dict,"Borda-count","abcde","yelp"),axis=1)
	table_h2["Best Stacking_stat"] = table_h2.apply(lambda r, dict = raw_values_dict, f = significance_tests_h2_table: f(r,dict,"Best Stacking","abcde","yelp"),axis=1)
	table_h2["Best STREAM_stat"] = table_h2.apply(lambda r, dict = raw_values_dict, f = significance_tests_h2_table: f(r,dict,"Best STREAM","abcde","yelp"),axis=1)
	table_h2["Best LTRERS_stat"] = table_h2.apply(lambda r, dict = raw_values_dict, f = significance_tests_h2_table: f(r,dict,"Best LTRERS","abcde","yelp"),axis=1)

	columns = ["Best Solo RS","Borda-count","Best Stacking","Best STREAM","Best LTRERS"]
	table_h2["best"] = table_h2[columns].astype(float).idxmax(axis=1)

	for c in columns:
		table_h2[c] = table_h2.apply(lambda r,c=c : "\\textbf{" +  ((r[r["best"]])) + "}" if c == r["best"] else ((r[c])),axis=1)


	table_h2["Best Solo RS"] = table_h2["Best Solo RS"].astype(str) + table_h2["Best Solo RS_stat"]
	table_h2["Borda-count"] = table_h2["Borda-count"].astype(str) + table_h2["Borda-count_stat"]
	table_h2["Best Stacking"] = table_h2["Best Stacking"].astype(str) + table_h2["Best Stacking_stat"]
	table_h2["Best STREAM"] = table_h2["Best STREAM"].astype(str) + table_h2["Best STREAM_stat"]
	table_h2["Best LTRERS"] = table_h2["Best LTRERS"].astype(str) + table_h2["Best LTRERS_stat"]

	table_h2 = table_h2[["Dataset","Best Solo RS","Borda-count","Best Stacking","Best STREAM","Best LTRERS"]]
	table_h2.to_csv("../table_h1.csv",index=False)

	delta_plot_df = pd.DataFrame(raw_values_dict["Best LTRERS"],columns=["Best LTRERS"]).join(pd.DataFrame(raw_values_dict["Best STREAM"],columns=["Best STREAM"]))
	delta_plot_df["delta"] = delta_plot_df["Best LTRERS"] - delta_plot_df["Best STREAM"]
	delta_plot_df.to_csv("../experiment2/created_data/delta_h2.csv",index=False)

	## TABLE 4,  for hypothesis H1.e
	table_4 = pivoted_df[["Ensemble",'EF-val-pondered_RMSE_','EF-val-pondered_NDCG_', 'EF-val-raw_RMSE_', 'EF-val-raw_NDCG_']]

	table_4["p_is_complementary"] = table_4.apply(lambda r, dict=ndcg_lists,f=is_complementary: f(r,dict,"EF-val-pondered_NDCG_","EF-val-pondered_RMSE_","yelp"),axis=1)
	table_4["r_is_complementary"] = table_4.apply(lambda r, dict=ndcg_lists,f=is_complementary: f(r,dict,"EF-val-raw_NDCG_","EF-val-raw_RMSE_","yelp"),axis=1)
	
	table_4["best_p"] = table_4[["EF-val-pondered_NDCG_","EF-val-pondered_RMSE_"]].idxmax(axis=1)
	table_4["best_r"] = table_4[["EF-val-raw_NDCG_","EF-val-raw_RMSE_"]].idxmax(axis=1)

	for c in ["EF-val-pondered_NDCG_","EF-val-pondered_RMSE_"]:
		table_4[c] = table_4.apply(lambda r,c=c : "\\textbf{" +  ("%.3f" % (r[r["best_p"]])) + "}" if c == r["best_p"] else ( "%.3f" %  (r[c])),axis=1)

	for c in ["EF-val-raw_NDCG_","EF-val-raw_RMSE_"]:
		table_4[c] = table_4.apply(lambda r,c=c : "\\textbf{" +  ("%.3f" % (r[r["best_r"]])) + "}" if c == r["best_r"] else ( "%.3f" %  (r[c])),axis=1)		

	table_4["Ensemble"] = table_4.apply(lambda r: r["Ensemble"].split("-")[1] , axis=1)

	table_4 = table_4[['Ensemble','EF-val-pondered_NDCG_','EF-val-pondered_RMSE_','p_is_complementary','EF-val-raw_NDCG_','EF-val-raw_RMSE_','r_is_complementary']]
	table_4.columns = ["Ensemble","PE-pondered-NDCG (a) ","PE-pondered-RMSE (b)","(a) better than (b)","PE-raw-NDCG (c) ","PE-raw-RMSE (d)","(c) better than (d)"]
	table_4.to_csv("../table_4.csv",index=False)

	plot_delta_table_4 = pd.DataFrame(ndcg_lists[("ListNet","LTRERS","yelp","EF-val-pondered_NDCG_")],columns = ["LTRERS-ListNet EF-pondered-NDCG"]).join(pd.DataFrame(ndcg_lists[("ListNet","LTRERS","yelp","EF-val-pondered_RMSE_")],columns = ["LTRERS-ListNet EF-pondered-RMSE"]))
	plot_delta_table_4["delta"] = plot_delta_table_4["LTRERS-ListNet EF-pondered-NDCG"] - plot_delta_table_4["LTRERS-ListNet EF-pondered-RMSE"]
	plot_delta_table_4.to_csv("../experiment2/created_data/delta_h1e.csv",index=False)

	## TABLE 3,  for hypothesis H1.d
	table_3 = pivoted_df[["Ensemble",'EF-val-pondered_RMSE_','EF-train-pondered_RMSE_', 'EF-val-raw_RMSE_', 'EF-train-raw_RMSE_']]

	table_3["p_is_complementary"] = table_3.apply(lambda r, dict=ndcg_lists,f=is_complementary: f(r,dict,"EF-val-pondered_RMSE_","EF-train-pondered_RMSE_","yelp"),axis=1)
	table_3["r_is_complementary"] = table_3.apply(lambda r, dict=ndcg_lists,f=is_complementary: f(r,dict,"EF-val-raw_RMSE_","EF-train-raw_RMSE_","yelp"),axis=1)
	
	table_3["best_p"] = table_3[["EF-train-pondered_RMSE_","EF-val-pondered_RMSE_"]].idxmax(axis=1)
	table_3["best_r"] = table_3[["EF-train-raw_RMSE_","EF-val-raw_RMSE_"]].idxmax(axis=1)

	for c in ["EF-train-pondered_RMSE_","EF-val-pondered_RMSE_"]:
		table_3[c] = table_3.apply(lambda r,c=c : "\\textbf{" +  ("%.3f" % (r[r["best_p"]])) + "}" if c == r["best_p"] else ( "%.3f" %  (r[c])),axis=1)

	for c in ["EF-train-raw_RMSE_","EF-val-raw_RMSE_"]:
		table_3[c] = table_3.apply(lambda r,c=c : "\\textbf{" +  ("%.3f" % (r[r["best_r"]])) + "}" if c == r["best_r"] else ( "%.3f" %  (r[c])),axis=1)		

	table_3["Ensemble"] = table_3.apply(lambda r: r["Ensemble"].split("-")[1] , axis=1)

	table_3 = table_3[['Ensemble','EF-val-pondered_RMSE_','EF-train-pondered_RMSE_','p_is_complementary','EF-val-raw_RMSE_','EF-train-raw_RMSE_','r_is_complementary']]
	table_3.columns = ["Ensemble","PE-pondered-val (a) ","PE-pondered-train (b)","(a) better than (b)","PE-raw-val (c) ","PE-raw-train (d)","(c) better than (d)"]
	table_3.to_csv("../table_3.csv",index=False)
	
	plot_delta_table_3 = pd.DataFrame(ndcg_lists[("ListNet","LTRERS","yelp","EF-val-pondered_RMSE_")],columns = ["LTRERS-ListNet EF-pondered-NDCG"]).join(pd.DataFrame(ndcg_lists[("ListNet","LTRERS","yelp","EF-train-pondered_RMSE_")],columns = ["LTRERS-ListNet EF-pondered-RMSE"]))
	plot_delta_table_3["delta"] = plot_delta_table_3["LTRERS-ListNet EF-pondered-NDCG"] - plot_delta_table_3["LTRERS-ListNet EF-pondered-RMSE"]
	plot_delta_table_3.to_csv("../experiment2/created_data/delta_h1d.csv",index=False)

	## TABLE 2,  for hypothesis H1.c
	table_2 = pivoted_df[["Ensemble",'EF-val-pondered_RMSE_','EF-val-pondered_RMSE__with_MF', 'EF-val-raw_RMSE_', 'EF-val-raw_RMSE__with_MF']]

	table_2["p_is_complementary"] = table_2.apply(lambda r, dict=ndcg_lists,f=is_complementary: f(r,dict,"EF-val-pondered_RMSE__with_MF","EF-val-pondered_RMSE_","yelp"),axis=1)
	table_2["r_is_complementary"] = table_2.apply(lambda r, dict=ndcg_lists,f=is_complementary: f(r,dict,"EF-val-raw_RMSE__with_MF","EF-val-raw_RMSE_","yelp"),axis=1)
	
	table_2["best_p"] = table_2[["EF-val-pondered_RMSE__with_MF","EF-val-pondered_RMSE_"]].idxmax(axis=1)
	table_2["best_r"] = table_2[["EF-val-raw_RMSE__with_MF","EF-val-raw_RMSE_"]].idxmax(axis=1)

	for c in ["EF-val-pondered_RMSE__with_MF","EF-val-pondered_RMSE_"]:
		table_2[c] = table_2.apply(lambda r,c=c : "\\textbf{" +  ("%.3f" % (r[r["best_p"]])) + "}" if c == r["best_p"] else ( "%.3f" %  (r[c])),axis=1)

	for c in ["EF-val-raw_RMSE__with_MF","EF-val-raw_RMSE_"]:
		table_2[c] = table_2.apply(lambda r,c=c : "\\textbf{" +  ("%.3f" % (r[r["best_r"]])) + "}" if c == r["best_r"] else ( "%.3f" %  (r[c])),axis=1)		

	table_2["Ensemble"] = table_2.apply(lambda r: r["Ensemble"].split("-")[1] , axis=1)

	table_2 = table_2[['Ensemble','EF-val-pondered_RMSE_','EF-val-pondered_RMSE__with_MF','p_is_complementary','EF-val-raw_RMSE_','EF-val-raw_RMSE__with_MF','r_is_complementary']]
	table_2.columns = ["Ensemble","PE-pondered (a) ","PE-pondered + MF","(a) complementary to MF","PE-raw (b) ","PE-raw + MF","(b) complementary to MF"]
	table_2.to_csv("../table_2.csv",index=False)

	plot_delta_table_2 = pd.DataFrame(ndcg_lists[("ListNet","LTRERS","yelp","EF-val-pondered_RMSE_")],columns = ["LTRERS-ListNet EF-pondered-NDCG"]).join(pd.DataFrame(ndcg_lists[("ListNet","LTRERS","yelp","EF-train-pondered_RMSE__with_MF")],columns = ["LTRERS-ListNet EF-pondered-RMSE"]))
	plot_delta_table_2["delta"] = plot_delta_table_2["LTRERS-ListNet EF-pondered-RMSE"] - plot_delta_table_2["LTRERS-ListNet EF-pondered-NDCG"] 
	plot_delta_table_2.to_csv("../experiment2/created_data/delta_h1c.csv",index=False)

	## TABLE 1,  for hypothesis H1.a and H1.b
	table_1 = pivoted_df[["Ensemble",'none__', 'MF__', 'EF-val-pondered_RMSE_', 'EF-val-raw_RMSE_']]

	table_1["RS Pred. only_stat"] = table_1.apply(lambda r,dict=ndcg_lists,f=significance_tests: f(r,dict,"none__","abcd","yelp"),axis=1)
	table_1["+ MF_stat"] = table_1.apply(lambda r,dict=ndcg_lists,f=significance_tests: f(r,dict,"MF__","abcd","yelp"),axis=1)
	table_1["PE-v-p-rmse_stat"] = table_1.apply(lambda r,dict=ndcg_lists,f=significance_tests: f(r,dict,"EF-val-pondered_RMSE_","abcd","yelp"),axis=1)
	table_1["PE-v-r-rmse_stat"] = table_1.apply(lambda r,dict=ndcg_lists,f=significance_tests: f(r,dict,"EF-val-raw_RMSE_","abcd","yelp"),axis=1)

	columns = [c for c in table_1 if c != "Ensemble" and "_stat" not in c]
	table_1["best"] = table_1[columns].idxmax(axis=1)

	table_1["PE-v-p-rmse_imprv_a"] = table_1.apply(lambda r: ("%.2f" % (((r["EF-val-pondered_RMSE_"] - r["none__"])/r["none__"])* 100)) + " \%",axis=1)
	table_1["PE-v-p-rmse_imprv_b"] = table_1.apply(lambda r: ("%.2f" % (((r["EF-val-pondered_RMSE_"] - r["MF__"])/r["MF__"])* 100)) + " \%",axis=1)
	table_1["PE-v-r-rmse_imprv_a"] = table_1.apply(lambda r: ("%.2f" % (((r["EF-val-raw_RMSE_"] - r["none__"])/r["none__"])* 100)) + " \%",axis=1)
	table_1["PE-v-r-rmse_imprv_b"] = table_1.apply(lambda r: ("%.2f" % (((r["EF-val-raw_RMSE_"] - r["MF__"])/r["MF__"])* 100)) + " \%",axis=1)

	for c in columns:
		table_1[c] = table_1.apply(lambda r,c=c : "\\textbf{" +  ("%.3f" % (r[r["best"]])) + "}" if c == r["best"] else ( "%.3f" %  (r[c])),axis=1)

	table_1.columns = ["Ensemble","RS Pred. only","+ MF", "+ PE-v-p-rmse", "+ PE-v-r-rmse", 'RS Pred. only_stat', '+ MF_stat','PE-v-p-rmse_stat','PE-v-r-rmse_stat','best',"PE-v-p-rmse_imprv_a","PE-v-p-rmse_imprv_b","PE-v-r-rmse_imprv_a","PE-v-r-rmse_imprv_b"]

	table_1["RS Pred. only"] = table_1["RS Pred. only"].astype(str) + table_1["RS Pred. only_stat"]
	table_1["+ MF"] = table_1["+ MF"].astype(str) + table_1["+ MF_stat"]
	table_1["+ PE-v-p-rmse"] = table_1["+ PE-v-p-rmse"].astype(str) + table_1["PE-v-p-rmse_stat"]
	table_1["+ PE-v-r-rmse"] = table_1["+ PE-v-r-rmse"].astype(str) + table_1["PE-v-r-rmse_stat"]


	table_1 = table_1[["Ensemble","RS Pred. only","+ MF","+ PE-v-p-rmse","PE-v-p-rmse_imprv_a","PE-v-p-rmse_imprv_b","+ PE-v-r-rmse","PE-v-r-rmse_imprv_a","PE-v-r-rmse_imprv_b"]]
	table_1.columns = ["Ensemble","RS Pred. only (a)","+ MF (b)","+ PE-v-p-rmse (c)","(c) improv. over (a)"," (c) improv. over (b)","+ PE-v-r-rmse (d)","(d) improv. over (a)"," (d) improv. over (b)"]
	table_1["Ensemble"] = table_1.apply(lambda r: r["Ensemble"].split("-")[1] , axis=1)
		
	for c in [c for c in table_1.columns if c != "Ensemble" and "improv" not in c]:
		table_1[c] = "$"+ table_1[c].astype(str) + "$"

	# table_1.to_csv("../table_1.csv",index=False)
	table_1[[c for c in table_1 if "improv" not in c]].to_csv("../table_1.csv",index=False)

	plot_delta_table_1_a = pd.DataFrame(ndcg_lists[("ListNet","LTRERS","yelp","EF-val-pondered_RMSE_")],columns = ["LTRERS-ListNet EF-pondered-NDCG"]).join(pd.DataFrame(ndcg_lists[("ListNet","LTRERS","yelp","none__")],columns = ["LTRERS-ListNet EF-pondered-RMSE"]))
	plot_delta_table_1_a["delta"] = plot_delta_table_1_a["LTRERS-ListNet EF-pondered-NDCG"] - plot_delta_table_1_a["LTRERS-ListNet EF-pondered-RMSE"]
	plot_delta_table_1_a.to_csv("../experiment2/created_data/delta_h1b.csv",index=False)

	plot_delta_table_1_b = pd.DataFrame(ndcg_lists[("ListNet","LTRERS","yelp","EF-val-pondered_RMSE_")],columns = ["LTRERS-ListNet EF-pondered-NDCG"]).join(pd.DataFrame(ndcg_lists[("ListNet","LTRERS","yelp","MF__")],columns = ["LTRERS-ListNet EF-pondered-RMSE"]))
	plot_delta_table_1_b["delta"] = plot_delta_table_1_b["LTRERS-ListNet EF-pondered-NDCG"] - plot_delta_table_1_b["LTRERS-ListNet EF-pondered-RMSE"]
	plot_delta_table_1_b.to_csv("../experiment2/created_data/delta_h1a.csv",index=False)
	
	# for dataset in pivoted_df.dataset.unique():
	# 	no_features_vs_error_features = {"LTRERS":0,"STREAM":0,"total":0}
	# 	print(dataset)
	# 	print("=============================================")
	# 	print("Comparing no features vs using error-features val")
	# 	print("=============================================")
		
	# 	for ensemble in ["LTRERS","STREAM"]:

	# 		for model in ["LinearRegression",'AdaBoost','ListNet','ExtraTreesRegressor','GBoostingRegressor','KNN','KNeighborsRegressor','LR','LambdaMART','LinearReg','LinearSVR','ListNet','MLP','NaiveBayes','RF','RankBoost','SGDRegressor','SVC','SVR','XGB']: 
	# 				model_1 = (model, ensemble, dataset, "error-features-val")
	# 				if(model_1 in ndcg_lists):
	# 					model_2 = (model, ensemble, dataset, "none")

	# 					# print("  Mean of "+ str(model_1) + " = " + str(pd.DataFrame(ndcg_lists[(model_1)]).mean()[0]))
	# 					# print("  Mean of "+ str(model_2) + " = " + str(pd.DataFrame(ndcg_lists[(model_2)]).mean()[0]))

	# 					t_value, p_value = stats.ttest_rel(ndcg_lists[(model_1)],ndcg_lists[model_2])
	# 					# print("  P value = "+str(p_value))
	# 					# print("  Test is " +str(p_value<0.05))
	# 					if(p_value<0.05 and t_value>0):
	# 						no_features_vs_error_features[ensemble] += 1
	# 					no_features_vs_error_features["total"] += 1
		
	# 	print(str(sum([no_features_vs_error_features[k] for k in no_features_vs_error_features.keys() if k != "total"])) + \
	# 		" out of " + str(no_features_vs_error_features["total"]) + " models with error-features-val are statistically better than not using any features.")
	# 	print(no_features_vs_error_features)

	# for dataset in pivoted_df.dataset.unique():
	# 	no_features_vs_error_features_pondered = {"LTRERS":0,"STREAM":0,"total":0}
	# 	print(dataset)
	# 	print("=============================================")
	# 	print("Comparing no features vs using error-features-val-pondered")
	# 	print("=============================================")
		
	# 	for ensemble in ["LTRERS","STREAM"]:

	# 		for model in ["LinearRegression",'AdaBoost','ListNet','ExtraTreesRegressor','GBoostingRegressor','KNN','KNeighborsRegressor','LR','LambdaMART','LinearReg','LinearSVR','ListNet','MLP','NaiveBayes','RF','RankBoost','SGDRegressor','SVC','SVR','XGB']: 
	# 				model_1 = (model, ensemble, dataset, "EF-val-pondered")
	# 				if(model_1 in ndcg_lists):
	# 					model_2 = (model, ensemble, dataset, "none")

	# 					# print("  Mean of "+ str(model_1) + " = " + str(pd.DataFrame(ndcg_lists[(model_1)]).mean()[0]))
	# 					# print("  Mean of "+ str(model_2) + " = " + str(pd.DataFrame(ndcg_lists[(model_2)]).mean()[0]))

	# 					t_value, p_value = stats.ttest_rel(ndcg_lists[(model_1)],ndcg_lists[model_2])
	# 					# print("  P value = "+str(p_value))
	# 					# print("  Test is " +str(p_value<0.05))
	# 					if(p_value<0.05 and t_value>0):
	# 						no_features_vs_error_features_pondered[ensemble] += 1
	# 					no_features_vs_error_features_pondered["total"] += 1
		
	# 	print(str(sum([no_features_vs_error_features_pondered[k] for k in no_features_vs_error_features_pondered.keys() if k != "total"])) + \
	# 		" out of " + str(no_features_vs_error_features_pondered["total"]) + " models with error-features-val-pondered are statistically better than not using any features.")
	# 	print(no_features_vs_error_features_pondered)

	# 	all_vs_meta_features = {"LTRERS":0,"STREAM":0,"total":0}
	# 	print("\n=============================================")
	# 	print("Comparing all vs using meta-features  	    ")
	# 	print("=============================================")
	# 	for ensemble in ["LTRERS","STREAM"]:

	# 		for model in ["LinearRegression",'AdaBoost','ListNet','ExtraTreesRegressor','GBoostingRegressor','KNN','KNeighborsRegressor','LR','LambdaMART','LinearReg','LinearSVR','ListNet','MLP','NaiveBayes','RF','RankBoost','SGDRegressor','SVC','SVR','XGB']: 
	# 				model_1 = (model, ensemble, dataset, "all")
	# 				if(model_1 in ndcg_lists):
	# 					model_2 = (model, ensemble, dataset, "meta-features")

	# 					# print("  Mean of "+ str(model_1) + " = " + str(pd.DataFrame(ndcg_lists[(model_1)]).mean()[0]))
	# 					# print("  Mean of "+ str(model_2) + " = " + str(pd.DataFrame(ndcg_lists[(model_2)]).mean()[0]))

	# 					t_value, p_value = stats.ttest_rel(ndcg_lists[(model_1)],ndcg_lists[model_2])
	# 					# print("  P value = "+str(p_value))
	# 					# print("  Test is " +str(p_value<0.05))
	# 					if(p_value<0.05 and t_value>0):
	# 						all_vs_meta_features[ensemble] += 1
	# 					all_vs_meta_features["total"] += 1
		
	# 	print(str(sum([all_vs_meta_features[k] for k in all_vs_meta_features.keys() if k != "total"])) + \
	# 		" out of " + str(all_vs_meta_features["total"]) + " models with using all features is statistically better than using solely meta-features.")
	# 	print(all_vs_meta_features)


	# 	#? add EF-val-pondered to all? or separate into two analysis.

	# 	ef_vs_meta_features = {"LTRERS":0,"SCB" :0,"STREAM":0,"total":0,"FWLS":0}
	# 	print("\n=============================================")
	# 	print("Comparing error-features-val vs using meta-features")
	# 	print("=============================================")
	# 	for ensemble in ["LTRERS","SCB","STREAM","FWLS"]:

	# 		for model in ["LinearRegression",'AdaBoost','ListNet','ExtraTreesRegressor','GBoostingRegressor','KNN','KNeighborsRegressor','LR','LambdaMART','LinearReg','LinearSVR','ListNet','MLP','NaiveBayes','RF','RankBoost','SGDRegressor','SVC','SVR','XGB']: 
	# 				model_1 = (model, ensemble, dataset, "error-features-val")
	# 				if(model_1 in ndcg_lists):
	# 					model_2 = (model, ensemble, dataset, "meta-features")

	# 					# print("  Mean of "+ str(model_1) + " = " + str(pd.DataFrame(ndcg_lists[(model_1)]).mean()[0]))
	# 					# print("  Mean of "+ str(model_2) + " = " + str(pd.DataFrame(ndcg_lists[(model_2)]).mean()[0]))

	# 					t_value, p_value = stats.ttest_rel(ndcg_lists[(model_1)],ndcg_lists[model_2])
	# 					# print("  P value = "+str(p_value))
	# 					# print("  Test is " +str(p_value<0.05))
	# 					if(p_value<0.05 and t_value>0):
	# 						ef_vs_meta_features[ensemble] += 1
	# 					ef_vs_meta_features["total"] += 1
		
	# 	print(str(sum([ef_vs_meta_features[k] for k in ef_vs_meta_features.keys() if k != "total"])) + \
	# 		" out of " + str(ef_vs_meta_features["total"]) + " models with using error-features is statistically better than using meta-features.")
	# 	print(ef_vs_meta_features)


	# 	ef_pondered_vs_meta_features = {"LTRERS":0,"STREAM":0,"total":0}
	# 	print("\n=============================================")
	# 	print("Comparing error-features-val-pondered vs using meta-features")
	# 	print("=============================================")
	# 	for ensemble in ["LTRERS","STREAM"]:

	# 		for model in ["LinearRegression",'AdaBoost','ListNet','ExtraTreesRegressor','GBoostingRegressor','KNN','KNeighborsRegressor','LR','LambdaMART','LinearReg','LinearSVR','ListNet','MLP','NaiveBayes','RF','RankBoost','SGDRegressor','SVC','SVR','XGB']: 
	# 				model_1 = (model, ensemble, dataset, "EF-val-pondered")
	# 				if(model_1 in ndcg_lists):
	# 					model_2 = (model, ensemble, dataset, "meta-features")

	# 					# print("  Mean of "+ str(model_1) + " = " + str(pd.DataFrame(ndcg_lists[(model_1)]).mean()[0]))
	# 					# print("  Mean of "+ str(model_2) + " = " + str(pd.DataFrame(ndcg_lists[(model_2)]).mean()[0]))

	# 					t_value, p_value = stats.ttest_rel(ndcg_lists[(model_1)],ndcg_lists[model_2])
	# 					# print("  P value = "+str(p_value))
	# 					# print("  Test is " +str(p_value<0.05))
	# 					if(p_value<0.05 and t_value>0):
	# 						ef_pondered_vs_meta_features[ensemble] += 1
	# 					ef_pondered_vs_meta_features["total"] += 1
		
	# 	print(str(sum([ef_pondered_vs_meta_features[k] for k in ef_pondered_vs_meta_features.keys() if k != "total"])) + \
	# 		" out of " + str(ef_pondered_vs_meta_features["total"]) + " models with using error-features-pondered is statistically better than using meta-features.")
	# 	print(ef_pondered_vs_meta_features)


	# 	ef_val_vs_ef_train = {"LTRERS":0,"SCB" :0,"STREAM":0,"total":0,"FWLS":0}
	# 	print("\n=============================================")
	# 	print("Comparing error-features-val vs error-features-train")
	# 	print("=============================================")
	# 	for ensemble in ["LTRERS","SCB","STREAM","FWLS"]:

	# 		for model in ["LinearRegression",'AdaBoost','ListNet','ExtraTreesRegressor','GBoostingRegressor','KNN','KNeighborsRegressor','LR','LambdaMART','LinearReg','LinearSVR','ListNet','MLP','NaiveBayes','RF','RankBoost','SGDRegressor','SVC','SVR','XGB']: 
	# 				model_1 = (model, ensemble, dataset, "error-features-val")
	# 				if(model_1 in ndcg_lists):
	# 					model_2 = (model, ensemble, dataset, "error-features")

	# 					# print("  Mean of "+ str(model_1) + " = " + str(pd.DataFrame(ndcg_lists[(model_1)]).mean()[0]))
	# 					# print("  Mean of "+ str(model_2) + " = " + str(pd.DataFrame(ndcg_lists[(model_2)]).mean()[0]))

	# 					t_value, p_value = stats.ttest_rel(ndcg_lists[(model_1)],ndcg_lists[model_2])
	# 					# print("  P value = "+str(p_value))
	# 					# print("  Test is " +str(p_value<0.05))
	# 					if(p_value<0.05 and t_value>0):
	# 						ef_val_vs_ef_train[ensemble] += 1
	# 					ef_val_vs_ef_train["total"] += 1
		
	# 	print(str(sum([ef_val_vs_ef_train[k] for k in ef_val_vs_ef_train.keys() if k != "total"])) + \
	# 		" out of " + str(ef_val_vs_ef_train["total"]) + " models with using error-features-val is statistically better than using using error-features-train.")
	# 	print(ef_val_vs_ef_train)

	# def significance_tests(r,dict,col):
	# 	models = ""
	# 	codes = "abcdef"
	# 	to_compare = ["EF-val-pondered","all","error-features","error-features-val", "meta-features", "none"]
	# 	model_1 = (r["model"], r["ensemble"], r["dataset"], col)

	# 	for idx,col_2 in enumerate(to_compare):
	# 		model_2 = (r["model"], r["ensemble"], r["dataset"], col_2)
	# 		if(model_1 in ndcg_lists and model_2 in ndcg_lists):
	# 			t_value, p_value = stats.ttest_rel(ndcg_lists[(model_1)],ndcg_lists[model_2])
	# 			if(p_value<0.05 and t_value>0):
	# 				models+=codes[idx]

	# 	return models

	# pivoted_df["all_stat"] = pivoted_df.apply(lambda r,dict=ndcg_lists,f=significance_tests: f(r,dict,"all"),axis=1)
	# pivoted_df["error-features_stat"] = pivoted_df.apply(lambda r,dict=ndcg_lists,f=significance_tests: f(r,dict,"error-features"),axis=1)
	# pivoted_df["error-features-val_stat"] = pivoted_df.apply(lambda r,dict=ndcg_lists,f=significance_tests: f(r,dict,"error-features-val"),axis=1)
	# pivoted_df["meta-features_stat"] = pivoted_df.apply(lambda r,dict=ndcg_lists,f=significance_tests: f(r,dict,"meta-features"),axis=1)
	# pivoted_df["none_stat"] = pivoted_df.apply(lambda r,dict=ndcg_lists,f=significance_tests: f(r,dict,"none"),axis=1)
	# pivoted_df["EF-val-pondered_stat"] = pivoted_df.apply(lambda r,dict=ndcg_lists,f=significance_tests: f(r,dict,"EF-val-pondered"),axis=1)

	# #Adding line with best solo RS
	# remaining_df = []
	# for dataset in datasets:
	# 	df_borda_count = raw_eval[raw_eval["model"]=="borda-count"]
	# 	# df_borda_count = raw_eval[raw_eval["model"].isnull()]
	# 	df_borda_count[df_borda_count["dataset"] == dataset]
	# 	remaining_df.append(["borda-count","RankAggregation",dataset,"-","-","-","-",round(df_borda_count["NDCG"].mean(),3),"-","-","-","-","-","-","-","-"])
		
	# 	remaining_df.append([winning_rs_models[dataset].split("_")[1],"-",dataset,"-","-","-","-",round(rs_scores[winning_rs_models[dataset]].mean(),3),"-","-","-","-","-","-","-","-"])

	# remaining_df = pd.DataFrame(remaining_df, columns = pivoted_df.columns)	

	# columns_order = [
	#  'dataset',
 # 	 'ensemble',
	#  'model',
	#  'EF-val-pondered',
	#  'EF-val-pondered_stat',
	#  'all',
	#  'all_stat',
	#  'error-features',
	#  'error-features_stat',
	#  'error-features-val',
	#  'error-features-val_stat',
	#  'meta-features',
	#  'meta-features_stat',
	#  'none',
	#  'none_stat',
	#  'best_comb']

	# pd.concat([pivoted_df,remaining_df]).replace(-1,"-")[columns_order].to_csv("../experiment2/results_scripts/robustness_analysis_table.csv",index=False)

	# os.system("Rscript robustness_to_models.R")

if __name__ == '__main__':
	main()