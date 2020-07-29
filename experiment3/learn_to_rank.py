import optparse
from IPython import embed
import os
import itertools
import sys
sys.path.append("../experiment2/")
from config import *
from calculate_user_features import UserFeatures
from calculate_item_features import ItemFeatures
from h2_ensemble import *
import pandas as pd
import csv
from surprise import dump
from surprise import Dataset, Reader

L2R = [
    {
    "name": "LambdaMART", #pairwise/listwise
    "algo_number": "6",
    "params":{
         '-tree': [1000],#1500],
         '-leaf': [10]#,15]
        }
    }
    ,
    {
    "name": "ListNet", #listwise
    "algo_number": "7",
    "params":{
        '-epoch' : [100]#, 200, 500]
        }
    },
    {
    "name": "AdaRank", #listwise
    "algo_number": "3",
    "params":{		 
         '-tolerance':[0.003]#,0.005]
        }	
    },
    {
    "name": "RankBoost", #pairwise
    "algo_number": "2",
    "params":{
         '-round': [300]#,500]
        }	
    }	
]

features_combinations = ["none_","MF_","EF-val-pondered_RMSE","EF-train-pondered_RMSE","EF-val-pondered_MF_RMSE","EF-train-pondered_MF_RMSE","EF-val-pondered_NDCG","EF-train-pondered_NDCG","EF-val-pondered_MF_NDCG","EF-train-pondered_MF_NDCG","EF-val-raw_RMSE","EF-train-raw_RMSE","EF-val-raw_MF_RMSE","EF-train-raw_MF_RMSE","EF-val-raw_NDCG","EF-train-raw_NDCG","EF-val-raw_MF_NDCG","EF-train-raw_MF_NDCG"]
# features_combinations = ["EF-val-raw_RMSE","EF-train-raw_RMSE","EF-val-raw_MF_RMSE","EF-train-raw_MF_RMSE","EF-val-raw_NDCG","EF-train-raw_NDCG","EF-val-raw_MF_NDCG","EF-train-raw_MF_NDCG"]

def grid_search_l2r(datasets, folds = 3):
    datasets_best_performances = []	
    for dataset_name in datasets:
        for features_comb in features_combinations:
            path = "./created_data/train/l2r_"+dataset_name+"_"+features_comb+".csv"
            best_model = ("",[])
            best_model_performance = -1
            for model in L2R:
                params = []
                for param in model["params"].keys():
                    params.append(model["params"][param])
                params_comb = list(itertools.product(*params))
                # best_performance = -1
                # best_comb = []
                for comb in params_comb:
                # 	command = "java -jar RankLib-2.8.jar -train "+path+" -ranker "+model["algo_number"]+\
                # 		" -kcv " + str(folds) + " -metric2t NDCG@20 " 
                # 		# " -kcv " + str(folds) + " -kcvmd created_data/models/ -kcvmn "+model["name"] + " -metric2t NDCG@20 " 
                    params_comb = []
                    for param in comb:
                        param_name = [n for n in model["params"].keys() if param in model["params"][n]][0]
                # 		command += param_name + " " +str(param) + " "
                        params_comb.append((param_name,param))
                # 	command += " > created_data/models/"+model["name"]+"_log.txt"
                # 	print(command)
                # 	os.system(command)
                # 	performance_file = "created_data/models/"+model["name"]+"_log.txt"
                # 	f = open(performance_file,"r")
                # 	lines = f.readlines()				
                # 	model_performance = float(lines[-1].split("|")[-1].strip())
                # 	print("Feature comb " + features_comb)
                # 	print(model_performance)
                # 	if(model_performance> best_performance):
                # 		best_performance = model_performance
                # 		best_comb = params_comb
                # if(best_performance > best_model_performance):
                    # best_model_performance = best_performance
                    # best_model = (model["algo_number"], best_comb)
                    
                # best_model = (model["algo_number"], params_comb[0])
                
                robustness_analysis_command = ("java -jar RankLib-2.8.jar -train "+path+" -ranker " + model["algo_number"] +" "+ \
                                                " ".join([p[0] +" "+ str(p[1])+" " for p in params_comb]) + \
                                                "-metric2t NDCG@20 -save ./created_data/models/robustness_analysis_"+model["name"]+"_"+dataset_name+"_"+features_comb)
                robustness_analysis_command += "> created_data/models/robustness_analysis_"+model["name"]+"_"+dataset_name+"_"+features_comb+"_log.txt"
                print(robustness_analysis_command)
                os.system(robustness_analysis_command)

            # best_model_command = "java -jar RankLib-2.8.jar -train "+path+" -ranker " + best_model[0] +" "+ \
            #  " ".join([p[0] +" "+ str(p[1])+" " for p in best_model[1]]) + \
            #  "-metric2t NDCG@20 -save ./created_data/models/best_"+dataset_name+"_"+features_comb 
            # best_model_command += "> created_data/models/best_"+dataset_name+"_"+features_comb+"_log.txt"
            # print(best_model_command)
            # os.system(best_model_command)
            datasets_best_performances.append((path,best_model))			

    return datasets_best_performances


def pd_to_l2r(df, dataset, return_features_oder= False):
    """
    from df to : 
    0 qid:18219 1:0.052893 2:1.000000 3:0.750000 4:1.000000 5:0.066225 6:0.000000 7:0.000000 8:0.000000 9:0.000000 10:0.000000 11:0.047634 12:1.000000 13:0.740506 14:1.000000 15:0.058539 16:0.003995 17:0.500000 18:0.400000 19:0.400000 20:0.004121 21:1.000000 22:1.000000 23:0.974510 24:1.000000 25:0.929240 26:1.000000 27:1.000000 28:0.829951 29:1.000000 30:1.000000 31:0.768123 32:1.000000 33:1.000000 34:1.000000 35:1.000000 36:1.000000 37:1.000000 38:1.000000 39:0.998377 40:1.000000 41:0.333333 42:0.434783 43:0.000000 44:0.396910 45:0.447368 46:0.966667 #docid = GX004-93-7097963 inc = 0.0428115405134536 prob = 0.860366
    0 qid:18219 1:0.004959 2:0.000000 3:0.250000 4:0.500000 5:0.006623 6:0.000000 7:0.000000 8:0.000000 9:0.000000 10:0.000000 11:0.004971 12:0.000000 13:0.259494 14:0.521932 15:0.006639 16:0.000896 17:0.714286 18:0.700000 19:0.000000 20:0.001093 21:0.229604 22:0.237068 23:0.200021 24:0.063318 25:0.000000 26:0.000000 27:0.000000 28:0.000000 29:0.310838 30:0.033799 31:0.001398 32:0.025976 33:0.576917 34:0.036302 35:0.001129 36:0.022642 37:0.141223 38:0.212802 39:0.168053 40:0.069556 41:0.333333 42:0.000000 43:0.000000 44:0.019255 45:0.421053 46:0.000000 #docid = GX010-40-4497720 inc = 0.00110683825421716 prob = 0.089706
    """
    df = df.copy()	
    if("amazon" in dataset):
        df["userId"] = df.apply(lambda r : "qid:" + str(r["userId"]),axis=1)
    else:
        df["userId"] = df.apply(lambda r : "qid:" + str(int(r["userId"])),axis=1)
    df["relevance"] = df["relevance"].fillna(0.0) # dunno why there are some NAs here, but they are not on test , so saying they have relevance 0
    df["relevance"] = df["relevance"].astype(int)
    if("amazon" in dataset):
        df["movieIdComment"] = df.apply(lambda r : "#" + str(r["movieId"]),axis=1)
    else:
        df["movieIdComment"] = df.apply(lambda r : "#" + str(int(r["movieId"])),axis=1)
    df_features = [c for c in df.columns if c not in ["relevance","rating","userId","userId.1","movieId","movieIdComment"]]

    for i, f in enumerate(sorted(df_features, reverse=True)):
        df[f] = df.apply(lambda r, f=f, i=i: str(i+1) + ":" + str(r[f]) if (str(r[f])!= "nan") else str(i+1) + ":0.0", axis=1)

    df_final = df[["relevance","userId"] + sorted(df_features, reverse=True) + ["movieIdComment"]]
    print(df_final.columns)

    if(return_features_oder):
        return df_final, sorted(df_features, reverse=True)
    else:
        return df_final


def generate_l2r_train_datasets(datasets):

    for dataset in datasets:
        filehandler = open("../experiment2/created_data/tmp/"+dataset+"_with_relevance.pkl",'rb')
        rel_df = pickle.load(filehandler)
        filehandler.close()
        # rel_df = pd.read_csv("../experiment2/created_data/tmp/"+dataset+"_with_relevance.csv")
        rel_df = rel_df[["userId","movieId","relevance"]]

        train_df = pd.read_csv("../experiment2/created_data/l2r/"+dataset+"_train.csv")
        train_df = train_df[[c for c in train_df.columns if c != "timestamp"]]
        train_df = train_df.merge(rel_df,on=["userId","movieId"],how="left")
        del(rel_df)
        train_df.loc[train_df['rating'] == -1, 'relevance'] = 0

        users_with_train_data = train_df[train_df["relevance"] != 0 ].groupby("userId").filter(lambda x: len(x) >1)		
        train_df = train_df[train_df["userId"].isin(users_with_train_data.userId.unique())]
        del(users_with_train_data)
        cols_for_all = ["userId","relevance","movieId"]
        pred_cols = [c for c in train_df.columns if "prediction" in c]

        print(train_df.shape)		
        if(dataset == "netflix"):
            train_df = train_df.sample(int(train_df.shape[0]/h2_sample_division)).sort_values("userId")		
        print(train_df.shape)

        df_preds_only = train_df[[c for c in train_df.columns if c!= "label" and c != "userId.1" and "prediction" in c] + cols_for_all]
        print("Generating preds only train set")
        pd_to_l2r(df_preds_only,dataset).to_csv("./created_data/train/l2r_"+dataset+"_none_.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)		
        del(df_preds_only)

        mf_columns = [c for c in train_df.columns if c!= "label" and c != "userId.1" and "MAE" not in c and "RMSE" not in c and "MSE" not in c and "RR" not in c and "NDCG" not in c and "Precision" not in c and  "AP" not in c]
        df_meta_features_only = train_df[mf_columns]
        print("Generating MF only train set")
        pd_to_l2r(df_meta_features_only,dataset).to_csv("./created_data/train/l2r_"+dataset+"_MF_.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)	
        del(df_meta_features_only)

        ## raw

        df_ef_train_raw_rmse = train_df[[c for c in train_df.columns if c!= "label" and  ("RMSE" in c) and "val_set" not in c] + cols_for_all + pred_cols]
        print("Generating EF-train-raw train set with RMSE")
        pd_to_l2r(df_ef_train_raw_rmse,dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-train-raw_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)		
        del(df_ef_train_raw_rmse)

        df_ef_train_raw_ndcg = train_df[[c for c in train_df.columns if c!= "label" and  ("NDCG" in c) and "val_set" not in c]+ cols_for_all + pred_cols]
        print("Generating EF-train-raw train set with NDCG")
        pd_to_l2r(df_ef_train_raw_ndcg,dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-train-raw_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)		
        del(df_ef_train_raw_ndcg)


        df_ef_val_raw_rmse = train_df[[c for c in train_df.columns if c!= "label" and  ("RMSE" in c) and "val_set" in c] + cols_for_all + pred_cols]
        print("Generating EF-val-raw train set with RMSE")
        pd_to_l2r(df_ef_val_raw_rmse,dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-val-raw_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)		
        del(df_ef_val_raw_rmse)

        df_ef_val_raw_ndcg = train_df[[c for c in train_df.columns if c!= "label" and  ("NDCG" in c) and "val_set"  in c]+ cols_for_all + pred_cols]
        print("Generating EF-val-raw train set with NDCG")
        pd_to_l2r(df_ef_val_raw_ndcg,dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-val-raw_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)		
        del(df_ef_val_raw_ndcg)

        ## raw + MF
        mf_columns = [c for c in mf_columns if "prediction" not in c and c not in cols_for_all]
        df_ef_train_raw_rmse_mf = train_df[[c for c in train_df.columns if c!= "label" and  ("RMSE" in c) and "val_set" not in c] + cols_for_all + pred_cols + mf_columns]
        print("Generating EF-train-raw train set with RMSE and MF")
        pd_to_l2r(df_ef_train_raw_rmse_mf,dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-train-raw_MF_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)		
        del(df_ef_train_raw_rmse_mf)

        df_ef_train_raw_ndcg_mf = train_df[[c for c in train_df.columns if c!= "label" and  ("NDCG" in c) and "val_set" not in c]+ cols_for_all + pred_cols + mf_columns]
        print("Generating EF-train-raw train set with NDCG and MF")
        pd_to_l2r(df_ef_train_raw_ndcg_mf,dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-train-raw_MF_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)		
        del(df_ef_train_raw_ndcg_mf)


        df_ef_val_raw_rmse_mf = train_df[[c for c in train_df.columns if c!= "label" and  ("RMSE" in c) and "val_set" in c] + cols_for_all + pred_cols + mf_columns]
        print("Generating EF-val-raw train set with RMSE and MF")
        pd_to_l2r(df_ef_val_raw_rmse_mf,dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-val-raw_MF_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)		
        del(df_ef_val_raw_rmse_mf)

        df_ef_val_raw_ndcg_mf = train_df[[c for c in train_df.columns if c!= "label" and  ("NDCG" in c) and "val_set"  in c]+ cols_for_all + pred_cols + mf_columns]
        print("Generating EF-val-raw train set with NDCG and MF")
        pd_to_l2r(df_ef_val_raw_ndcg_mf,dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-val-raw_MF_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)		
        del(df_ef_val_raw_ndcg_mf)

        # pondered

        print("Generating EF-val-pondered train set with NDCG")
        def_ef_val_pondered_ndcg = ponder_rs_preds_by_ef(train_df[pred_cols + cols_for_all], train_df[[c for c in train_df.columns if  ("NDCG" in c) and "val_set"  in c]], "val")
        pd_to_l2r(def_ef_val_pondered_ndcg, dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-val-pondered_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)
        del(def_ef_val_pondered_ndcg)

        print("Generating EF-val-pondered train set with RMSE")
        def_ef_val_pondered_rmse = ponder_rs_preds_by_ef(train_df[pred_cols + cols_for_all], train_df[[c for c in train_df.columns if  ("RMSE" in c) and "val_set"  in c]], "val")
        pd_to_l2r(def_ef_val_pondered_rmse, dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-val-pondered_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)
        del(def_ef_val_pondered_rmse)

        print("Generating EF-train-pondered train set with NDCG")
        def_ef_train_pondered_ndcg = ponder_rs_preds_by_ef(train_df[pred_cols + cols_for_all], train_df[[c for c in train_df.columns if  ("NDCG" in c) and "val_set" not in c]], "train")
        pd_to_l2r(def_ef_train_pondered_ndcg, dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-train-pondered_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)
        del(def_ef_train_pondered_ndcg)

        print("Generating EF-train-pondered train set with RMSE")
        def_ef_train_pondered_rmse = ponder_rs_preds_by_ef(train_df[pred_cols + cols_for_all], train_df[[c for c in train_df.columns if  ("RMSE" in c) and "val_set" not  in c]], "train")
        pd_to_l2r(def_ef_train_pondered_rmse, dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-train-pondered_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)
        del(def_ef_train_pondered_rmse)

        #pondered + MF

        print("Generating EF-val-pondered train set with NDCG and MF")
        def_ef_val_pondered_ndcg = ponder_rs_preds_by_ef(train_df[pred_cols + cols_for_all + mf_columns], train_df[[c for c in train_df.columns if  ("NDCG" in c) and "val_set"  in c]], "val")
        pd_to_l2r(def_ef_val_pondered_ndcg, dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-val-pondered_MF_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)
        del(def_ef_val_pondered_ndcg)

        print("Generating EF-val-pondered train set with RMSE and MF")
        def_ef_val_pondered_rmse = ponder_rs_preds_by_ef(train_df[pred_cols + cols_for_all + mf_columns], train_df[[c for c in train_df.columns if  ("RMSE" in c) and "val_set"  in c]], "val")		
        pd_to_l2r(def_ef_val_pondered_rmse, dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-val-pondered_MF_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)
        del(def_ef_val_pondered_rmse)

        print("Generating EF-train-pondered train set with NDCG and MF")
        def_ef_train_pondered_ndcg = ponder_rs_preds_by_ef(train_df[pred_cols + cols_for_all + mf_columns], train_df[[c for c in train_df.columns if  ("NDCG" in c) and "val_set" not in c]], "train")
        pd_to_l2r(def_ef_train_pondered_ndcg, dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-train-pondered_MF_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)
        del(def_ef_train_pondered_ndcg)

        print("Generating EF-train-pondered train set with RMSE and MF")
        def_ef_train_pondered_rmse = ponder_rs_preds_by_ef(train_df[pred_cols + cols_for_all + mf_columns], train_df[[c for c in train_df.columns if  ("RMSE" in c) and "val_set" not in c]], "train")		
        pd_to_l2r(def_ef_train_pondered_rmse, dataset).to_csv("./created_data/train/l2r_"+dataset+"_EF-train-pondered_MF_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)
        del(def_ef_train_pondered_rmse)

        # df_all = train_df[[c for c in train_df.columns if c!= "label"]]		
        # print("Generating all features train set")
        # pd_to_l2r(df_all,dataset).to_csv("./created_data/train/l2r_"+dataset+"_all.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE)
        # del(df_all)

def generate_l2r_test_datasets(datasets):
    for dataset in datasets:
        filehandler = open("../experiment2/created_data/tmp/"+dataset+"_with_relevance.pkl",'rb')
        rel_df = pickle.load(filehandler)
        filehandler.close()
        # rel_df = pd.read_csv("../experiment2/created_data/tmp/"+dataset+"_with_relevance.csv")
        rel_df = rel_df[["userId","movieId","relevance"]]

        # rs_preds = pd.read_csv("../experiment2/created_data/tmp/predictions_H2"+dataset+".csv")
        filehandler = open("../experiment2/created_data/tmp/predictions_all_H2_"+dataset+"_LinearReg_none_.pkl","rb")
        rs_preds = pickle.load(filehandler)
        filehandler.close()		
        rs_preds = rs_preds[[c for c in rs_preds.columns if "prediction" in c and "ensemble" not in c and "pondered" not in c] + ["userId","movieId","rating"]]

        preds_with_rel = rs_preds.merge(rel_df, on =["userId","movieId"],how="left")
        rs_preds_shape = rs_preds.shape[0]
        del(rs_preds)
        del(rel_df)
        preds_with_rel.loc[preds_with_rel['rating'] == -1, 'relevance'] = 0
        print("Generating preds only test df (none)")
        pd_to_l2r(preds_with_rel.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_none_.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)
        
        reader = Reader(line_format='user item rating timestamp', sep=',')
        train = Dataset.load_from_file("../experiment2/created_data/"+dataset+"_train.csv", reader=reader)
        uf = UserFeatures(pd.DataFrame(train.raw_ratings,columns = ["userId","movieId","rating","timestamp"]),False)
        user_features = uf.get_all_user_features()
        # user_features.to_csv("../experiment2/created_data/tmp/h1_"+dataset+"_user_features_df.csv",index=False,header=True)		
        # user_features = pd.read_csv("../experiment2/created_data/tmp/h1_"+dataset+"_user_features_df.csv")

        itemF = ItemFeatures(pd.DataFrame(train.raw_ratings,columns = ["userId","movieId","rating","timestamp"]),False)
        del(train)
        item_features = itemF.get_all_item_features()
        # item_features.to_csv("./created_data/tmp/h1_"+dataset_name+"_item_features_df.csv",index=False)
        # item_features = pd.read_csv("../experiment2/created_data/tmp/h1_"+dataset+"_item_features_df.csv")
                
        preds_with_rel_mf = preds_with_rel.merge(item_features,on = ["movieId"], how = "left").fillna(0.0)
        preds_with_rel_mf = preds_with_rel_mf.merge(user_features,on = ["userId"])
        del(user_features)
        del(item_features)

        print("Generating MF only test df")
        pd_to_l2r(preds_with_rel_mf.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_MF_.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)		

        error_features = pd.read_csv("../experiment2/created_data/tmp/h2_"+dataset+"_user_train_time_features.csv") 
        error_features_val_set = pd.read_csv("../experiment2/created_data/tmp/h2_"+dataset+"_user_train_time_features_val_set.csv") 

        # Pondered

            #train
        print("Generating EF-train-pondered test df with RMSE")
        preds_and_ef_pondered_val_set = ponder_rs_preds_by_ef(preds_with_rel, error_features[[c for c in error_features if "RMSE" in c or "userId" in c]], "train")
        pd_to_l2r(preds_and_ef_pondered_val_set.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-train-pondered_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)

        print("Generating EF-train-pondered test df with NDCG")
        preds_and_ef_pondered_val_set = ponder_rs_preds_by_ef(preds_with_rel, error_features[[c for c in error_features if "NDCG" in c or "userId" in c]], "train")
        pd_to_l2r(preds_and_ef_pondered_val_set.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-train-pondered_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)

            #val
        print("Generating EF-val-pondered test df with RMSE")
        preds_and_ef_pondered_val_set = ponder_rs_preds_by_ef(preds_with_rel, error_features_val_set[[c for c in error_features_val_set if "RMSE" in c or "userId" in c]], "val")
        pd_to_l2r(preds_and_ef_pondered_val_set.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-val-pondered_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)

        print("Generating EF-val-pondered test df with NDCG")
        preds_and_ef_pondered_val_set = ponder_rs_preds_by_ef(preds_with_rel, error_features_val_set[[c for c in error_features_val_set if "NDCG" in c or "userId" in c]], "val")
        pd_to_l2r(preds_and_ef_pondered_val_set.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-val-pondered_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)


        #Pondered + MF
        print("Generating EF-train-pondered test df with RMSE and MF")
        preds_and_ef_pondered_val_set = ponder_rs_preds_by_ef(preds_with_rel_mf, error_features[[c for c in error_features if "RMSE" in c or "userId" in c]], "train")
        pd_to_l2r(preds_and_ef_pondered_val_set.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-train-pondered_MF_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)

        print("Generating EF-train-pondered test df with NDCG and MF")
        preds_and_ef_pondered_val_set = ponder_rs_preds_by_ef(preds_with_rel_mf, error_features[[c for c in error_features if "NDCG" in c or "userId" in c]], "train")
        pd_to_l2r(preds_and_ef_pondered_val_set.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-train-pondered_MF_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)

        print("Generating EF-val-pondered test df with RMSE and MF")
        preds_and_ef_pondered_val_set = ponder_rs_preds_by_ef(preds_with_rel_mf, error_features_val_set[[c for c in error_features_val_set if "RMSE" in c or "userId" in c]], "val")
        pd_to_l2r(preds_and_ef_pondered_val_set.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-val-pondered_MF_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)

        print("Generating EF-val-pondered test df with NDCG and MF")
        preds_and_ef_pondered_val_set = ponder_rs_preds_by_ef(preds_with_rel_mf, error_features_val_set[[c for c in error_features_val_set if "NDCG" in c or "userId" in c]],"val")
        pd_to_l2r(preds_and_ef_pondered_val_set.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-val-pondered_MF_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)


        # RAW 
            #train
        error_features["userId"] = error_features["userId"].astype(str)
        error_features_val_set["userId"] = error_features_val_set["userId"].astype(str)
        preds_with_ef = preds_with_rel.merge(error_features[[c for c in error_features if "RMSE" in c or "userId" in c]], on = ["userId"])
        print("Generating Ef-train-raw test df with RMSE")
        pd_to_l2r(preds_with_ef.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-train-raw_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)
        
        preds_with_ef = preds_with_rel.merge(error_features[[c for c in error_features if "NDCG" in c or "userId" in c]], on = ["userId"])
        print("Generating Ef-train-raw test df with NDCG")
        pd_to_l2r(preds_with_ef.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-train-raw_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)
        del(preds_with_ef)

            #val
        print("Generating EF-val-raw test df with RMSE")
        preds_with_ef_val_set = preds_with_rel.merge(error_features_val_set[[c for c in error_features_val_set if "RMSE"in c or "userId" in c]], on = ["userId"])		
        pd_to_l2r(preds_with_ef_val_set.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-val-raw_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)

        print("Generating EF-val-raw test df with NDCG")
        preds_with_ef_val_set = preds_with_rel.merge(error_features_val_set[[c for c in error_features_val_set if "NDCG"in c or "userId" in c]], on = ["userId"])		
        pd_to_l2r(preds_with_ef_val_set.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-val-raw_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)

        del(preds_with_ef_val_set)

        # RAW + MF
        
        preds_with_ef = preds_with_rel_mf.merge(error_features[[c for c in error_features if "RMSE" in c or "userId" in c]], on = ["userId"])
        print("Generating Ef-train-raw test df with RMSE and MF")
        pd_to_l2r(preds_with_ef.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-train-raw_MF_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)
        
        preds_with_ef = preds_with_rel_mf.merge(error_features[[c for c in error_features if "NDCG" in c or "userId" in c]], on = ["userId"])
        print("Generating Ef-train-raw test df with NDCG and MF")
        pd_to_l2r(preds_with_ef.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-train-raw_MF_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)
        del(preds_with_ef)
        
        print("Generating EF-val-raw test df with RMSE and MF")
        preds_with_ef_val_set = preds_with_rel_mf.merge(error_features_val_set[[c for c in error_features_val_set if "RMSE"in c or "userId" in c]], on = ["userId"])		
        pd_to_l2r(preds_with_ef_val_set.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-val-raw_MF_RMSE.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)

        print("Generating EF-val-raw test df with NDCG and MF")
        preds_with_ef_val_set = preds_with_rel_mf.merge(error_features_val_set[[c for c in error_features_val_set if "NDCG"in c or "userId" in c]], on = ["userId"])		
        pd_to_l2r(preds_with_ef_val_set.sort_values("userId"),dataset).to_csv("./created_data/test/l2r_"+dataset+"_EF-val-raw_MF_NDCG.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)

        del(preds_with_ef_val_set)

        # preds_with_rel_mf_and_ef  = preds_with_rel_mf.merge(error_features, on = ["userId"]).merge(error_features_val_set,on = ["userId"],how="left")
        # preds_with_rel_mf_and_ef = preds_with_rel_mf_and_ef.fillna(0.0)

        # print("Generating all features test df")
        # df,features = pd_to_l2r(preds_with_rel_mf_and_ef.sort_values("userId"),dataset,True)
        # df.to_csv("./created_data/test/l2r_"+dataset+"_all.csv",index=False,header=False,sep=" ",quoting=csv.QUOTE_NONE,escapechar=None)
        # pd.DataFrame(features).reset_index().to_csv("./created_data/test/l2r_"+dataset+"_all_features_order.csv")


def make_predictions(datasets):	
    for dataset_name in datasets:
        for features_comb in features_combinations:
            # predictions_command = "java -jar RankLib-2.8.jar -rank ./created_data/test/l2r_"+dataset_name+"_"+ features_comb+".csv " + \
            # "-metric2t NDCG@20 -load ./created_data/models/best_"+dataset_name+"_"+features_comb +  " -score ./created_data/predictions/"+dataset_name+"_"+features_comb
            # # print(predictions_command)
            # os.system(predictions_command)

            # test_command = "java -jar RankLib-2.8.jar -test ./created_data/test/l2r_"+dataset_name+"_"+ features_comb+".csv " + \
            # "-metric2t NDCG@20 -load ./created_data/models/best_"+dataset_name+"_"+features_comb
            # # print(test_command)
            # os.system(test_command)

            for model in L2R:
                predictions_command_robustness = "java -jar RankLib-2.8.jar -rank ./created_data/test/l2r_"+dataset_name+"_"+ features_comb+".csv " + \
                "-metric2t NDCG@20 -load ./created_data/models/robustness_analysis_"+model["name"]+"_"+dataset_name+"_"+features_comb +  " -score ./created_data/predictions/robustness_analysis_"+model["name"]+"_"+dataset_name+"_"+features_comb
                # print(predictions_command_robustness)
                os.system(predictions_command_robustness)

                # test_command_robustness = "java -jar RankLib-2.8.jar -test ./created_data/test/l2r_"+dataset_name+"_"+ features_comb+".csv " + \
                # "-metric2t NDCG@20 -load ./created_data/models/robustness_analysis_"+model["name"]+"_"+dataset_name+"_"+features_comb
                # # print(test_command_robustness)
                # os.system(test_command_robustness)




def write_predictions_for_evaluation(datasets):
    #gather predictions from pred. file and make a dataset that can be used by evaluate_ensembles.py easily	
    for dataset_name in datasets:
        for features_comb in features_combinations:
            # l2r_predictions_file = pd.read_csv("./created_data/predictions/"+dataset_name+"_"+features_comb, sep="	", names = ["userId","docId","score"])
            l2r_test_file = pd.read_csv("./created_data/test/l2r_"+dataset_name+"_"+features_comb+".csv", sep = " ",header= None)
            doc_column = l2r_test_file.columns[-1]
            l2r_test_file["rating"] = 1000
            l2r_test_file.loc[l2r_test_file[0] == 0, 'rating'] = -1
            
            if("amazon" in dataset_name):
                l2r_test_file["movieId"] = l2r_test_file.apply(lambda r,c = doc_column: (str(r[c]).split("#")[1]) ,axis=1)
            else:
                l2r_test_file["movieId"] = l2r_test_file.apply(lambda r,c = doc_column: int(str(r[c]).split("#")[1]) ,axis=1)
            
            if("amazon" in dataset_name):
                l2r_test_file['userId'] = l2r_test_file.apply(lambda r: (str(r[1]).split(":")[1]) ,axis=1)
            else:
                l2r_test_file['userId'] = l2r_test_file.apply(lambda r: int(str(r[1]).split(":")[1]) ,axis=1)
            l2r_test_file['docId'] = l2r_test_file.groupby('userId').cumcount()
            # preds_with_item_id = l2r_predictions_file.merge(l2r_test_file, on = ["userId","docId"])
            # preds_with_item_id["prediction_ensemble"] = preds_with_item_id["score"]
            # preds_with_item_id[["userId","movieId","prediction_ensemble","rating"]].to_csv("./created_data/predictions/predictions_l2r_"+dataset_name+"_"+features_comb+".csv",index=False)

            for model in L2R:
                l2r_predictions_file = pd.read_csv("./created_data/predictions/robustness_analysis_"+model["name"]+"_"+dataset_name+"_"+features_comb, sep="	", names = ["userId","docId","score"])				
                doc_column = l2r_test_file.columns[-1]
                preds_with_item_id = l2r_predictions_file.merge(l2r_test_file, on = ["userId","docId"])
                preds_with_item_id["prediction_ensemble"] = preds_with_item_id["score"]
                preds_with_item_id[["userId","movieId","prediction_ensemble","rating"]].to_csv("./created_data/predictions/predictions_l2r_robustness_analysis_"+ model["name"]+"_"+dataset_name+"_"+features_comb+".csv",index=False)
def main():
    parser = optparse.OptionParser()
    parser.add_option('-d', '--datasets', 
                        dest="datasets")

    options, remainder = parser.parse_args()	

    datasets = options.datasets.split(",")

    print(datasets)
    generate_l2r_train_datasets(datasets)
    grid_search_l2r(datasets)
    generate_l2r_test_datasets(datasets)
    make_predictions(datasets)
    write_predictions_for_evaluation(datasets)

if __name__ == "__main__":
    main()
