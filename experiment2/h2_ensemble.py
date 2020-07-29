import optparse
from config import *
from IPython import embed

import os.path

import pandas as pd
import numpy as np

import pickle
# from xgboost import XGBRegressor
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

from surprise import dump
from surprise import Dataset, Reader

from calculate_user_features import UserFeatures
from calculate_item_features import ItemFeatures
from create_hypothesis_dataset import *

import warnings

warnings.filterwarnings('ignore')
random_state=42

create_h2_test_set_cache = {}

models = [
          ("LinearReg",LinearRegression()),
          ("GBoostingRegressor",GradientBoostingRegressor(random_state=random_state)),
          ("RF",RandomForestRegressor(random_state=random_state)),
            ("LinearSVR",LinearSVR(random_state=random_state)),
          ("MLP",MLPRegressor(random_state=random_state)),
          ]


hyperparameters = [ [("normalize",[True,False])],
                    [("n_estimators",[100, 200]),("max_depth",[2,3])],
                    [("n_estimators",[100, 200])],
                    [("C",[1,0.98,0.97])],						
                    [("hidden_layer_sizes",[(30,),(40,),(50,),(30, 20,)]),("activation",['logistic', 'tanh', 'relu'])]
                    ]
# models = [
# 		  ("LinearReg",LinearRegression()),
#   		  # ("KNeighborsRegressor",KNeighborsRegressor()),
# 		  ("ExtraTreesRegressor",ExtraTreesRegressor(random_state=random_state)),
# 		  ## ("GBoostingRegressor",GradientBoostingRegressor(arandom_state=random_state)),
# 		  ("RF",RandomForestRegressor(random_state=random_state)),
#   		  ##("SVR",SVR()),
#   		  ("LinearSVR",LinearSVR(random_state=random_state)),
# 		  ("SGDRegressor",SGDRegressor(random_state=random_state))#,
# 		  ## ("MLP",MLPRegressor(hidden_layer_sizes=(30,20,),random_state=random_state))#,
# 		  ]

# # hyperparameters = [[] for m in models]

# hyperparameters = [ [("normalize",[True,False])],
# 					# [("n_neighbors",[5,6,7])],
# 				    [("n_estimators",[10,15,50])],
# 				    ## [("n_estimators",[100,110]),("max_depth",[2,3])],
# 				    [("n_estimators",[10,15,50])],
# 					[("C",[1,0.98,0.97])],
# 					## [("C",[1,0.98,0.97])],
# 					[("alpha",[0.0001,0.0002,0.0005])],
# 					[]
# 				    ## [("hidden_layer_sizes",[(30,),(40,),(50,),(30,10,)]),("activation",['logistic', 'tanh', 'relu'])]						
# 					]

def model_selection(X,y, rep=10, is_fwls = False):
    """ Uses grid search and cross validation to choose the best clf for the task (X,y)"""
    all_models_from_grid_search = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    global models
    global hyperparameters

    # linear stacking if FWLS baseline is being used
    if(is_fwls)	: 
        models = [models[0]]
        hyperparameters = [hyperparameters[0]]

    best_est = None
    best_score = 10000000000000000
    results_summary = []
    for model, hyperp_setting in zip(models,hyperparameters):
        print("Fitting "+model[0])		
        
        pipeline = Pipeline([model])		
        param_grid = {}
        
        for param in hyperp_setting:
            param_grid[model[0]+"__"+param[0]] = param[1]
        
        grid_search = GridSearchCV(pipeline,param_grid=param_grid,verbose=True,scoring="neg_mean_absolute_error",cv=3, n_jobs=5)
        grid_search.fit(X_train,y_train)

        clf = grid_search.best_estimator_
        scores = []
        # np.random.seed(random_state)
        # for i in range(0,rep):
        # 	rows = np.random.randint(2, size=len(X_train)).astype('bool')									
        # 	clf.fit(np.array(X_train)[rows],np.array(y_train)[rows])
        # 	preds = clf.predict(X_test)
        # 	scores.append(mean_absolute_error(y_test,preds))
        # results_summary.append([model,scores])
        # avg_score = pd.DataFrame(scores).mean()[0]
        # if(avg_score < best_score):
            # best_score = avg_score
            # best_est = clf

        if(not is_fwls):
            clf.fit(X,y)
            all_models_from_grid_search.append((model[0],clf))

    return clf, best_score, results_summary, all_models_from_grid_search
    # return best_est, best_score, results_summary, all_models_from_grid_search

def create_h2_test_set(dataset_name):

    reader = Reader(line_format='user item rating timestamp', sep=',')
    train = Dataset.load_from_file("./created_data/"+dataset_name+"_train.csv", reader=reader)
    test_ensembles = Dataset.load_from_file("./created_data/"+dataset_name+"_test_ensembles.csv", reader=reader)
        
    uf = UserFeatures(pd.DataFrame(train.raw_ratings,columns = ["userId","movieId","rating","timestamp"]),False)
    user_features_df = uf.get_all_user_features().fillna(0.0)

    itemF = ItemFeatures(pd.DataFrame(train.raw_ratings,columns = ["userId","movieId","rating","timestamp"]),False)
    item_features_df = itemF.get_all_item_features()

    user_train_time_features = pd.read_csv("./created_data/tmp/h2_"+dataset_name+"_user_train_time_features.csv")	
    user_train_time_features["userId"] = user_train_time_features["userId"].astype(str)

    user_train_time_features_val = pd.read_csv("./created_data/tmp/h2_"+dataset_name+"_user_train_time_features_val_set.csv")	
    user_train_time_features_val["userId"] = user_train_time_features_val["userId"].astype(str)

    recs_predictions = pd.DataFrame(test_ensembles.raw_ratings,columns = ["userId","movieId","rating","timestamp"])	
    for rs in RS:
        if("amazon" in dataset_name and rs["name"] == "SlopeOne"):
            continue
        #Memory error for 16GB machine or float division error for lastfm
        if("KNN" in rs["name"] and dataset_name in datasets_knn_mem_error):
            continue
        file_name = os.path.expanduser('./created_data/trained_RS/dump_file_'+dataset_name+'_'+rs["name"])
        _, loaded_algo = dump.load(file_name)

        predictions = loaded_algo.test(test_ensembles.build_full_trainset().build_testset())
        predictions_df = pd.DataFrame(predictions,columns = ["userId","movieId","rating","prediction_"+rs["name"],"details"])			
        recs_predictions = recs_predictions.merge(predictions_df[["userId","movieId","prediction_"+rs["name"]]],on = ["userId","movieId"])
        
    X = ponder_rs_preds_by_ef(recs_predictions, user_train_time_features_val, "val")	
    X = ponder_rs_preds_by_ef(X, user_train_time_features, "train")	
    X = X.merge(user_features_df,on="userId")
    X = X.merge(item_features_df,on="movieId", how = "left").fillna(0.0)
    X = X.merge(user_train_time_features,on="userId")
    X = X.merge(user_train_time_features_val,on="userId",how="left")

    y = X[["rating"]]
    #TODO check if this is exatcly the same as H2_hypothesis data
    X = X[[c for c in X.columns if c not in ["index","userId","movieId","timestamp","rating"]]]

    assert recs_predictions.shape[0] == X.shape[0]
    assert X.isnull().values.any() == False
    return X,y,recs_predictions

def ponder_rs_preds_by_ef(preds_df, estimates_df, prefix, to_copy = True):
    if(to_copy):
        df = preds_df.copy()
    else:
        df = preds_df

    for pred_col in df.columns:
        if ("prediction" in pred_col and "pondered" not in pred_col):
            correct_column =[c for c in estimates_df if "NDCG" in c and ''.join(pred_col.split("prediction_")) in c]
            if(len(correct_column)>0) :
                ndcg_estimate = correct_column[0]
                df[prefix+"_NDCG_pondered_"+pred_col] = df[pred_col] * estimates_df[ndcg_estimate]
                
            correct_column =[c for c in estimates_df if "RMSE" in c and ''.join(pred_col.split("prediction_")) in c]
            if(len(correct_column)>0) :
                rmse_estimate = correct_column[0]
                df[prefix+"_RMSE_pondered_"+pred_col] = df[pred_col] * estimates_df[rmse_estimate]
    return df

def fit_h2_ensemble(dataset_name,regression_task_df, features_used, performance_estimate):
    global create_h2_test_set_cache
    reps = 1
    sample_division_size = 1
    if(dataset_name in h2_sample):
        sample_division_size = h2_sample_division
        print(regression_task_df.shape)
        regression_task_df = regression_task_df[0:regression_task_df.shape[0]/sample_division_size]
        print(regression_task_df.shape)

    print("Fitting H2 for "+ dataset_name + " using features: "+ features_used + " and estimate "+performance_estimate)
    pred_cols = [c for c in regression_task_df.columns if "prediction" in c]

    if(features_used == "none"):
        X = regression_task_df[[c for c in regression_task_df.columns if c!= "label" and c != "userId.1" and "prediction" in c]]
    elif(features_used == "MF"):
        X = regression_task_df[[c for c in regression_task_df.columns if c!= "label" and c != "userId.1" and ("MAE" not in c and "RMSE" not in c and "MSE" not in c and "RR" not in c and "NDCG" not in c and "Precision" not in c and  "AP" not in c)]]
    elif("EF-val-raw" in features_used ):
        X = regression_task_df[[c for c in regression_task_df.columns if c!= "label" and c != "userId.1" and ("MAE" in c or "NDCG" in c or "Precision" in c or "AP" in c or "RMSE" in c or "MSE" in c or "RR" in c) and "val_set" in c] + pred_cols]
    elif("EF-train-raw" in features_used ):
        X = regression_task_df[[c for c in regression_task_df.columns if c!= "label" and c != "userId.1" and ("MAE" in c or "NDCG" in c or "Precision" in c or "AP" in c or "RMSE" in c or "MSE" in c or "RR" in c) and "val_set" not in c] + pred_cols]
    elif("EF-val-pondered" in features_used ):
        X = ponder_rs_preds_by_ef(regression_task_df[pred_cols], regression_task_df[[c for c in regression_task_df.columns if c!= "label" and c != "userId.1" and ("MAE" in c or "NDCG" in c or "Precision" in c or "AP" in c or "RMSE" in c or "MSE" in c or "RR" in c) and "val_set" in c]],"val")
    elif("EF-train-pondered" in features_used ):
        X = ponder_rs_preds_by_ef(regression_task_df[pred_cols], regression_task_df[[c for c in regression_task_df.columns if c!= "label" and c != "userId.1" and ("MAE" in c or "NDCG" in c or "Precision" in c or "AP" in c or "RMSE" in c or "MSE" in c or "RR" in c) and "val_set" not in c]],"train")

    if(performance_estimate == "RMSE"):
        X = X[[c for c in X.columns if "NDCG" not in c and "MSE_VAR" not in c]]
    if(performance_estimate == "NDCG"):
        X = X[[c for c in X.columns if "RMSE" not in c and "MSE_VAR" not in c]]

    if("_MF" in features_used):
        X = X.join(regression_task_df[[c for c in regression_task_df.columns if "prediction" not in c and c!= "label" and c != "userId.1" and ("MAE" not in c and "RMSE" not in c and "MSE" not in c and "RR" not in c and "NDCG" not in c and "Precision" not in c and  "AP" not in c)]])	

    X = X[sorted(list(X.columns))]	
    y = regression_task_df["label"] 	
    
    best_regressor, mae, results_summary, all_models = model_selection(X,y,rep=reps)
    print("Finished fit of all models.")
    # print("Best model: ")
    # print(best_regressor)
    # print(pd.DataFrame(results_summary[0][1]).mean())	
    
    if(dataset_name not in create_h2_test_set_cache):
        print("Creating test set.")
        X_test,y_test,raw_ratings = create_h2_test_set(dataset_name)
        create_h2_test_set_cache[dataset_name] = (X_test,y_test,raw_ratings)
    else:
        print("Loading test set from cache.")
        X_test,y_test,raw_ratings = create_h2_test_set_cache[dataset_name]
    
    X_test = X_test[sorted([c for c in X_test.columns if c in X.columns])]

    # print("Fitting model")
    # best_regressor.fit(X,y)
    # preds = best_regressor.predict(X_test)
    # print(mean_absolute_error(preds,y_test))
    # print(np.sqrt(mean_squared_error(preds,y_test)))
    # predictions_df = raw_ratings.join(pd.DataFrame(preds,columns=["prediction_ensemble"]))
    
    # print("Fitting fixed model")
    # fixed_regressor = RandomForestRegressor(random_state=42,n_estimators = 50)
    # fixed_regressor.fit(X,y)	
    # preds = fixed_regressor.predict(X_test)
    # predictions_df_fixed = raw_ratings.join(pd.DataFrame(preds,columns=["prediction_ensemble"]))

    print("Predicting with all models for robustness analysis")
    i=0
    for (model_name, clf) in all_models:		
        predictions_model = clf.predict(X_test)
        preds_df = raw_ratings.join(pd.DataFrame(predictions_model,columns=["prediction_ensemble"]))
        filehandler = open("./created_data/tmp/predictions_all_H2_"+dataset_name+"_"+model_name+"_" + features_used +"_"+performance_estimate+".pkl","wb")		
        if(i==0):
            print(model_name)
            print(clf)
            pickle.dump(preds_df,filehandler)
            i+=1
        else:
            pickle.dump(preds_df[["userId","movieId","prediction_ensemble","rating"]],filehandler)
        filehandler.close()

    # if(features_used == "all"):
    # 	predictions_df.to_csv("./created_data/tmp/predictions_H2"+dataset_name+".csv",header=True,index=False)		
    # 	predictions_df_fixed.to_csv("./created_data/tmp/predictions_H2"+dataset_name+"_fixed.csv",header=True,index=False)		
    # elif(features_used == "none"):
    # 	predictions_df.to_csv("./created_data/tmp/predictions_H2"+dataset_name+"_no_meta_features.csv",header=True,index=False)
    # 	predictions_df_fixed.to_csv("./created_data/tmp/predictions_H2"+dataset_name+"_no_meta_features_fixed.csv",header=True,index=False)
    # elif(features_used == "meta-features"):
    # 	predictions_df.to_csv("./created_data/tmp/predictions_H2"+dataset_name+"_meta_features_only.csv",header=True,index=False)		
    # 	predictions_df_fixed.to_csv("./created_data/tmp/predictions_H2"+dataset_name+"_meta_features_only_fixed.csv",header=True,index=False)		
    # elif(features_used == "error-features"):
    # 	predictions_df.to_csv("./created_data/tmp/predictions_H2"+dataset_name+"_error_features_only.csv",header=True,index=False)		
    # 	predictions_df_fixed.to_csv("./created_data/tmp/predictions_H2"+dataset_name+"_error_features_only_fixed.csv",header=True,index=False)		
    # elif(features_used == "error-features-val"):
    # 	predictions_df.to_csv("./created_data/tmp/predictions_H2"+dataset_name+"_error_features_val_only.csv",header=True,index=False)		
    # 	predictions_df_fixed.to_csv("./created_data/tmp/predictions_H2"+dataset_name+"_error_features_val_only_fixed.csv",header=True,index=False)		
    # elif(features_used == "EF-val-pondered"):
    # 	predictions_df.to_csv("./created_data/tmp/predictions_H2"+dataset_name+"_error_features_val_pondered_only.csv",header=True,index=False)		
    # 	predictions_df_fixed.to_csv("./created_data/tmp/predictions_H2"+dataset_name+"_error_features_val_pondered_only_fixed.csv",header=True,index=False)		

def create_fwls_dataset(X,y,features_used,is_train=False):
    pred_cols = [c for c in X.columns if "prediction" in c]
    if(features_used == "all"):		
        feature_cols = [c for c in X.columns if "prediction" not in c]
    if(features_used == "meta-features"):
        feature_cols = [c for c in X.columns if "prediction" not in c and ("MAE" not in c and "RMSE" not in c and "MSE" not in c and "RR" not in c and "NDCG" not in c and "Precision" not in c and  "AP" not in c)]
    elif(features_used == "error-features"):
        feature_cols = [c for c in X.columns if "prediction" not in c and ("MAE" in c or "NDCG" in c or "Precision" in c or "AP" in c or "RMSE" in c or "MSE" in c or "RR" in c) and "val_set" not in c]
    elif(features_used == "error-features-val"):
        feature_cols = [c for c in X.columns if "prediction" not in c and ("MAE" in c or "NDCG" in c or "Precision" in c or "AP" in c or "RMSE" in c or "MSE" in c or "RR" in c) and "val_set" in c]

    feature_cols = [c for c in feature_cols if "MSE_VAR" not in c and c != "userId.1"]
    seen_models = set()
    feature_cols_filtered = []
    for c in feature_cols:
        model = ''.join(c.split("_")[0:2])
        if model not in seen_models:
            seen_models.add(model)
            feature_cols_filtered.append(c)

    for p_col in pred_cols:
        for f_col in feature_cols_filtered:
            X.loc[:,"FWLS_"+p_col+"_"+f_col] = X[p_col] * X[f_col]
    X = X[[c for c in X.columns if "FWLS" in c]]

    return X,y

def fit_fwls_baseline(dataset_name,regression_task_df, features_used):
    global create_h2_test_set_cache

    if(dataset_name in h2_sample):
        sample_division_size = h2_sample_division * 2
        print(regression_task_df.shape)		
        regression_task_df = regression_task_df[0:regression_task_df.shape[0]/sample_division_size]
        print(regression_task_df.shape)		

    print("Fitting FWLS for "+ dataset_name + " using features: "+ features_used)
    df_with_features = regression_task_df[[c for c in regression_task_df.columns if c!= "label"]]
    df_with_features = df_with_features[df_with_features.columns.sort_values()]
    X,y = create_fwls_dataset(df_with_features,regression_task_df["label"],features_used,is_train=True)	
    X = X[X.columns.sort_values()]
    reps = 1
    sample_division_size = 1


    best_regressor, mae, results_summary, _ = model_selection(X,y,rep=reps, is_fwls = True)
    print("Best model: ")
    print(best_regressor)
    print(pd.DataFrame(results_summary[0][1]).mean())	

    print("Fitting model")
    best_regressor.fit(X,y)
    del(X)

    if(dataset_name not in create_h2_test_set_cache):
        X_test,y_test,raw_ratings = create_h2_test_set(dataset_name)
        create_h2_test_set_cache[dataset_name] = (X_test,y_test,raw_ratings)
    else:
        X_test,y_test,raw_ratings = create_h2_test_set_cache[dataset_name]

    X_test = X_test[X_test.columns.sort_values()]
    X_test,y_test = create_fwls_dataset(X_test,y_test,features_used)
    X_test = X_test[X_test.columns.sort_values()]
    
    preds = best_regressor.predict(X_test)
    del(X_test)

    predictions_df = raw_ratings.join(pd.DataFrame(preds,columns=["prediction_ensemble"]))

    predictions_df.to_csv("./created_data/tmp/predictions_H2"+dataset_name+"_FWLS_baseline_"+features_used+".csv",header=True,index=False)

def main():
    parser = optparse.OptionParser()
    parser.add_option('-d', '--datasets', 
                        dest="datasets")

    options, remainder = parser.parse_args()	

    datasets = options.datasets.split(",")

    print(datasets)
    
    for dataset_name in datasets:		
        regression_task_df = pd.read_csv("./created_data/hypothesis_data/H2_"+dataset_name+".csv")		


        fit_h2_ensemble(dataset_name,regression_task_df,"none","")
        fit_h2_ensemble(dataset_name,regression_task_df,"MF","")

        # ====================#
        # Pondered features   #
        # ====================#

        #RMSE
        fit_h2_ensemble(dataset_name, regression_task_df,"EF-val-pondered","RMSE")
        fit_h2_ensemble(dataset_name, regression_task_df,"EF-train-pondered","RMSE")
            #Complementary to MF analysis
        fit_h2_ensemble(dataset_name, regression_task_df,"EF-val-pondered_MF","RMSE")
        fit_h2_ensemble(dataset_name, regression_task_df,"EF-train-pondered_MF","RMSE")

        #NDCG
        fit_h2_ensemble(dataset_name, regression_task_df,"EF-val-pondered","NDCG")
        fit_h2_ensemble(dataset_name, regression_task_df,"EF-train-pondered","NDCG")
            #Complementary to MF analysis
        fit_h2_ensemble(dataset_name, regression_task_df,"EF-val-pondered_MF","NDCG")
        fit_h2_ensemble(dataset_name, regression_task_df,"EF-train-pondered_MF","NDCG")
        
        # ====================#
        #	Raw features      #
        # ====================#

        #RMSE
        fit_h2_ensemble(dataset_name,regression_task_df,"EF-val-raw","RMSE")
        fit_h2_ensemble(dataset_name,regression_task_df,"EF-train-raw","RMSE")
            #Complementary to MF analysis			
        fit_h2_ensemble(dataset_name,regression_task_df,"EF-val-raw_MF","RMSE")
        fit_h2_ensemble(dataset_name,regression_task_df,"EF-train-raw_MF","RMSE")
            

        #NDCG
        fit_h2_ensemble(dataset_name,regression_task_df,"EF-val-raw","NDCG")
        fit_h2_ensemble(dataset_name,regression_task_df,"EF-train-raw","NDCG")
            #Complementary to MF analysis			
        fit_h2_ensemble(dataset_name,regression_task_df,"EF-val-raw_MF","NDCG")
        fit_h2_ensemble(dataset_name,regression_task_df,"EF-train-raw_MF","NDCG")


        # fit_fwls_baseline(dataset_name,regression_task_df,"error-features-val")
        # fit_fwls_baseline(dataset_name,regression_task_df,"all")
        # fit_fwls_baseline(dataset_name,regression_task_df,"meta-features")
        # fit_fwls_baseline(dataset_name,regression_task_df,"error-features")

if __name__ == "__main__":
    main()
