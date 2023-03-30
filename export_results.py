import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from matplotlib.pyplot import cm



data_set_name = "give_me_some_credit"
model_type = "ann_50"
sigmas2 = [0.005, 0.01, 0.015, 0.02]
invalidation_targets = [0.05,0.1,0.15, 0.20, 0.25, 0.30,0.35]
methods_save = ["wachter_rip","robust_counterfactuals_random_v2"] 
type_results = "/CROCO_PROBE"
access_file = "recourse_invalidation_results" + type_results + "/"


L = []
for method in methods_save : 
    for sigma in sigmas2 : 
        for invalidation_target in invalidation_targets : 
            df = pd.DataFrame(columns=["Invalidation_rate","Distance","Target","Estimator","Sigma"])
            name = access_file + method + "_" + data_set_name + "_" + model_type + "_sigma2_" + str(sigma)+ "_intarget_" + str(invalidation_target)
            invalidation_rate = pd.read_csv(name + "/delta_testtest.csv")
            recourse_cost = pd.read_csv(name + "/recourse_cost.csv")
            sucess_rate =  np.loadtxt(name + "/sucess.txt")
            if method=="wachter_rip" : 
                estimator = pd.read_csv(name + "/estimate_ir_test.csv")
            elif method=="robust_counterfactuals_v2" : 
                estimator = pd.read_csv(name + "/estimator_testtest.csv")
            
            df["Invalidation_rate"] = invalidation_rate.values.flatten()
            df["Distance"] = recourse_cost["Distance_2"]
            df["Target"] = invalidation_target
            df["Sigma"] = sigma
            df["Estimator"] = estimator.values.flatten()
            df["Method"] = method
            
            L.append(df)
            
access_file_wachter = "recourse_invalidation_results" + "/wachter" + "/"
name_wachter = access_file_wachter + "wachter" + "_" + data_set_name + "_" + model_type + "_sigma2_" + str(sigmas2[0])+ "_intarget_" + str(0.5)
for sigma in sigmas2 : 
    df = pd.DataFrame(columns=["Invalidation_rate","Distance","Target","Estimator","Sigma"])
    recourse_cost = pd.read_csv(name_wachter + "/recourse_cost.csv")
    sucess_rate =  np.loadtxt(name_wachter + "/sucess.txt")
    # Load invalidation rate 
    name_wachter_sigma = access_file_wachter + "wachter" + "_" + data_set_name + "_" + model_type + "_sigma2_" + str(sigma)+ "_intarget_" + str(0.5)
    invalidation_rate = pd.read_csv(name_wachter_sigma + "/delta_testtest.csv")
    
    df["Invalidation_rate"] = invalidation_rate.values.flatten()
    df["Distance"] = recourse_cost["Distance_2"]
    df["Target"] = np.nan
    df["Sigma"] = sigma
    df["Estimator"] = np.nan
    df["Method"] = "wachter"
    L.append(df)
    
df_all = pd.concat(L)
df_all.to_csv("recourse_invalidation_results/formated_results/Results_{}.csv".format(data_set_name))
