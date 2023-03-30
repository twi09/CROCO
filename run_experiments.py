import os
import numpy as np 
import pandas as pd
import torch
import time 
from torch.distributions.multivariate_normal import MultivariateNormal
from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances,predict_positive_instances
from carla.recourse_methods.catalog.actionable_recourse import ActionableRecourse
from carla.recourse_methods.catalog.wachter import Wachter
from carla.recourse_methods.catalog.w_rip import Wachter_rip
from carla.recourse_methods.catalog.robust_counterfactuals import Robust_counterfactuals
from carla.recourse_methods.catalog.growing_spheres import GrowingSpheres
from carla.recourse_methods.catalog.roar import ROAR
from carla.recourse_methods.catalog.arar import ARAR
from carla.recourse_methods.catalog.dice_diverse import DICEDiv
from carla.recourse_methods.catalog.dice import Dice
from carla.evaluation import get_distances,success_rate,compute_recourse_invalidation_rate,compute_estimator,compute_estimate_wachter_rip
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')

training_params_linear = {
    "adult": {"lr": 0.002,
              "epochs": 100,
              "batch_size": 1024},
    "compas": {"lr": 0.002,
               "epochs": 25,
               "batch_size": 128},
    "give_me_some_credit": {"lr": 0.002, 
                            "epochs": 50,
                            "batch_size": 2048},
    "breast_cancer" : {"lr": 0.002,
               "epochs": 25,
               "batch_size": 32} }

training_params_ann = {
    "adult": {"lr": 0.002,
              "epochs": 30,
              "batch_size": 1024},
    "compas": {"lr": 0.002,
               "epochs": 25,
               "batch_size": 25},
    "give_me_some_credit": {"lr": 0.002,
                            "epochs": 50,
                            "batch_size": 2048},
    "breast_cancer": {"lr": 0.002,
               "epochs": 25,
               "batch_size": 32}
}

training_params = {"linear": training_params_linear,
                   "ann": training_params_ann}


def wachter(model, test_factual):
    hyperparams = {"loss_type": "BCE",
                   "binary_cat_features": False}
    df_cfs =  Wachter(model, hyperparams).get_counterfactuals(test_factual)
    return df_cfs



def dice(model, test_factual):
    hyperparams = {"loss_type": "BCE",
                   "binary_cat_features": False}
    
    df_cfs = Dice(model,hyperparams).get_counterfactuals(test_factual)
    return df_cfs


def expect(model, test_factual, sigma2, invalidation_target):
    hyperparams = {"loss_type": "BCE",
                   "binary_cat_features": False,
                   "invalidation_target": invalidation_target,
                   "inval_target_eps": 0.010,
                   "noise_variance": sigma2,
                   "n_iter": 200,
                   "t_max_min": 0.50}
    df_cfs = Wachter_rip(model, hyperparams).get_counterfactuals(test_factual)
    return df_cfs

# OUR METHOD 
def robust_counterfactuals_v2(model,test_factual,sigma2,robustness_target,df_cfs,data,distribution) : 
    hyperparams = {"loss_type": "BCE",
                   "binary_cat_features": False,
                   "sigma2": sigma2,
                   "n_iter": 1000,
                   "t_max_min": 0.50,
                   "robustness_target" : robustness_target,
                   "init_random" : False,
                   "version" : "v2",
                   "robustness_epsilon" : 0.01,
                   "distribution" : distribution}
    df_refined = Robust_counterfactuals(model,hyperparams).get_counterfactuals(test_factual,df_cfs,data)
    return(df_refined)


 

def robust_counterfactuals_random_v2(model,test_factual,sigma2,robustness_target,data,n_samples,distribution) : 
    hyperparams = {"n_samples" : n_samples,
                   "binary_cat_features": False,
                   "sigma2": sigma2,
                   "n_iter": 200,
                   "t_max_min": 0.50,
                   "robustness_target" : robustness_target,
                   "m" : 0.1,
                   "init_random" : True,
                   "version" : "v2",
                   "robustness_epsilon" : 0.01,
                   "distribution" : distribution}
    df_refined = Robust_counterfactuals(model,hyperparams).get_counterfactuals(test_factual,None,data)
    return(df_refined)



def gs(model, test_factual):
    hyperparams = None
    df_cfs = GrowingSpheres(model, hyperparams).get_counterfactuals(test_factual)
    return df_cfs


def ar(model_tf, test_factual):
    coeffs, intercepts = None, None
    hyperparams = {"fs_size": 150}
    
    if model_tf.model_type == "linear":
        # get weights and bias of linear layer for negative class 0
        coeffs_neg = model_tf.raw_model.output.weight[0].detach().numpy()
        intercepts_neg = model_tf.raw_model.output.bias[0].detach().numpy()
        
        # get weights and bias of linear layer for positive class 1
        coeffs_pos = model_tf.raw_model.output.weight[1].detach().numpy()
        intercepts_pos = model_tf.raw_model.output.bias[1].detach().numpy()
        
        coeffs = -(coeffs_neg - coeffs_pos)
        intercepts = -(intercepts_neg - intercepts_pos)
        hyperparams = {"fs_size": 5}
    
    cfs = ActionableRecourse(
        model_tf, hyperparams, coeffs=coeffs, intercepts=intercepts
    ).get_counterfactuals(test_factual)
    return cfs


def arar(model, test_factual, delta=0.01):
    coeffs, intercepts = None, None
    hyperparams = {"delta": delta}
    
    if model.model_type == "linear":
        # get weights and bias of linear layer for negative class 0
        coeffs_neg = model.raw_model.output.weight[0].detach().numpy()
        intercepts_neg = model.raw_model.output.bias[0].detach().numpy()
        
        # get weights and bias of linear layer for positive class 1
        coeffs_pos = model.raw_model.output.weight[1].detach().numpy()
        intercepts_pos = model.raw_model.output.bias[1].detach().numpy()
        
        coeffs = -(coeffs_neg - coeffs_pos)
        intercepts = np.array(-(intercepts_neg - intercepts_pos)).reshape(-1)
    
    cfs = ARAR(
        model, hyperparams, coeffs=coeffs, intercept=intercepts
    ).get_counterfactuals(test_factual)
    return cfs


def roar(model, test_factual, delta=0.01):
    coeffs, intercepts = None, None
    hyperparams = {"delta": delta}
    
    if model.model_type == "linear":
        # get weights and bias of linear layer for negative class 0
        coeffs_neg = model.raw_model.output.weight[0].detach().numpy()
        intercepts_neg = model.raw_model.output.bias[0].detach().numpy()
        
        # get weights and bias of linear layer for positive class 1
        coeffs_pos = model.raw_model.output.weight[1].detach().numpy()
        intercepts_pos = model.raw_model.output.bias[1].detach().numpy()
        
        coeffs = -(coeffs_neg - coeffs_pos)
        intercepts = np.array(-(intercepts_neg - intercepts_pos)).reshape(-1)
    
    cfs = ROAR(
        model, hyperparams, coeffs=coeffs, intercept=intercepts
    ).get_counterfactuals(test_factual)
    return cfs




def run_experiment(cf_method,
                   hidden_width,
                   data_name,
                   model_type,
                   backend,
                   sigma2,
                   invalidation_target,
                   n_cfs=100,
                   n_samples=10_000,
                   distribution="gaussian"
                   ):
    print(
        f"Running experiments with: {cf_method} {data_name} {model_type} {hidden_width}"
    )
    
    data = DataCatalog(data_name)
    
    params = training_params[model_type][data_name]
    model = MLModelCatalog(
        data, model_type, load_online=False, use_pipeline=True, backend=backend
    )
    model.train(
        learning_rate=params["lr"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        hidden_size=hidden_width,
    )
    model.use_pipeline = False
    
    factuals = predict_negative_instances(model, data)
    
    test_factual = factuals.iloc[:n_cfs]
    
    if cf_method == "wachter":
        df_cfs = wachter(model, test_factual)
    elif cf_method == 'wachter_rip':
        if distribution!="gaussian" : 
            raise Exception("No handle other distributions than gaussian")
        df_cfs = expect(model,
                        test_factual,
                        sigma2=sigma2,
                        invalidation_target=invalidation_target)
    elif cf_method == "dice":
        df_cfs = dice(model, test_factual)
    elif cf_method == "ar":
        df_cfs = ar(model, test_factual)
    elif cf_method == "roar":
        df_cfs = roar(model,
                      test_factual,
                      delta=sigma2)  # 0.01
    elif cf_method == "arar":
        df_cfs = arar(model,
                      test_factual,
                      delta=sigma2)  # 0.01
    elif cf_method == "gs":
        df_cfs = gs(model, test_factual)
        
    elif cf_method == "robust_counterfactuals_v2" : 
        df_cfs_c = wachter(model, test_factual)
        df_cfs = robust_counterfactuals_v2(model,test_factual,sigma2,invalidation_target,df_cfs_c,data)

    elif cf_method =="robust_counterfactuals_random_v2" : 
        df_cfs = robust_counterfactuals_random_v2(model,test_factual,sigma2,invalidation_target,data,n_samples,distribution=distribution)

    else:
        raise ValueError(f"cf_method {cf_method} not recognized")
    
    df_cfs = df_cfs.drop(columns=data.target)
    
    
    folder_name = f"{cf_method}_{data_name}_{model_type}_{hidden_width[0]}_sigma2_{sigma2}_intarget_{invalidation_target}"
    if not os.path.exists(f"recourse_invalidation_results/{folder_name}"):
        os.makedirs(f"recourse_invalidation_results/{folder_name}")
    
    
    # normalize factual
    factual_predictions = test_factual[data.target]
    test_factual = model.perform_pipeline(test_factual)
    test_factual["prediction"] = factual_predictions
    
    # Compute invalidation rate metric and return counterfactuals dataFrame with a column that contains predictions 
    df_cfs = compute_recourse_invalidation_rate(df_cfs,model,folder_name,n_samples,sigma2,backend,distribution=distribution)
    
    if cf_method=="robust_counterfactuals_random_v2" : 
        # Compute our monte-carlo estimator 
        compute_estimator(df_cfs.drop(["prediction"],axis=1),model,folder_name,n_samples,sigma2,backend,distribution=distribution)
    elif cf_method == "wachter_rip" :
        # Compute estimate value with wachter rip 
        compute_estimate_wachter_rip(df_cfs.drop(["prediction"],axis=1),model,folder_name,n_samples,sigma2,backend)

    test_factual.to_csv(
        f"recourse_invalidation_results/{folder_name}/factual_testtest.csv",
        sep=",",
        index=False,
    )
    df_cfs.to_csv(
        f"recourse_invalidation_results/{folder_name}/counterfactual_testtest.csv",
        sep=",",
        index=False,
    )
    
    
    # csv with recourse cost 
    columns = ["Distance_1", "Distance_2", "Distance_3", "Distance_4"]
    costs_array = get_distances(np.array(test_factual.drop(["prediction"],axis=1)), np.array(df_cfs.drop(["prediction"],axis=1)))
    costs = pd.DataFrame(costs_array, columns=columns)
    costs.to_csv(
        f"recourse_invalidation_results/{folder_name}/recourse_cost.csv",
        sep=",",
        index=False,
    )
    
    # Np array with sucess 
    success = success_rate(df_cfs)
    np.savetxt(f"recourse_invalidation_results/{folder_name}/sucess.txt",np.array([success]))
    
    

     

    
if __name__ == "__main__":

    sigmas2 = [0.005, 0.01, 0.015, 0.02, 0.025]
    invalidation_targets = [0.05,0.10,0.15, 0.20, 0.25, 0.30,0.35]
    hidden_widths = [[50]]
    backend = "pytorch"
    methods = ["robust_counterfactuals_random_v2","wachter","wachter_rip"]
    datasets = ["compas", "give_me_some_credit", "adult"]
    models = ["ann"]
    n_cfs = 500 
    n_samples = 500
    for method in methods:
        if method in ['wachter_rip',"robust_counterfactuals_v2","robust_counterfactuals_random_v2"]:
            for model in models:
                for dataset in datasets:
                    if model == "ann":
                        for hidden_width in hidden_widths:
                            for sigma2 in sigmas2:
                                print(f'Generating recourses for sigma2={sigma2}')
                                for invalidation_target in invalidation_targets:
                                    run_experiment(
                                        method,
                                        hidden_width,
                                        dataset,
                                        model,
                                        backend,
                                        sigma2,
                                        invalidation_target,
                                        n_cfs,
                                        n_samples
                                        )
            
                    else:
                        for sigma2 in sigmas2:
                            hidden_width = [0]
                            for invalidation_target in invalidation_targets:
                                run_experiment(
                                    method,
                                    hidden_width,
                                    dataset,
                                    model,
                                    backend,
                                    n_cfs=n_cfs,
                                    n_samples=n_samples,
                                    sigma2=sigma2,
                                    invalidation_target=invalidation_target
                                    )
        
                                
        elif method in ["roar","arar"] :  
            for model in models : 
                for dataset in datasets : 
                    if model == "ann" : 
                        for hidden_width in hidden_widths : 
                            for sigma2 in sigmas2 : 
                                run_experiment(
                                    method,
                                    hidden_width,
                                    dataset,
                                    model,
                                    backend,
                                    n_cfs=n_cfs,
                                    n_samples=n_samples,
                                    sigma2=sigma2,
                                    invalidation_target=0.5)
        
        
        
                    else:
                        hidden_width = [0]
                        for sigma2 in sigmas2:
                            run_experiment(
                                method,
                                hidden_width,
                                dataset,
                                model,
                                backend,
                                n_cfs=n_cfs,
                                n_samples=n_samples,
                                sigma2=sigma2,
                                invalidation_target=0.5
                                )
            
                 
        else :
            for model in models:
                for dataset in datasets:
                    if model == "ann":
                        for hidden_width in hidden_widths:
                            run_experiment(
                                    method,
                                    hidden_width,
                                    dataset,
                                    model,
                                    backend,
                                    n_cfs=n_cfs,
                                    n_samples=n_samples,
                                    sigma2=0.01,
                                    invalidation_target=0.5
                                    )
                    else:
                        hidden_width = [0]
                        run_experiment(
                                method,
                                hidden_width,
                                dataset,
                                model,
                                backend,
                                n_cfs=n_cfs,
                                n_samples=n_samples,
                                sigma2=0.01,
                                invalidation_target=0.5)
                                
                            
                            
