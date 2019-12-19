# -*- coding: utf-8 -*-
'''Matrix Factorization algorithm implementation'''

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from math import sqrt

from src.general_helper import split_dataset, build_final_datasets
from importlib import import_module

def mf_algorithm(final_db, feat_sel, cross, singleclass, seed=13):
    '''Main function to use the KNN algorithm.
    Inputs:
        - final_db (Pandas dataframe): complete db after preprocessing
        - feat_sel (boolean): flag to indicate if online feature selection must be run
        - cross (boolean): flag to indicate if cross validation must be run
        - singleclass (boolean): flag to indicate if using binary or multiclass classification
        - seed (int) (fixed): random seed
    Output: 
        - X_final_train (Pandas dataframe): final train dataset with prediction
        - X_final_test (Pandas dataframe): final train dataset with prediction
    ''' 
    # Import modules
    tc = import_module("turicreate")
    mx = import_module("mxnet")
    mH = import_module("src.mf.helper_mf")
        
    # Set seed 
    np.random.seed(13)
    
    # Build score matrix + datasets containing side features
    print("Building initial matrix and side features for the MF algorithm...\n")
    X, X_side_species, X_side_cas = mH.build_matrices(final_db)
    
    # Split dataset in train and test and convert in SFrame (dataframe used by turicreate)
    print("Splitting dataset in train and test...\n")
    X_train, X_test = split_dataset(X, [], seed)
    
    # Convert in SFrame
    X_train = tc.SFrame(X_train)
    X_test = tc.SFrame(X_test)
    X_species = tc.SFrame(X_side_species)
    X_cas = tc.SFrame(X_side_cas)
    
    # Run feature selection if requested
    if (feat_sel):
        print("Starting feature selection for MF algorithm. ATTENTION: THIS REQUIRES ABOUT HALF AN HOUR.")
        best_chem, best_species = mH.feature_selection_mf(X_train, X_test, X_species, X_cas, singleclass, seed)
    else:
        print("Skipping feature selection for KNN. Best features loaded.")
        best_chem, best_species = mH.get_best_features(singleclass)
    
    print("Chemicals best features: {}\nSpecies best features: {}\n".format(best_chem, best_species))
    
    X_species = X_species.select_columns(best_species)
    X_cas = X_cas.select_columns(best_chem)

    # Run cross validation if requested
    if (cross):
        print("Starting CV algorithm. ATTENTION: THIS MAY REQUIRE 16 HOURS TO COMPLETE")
        factors = list(range(5, 12, 1))
        regularizs = np.logspace(-12, -1, 30)
        lin_regulars = np.logspace(-12, -1, 30)
        best_lin_regulariz, best_regulariz, best_factor = mH.cv_fact_class(X_train.to_dataframe(), X_cas, X_species, factors, regularizs, lin_regulars, singleclass, cv = 3, verbose=True, seed=seed)
    else:
        print("CV skipped. Best parameters loaded\n")
        if (singleclass):
            best_lin_regulariz = 0.0007847599703514623
            best_regulariz = 6.158482110660267e-08
            best_factor = 7
        else:
            best_lin_regulariz = 1.12883789168e-10
            best_regulariz = 0.000784759970351
            best_factor = 10
    print("Best parameters for CV:\n\t-factors: {}\n\t-lambda_1: {}\n\t-lambda_2: {}\n".format(best_factor, best_lin_regulariz, best_regulariz))
    
    # Run MF algorithm
    print("Run MF algorithm with best parameters...")
    y_pred_train, y_pred_test, acc, rmse = mH.run_mf(X_train, X_test, X_species, X_cas, best_lin_regulariz, best_regulariz, best_factor, singleclass, seed)
    print("Prediction achieved!\n")
    
    print("MF RMSE: {}".format(rmse))
    print("MF accuracy: {}\n".format(acc))
    
    # Build final dataset
    print("Building final dataset for future uses...\n")
    X_final_train, X_final_test = build_final_datasets(X_train.to_dataframe(), X_test.to_dataframe(), y_pred_train, y_pred_test)
     
    return X_final_train, X_final_test
    