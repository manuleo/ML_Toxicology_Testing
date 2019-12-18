# -*- coding: utf-8 -*-
'''Helper functions for the Matrix Factorization algorithm'''

import pandas as pd
import numpy as np
import itertools

import mxnet as mx
import turicreate as tc

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from importlib import import_module

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


def _get_poss_side():
    '''Get list of possible side features for chemicals and species (see report for more information)
    Inputs:
        - Empty
    Outputs:
        - chemicals (list): list of possible chemicals side features
        - species (list): list of possible species side features'''
    
    # Define variables
    species = ["species", "class", "tax_order", "family", "genus"]
    chemicals = ["test_cas", "alone_atom_number", "tripleBond", "doubleBond", "ring_number"]
    
    return chemicals, species

def build_matrices(final_db):
    '''Convert preprocessed dataset into three dataset usable into the MF algorithm
    Inputs:
        - final_db (Pandas dataframe): cleaned dataframe 
    Outputs:
        - X (Pandas dataframe): dataframe containing scores for the pairs (chemical, species) plus side features of the experiments
        - X_side_species (Pandas dataframe): dataframe with possible side features for the species
        - X_side_cas (Pandas dataframe): dataframe with possible side features for the chemicals
    '''
    # Build X matrix to factorize
    X = final_db.copy()
    X = X[["test_cas", "species", "exposure_type", "conc1_type", "obs_duration_mean", "score"]]\
        .reset_index(drop=True)
        
    # Get possible side features
    chemicals, species = _get_poss_side()
    
    # Build side matrix for species
    X_side_species = final_db.copy()
    X_side_species = X_side_species[species]
    X_side_species = X_side_species.drop_duplicates(subset="species").reset_index(drop=True)
    
    # Build side matrix for chemicals
    X_side_cas = final_db.copy()
    X_side_cas = X_side_cas[chemicals]
    X_side_cas = X_side_cas.drop_duplicates(subset="test_cas").reset_index(drop=True)
    
    return X, X_side_species, X_side_cas


def feature_selection_mf(X_train, X_test, X_species, X_cas, singleclass, seed=13):
    '''Perform online feature selection for the KNN algorithm. Default params are used due for time constraints
    Inputs:
        - X_train, X_test (SFrames): score matrix divide in train and test, with side features for the experiments
        - X_species (SFrame): SFrame with possible side features for the species
        - X_cas (SFrame): SFrame with possible side features for the chemicals
        - singleclass (bool): flag to indicate if perform a binary of multiclass classification
        - seed (int) (fixed): random seed to use
    Output: 
        - best_chem (list): list of best chemicals side features
        - best_species (list): list of best species side features
    '''
    
    # Set seed
    np.random.seed(13)
    
    # Best parameter to search
    best_rmse = np.inf
    best_species = []
    best_chem = []

    # Define array of total possible side features
    chemicals, species = _get_poss_side()
    poss_side = species + chemicals
    
    # Do all the possible combinations of features (from 0 side features to all)
    for k in range(0, 9):
        print("Starting k =",k)
        
        # All combinations with this fixed k
        poss_comb = list(itertools.combinations(range(0,8),k))
        
        for c in poss_comb:
            side_cas = ["test_cas"]
            side_spec = ["species"]
            no_cas = False
            no_spec = False
            
            # Add combination to correct list of side (first 4 are species)
            for i in list(c):
                if i in range(0, 4):
                    side_spec.append(poss_side[i])
                else:
                    side_cas.append(poss_side[i])
            
            # If one of the two dataset is not used in this combination, flag this
            if (len(side_cas)==1):
                no_cas = True
            if (len(side_spec)==1):
                no_spec = True

            # Build side features datasets to test
            X_species_test = X_species.copy()
            X_species_test = X_species_test.select_columns(side_spec)

            X_cas_test = X_cas.copy()
            X_cas_test = X_cas_test.select_columns(side_cas)

            # train different models with respect to the chosen features
            if (no_cas==True and no_spec==True):
                model = tc.recommender.factorization_recommender.create(X_train, user_id='test_cas', item_id='species', target='score', \
                                                                        solver="sgd", max_iterations=1000, verbose=False, random_seed=seed, binary_target=singleclass)
            elif(no_cas==True and no_spec==False):
                model = tc.recommender.factorization_recommender.create(X_train, user_id='test_cas', item_id='species', target='score', \
                                                                        item_data=X_species_test, solver="sgd", max_iterations=1000, verbose=False, random_seed=seed, binary_target=singleclass)
            elif(no_cas==False and no_spec==True):
                model = tc.recommender.factorization_recommender.create(X_train, user_id='test_cas', item_id='species', target='score', \
                                                                        user_data=X_cas_test, solver="sgd", max_iterations=1000, verbose=False, random_seed=seed, binary_target=singleclass)
            else:
                model = tc.recommender.factorization_recommender.create(X_train, user_id='test_cas', item_id='species', target='score', \
                                                                        item_data=X_species_test, user_data=X_cas_test, solver="sgd", max_iterations=1000, verbose=False, random_seed=seed, binary_target=singleclass)
            
            # Compute RMSE and compare with the best one so far
            rmse = model.evaluate_rmse(X_test, 'score')['rmse_overall']
            if rmse<best_rmse:
                best_rmse = rmse
                best_species = side_spec
                best_chem = side_cas
                print("Best combination found! RMSE: {}, species features: {}, chemical features: {}".format(best_rmse, best_species, best_chem))

    return best_chem, best_species

def get_best_features(singleclass):
    '''Get best side features for chemicals and species with respect to the algorithm to used.
    The features are already computed by using online features selection. More information in the report.
    Inputs:
        - singleclass (boolean): flag to indicate if perform a binary of multiclass classification
    Outputs:
        - best_chem (list): list of best chemicals side features
        - best_species (list): list of best species side features
        '''   
    # Define features already found   
    best_chem = ['test_cas', 'tripleBond']
    if (singleclass):
        best_species = ['species', 'class', 'tax_order', 'genus']
    else:
        best_species = ['species', 'class', 'tax_order', 'family', 'genus']
    
    return best_chem, best_species

def cv_fact_class(X, X_cas, X_species, factors, regularizs, lin_regulars, singleclass, cv = 3, verbose=True, seed=13):
    '''Perform Cross Validation on MF algorithm
    Inputs:
        - X (Pandas dataframe): scored matrix with side features for the experiments 
        - X_species (SFrame): SFrame with best side features for the species
        - X_cas (SFrame): SFrame with best side features for the chemicals
        - factors (list): list of latent factors to try
        - regularizs (list): list of regularization parameters (lambda_2 in report) to try
        - lin_regulars (list): list of linear regularization parameters (lambda_1 in report) to try
        - singleclass (boolean): flag to indicate if perform a binary of multiclass classification
        - seed (int): seed to use (fixed)
        - cv (int): number of fold to use in the CV
        - verbose (bool): verbosity level
    Output: 
        - best_lin_regulariz(int): best linear regularization parameter
        - best_regulariz (int): best regularization parameter
        - best_factor (int): best latent factors parameter
    '''
    
    # Define parameters to optimize and save
    best_factor = 0
    best_regulariz = 0
    best_lin_regulariz = 0
    best_acc = 0
    best_rmse = np.inf
    np.random.seed(seed)

    # Grid search on the parameters
    for factor in factors:
        for regulariz in regularizs:
            for lin_regular in lin_regulars:
                
                # Split using KFold
                accs = []
                rmses = []
                kf = KFold(n_splits=cv, shuffle=True)

                for train_ind, test_ind in kf.split(X):
                    
                    # Split train and test data, train model
                    train = tc.SFrame(X.loc[train_ind])
                    test = tc.SFrame(X.loc[test_ind])

                    model = tc.recommender.factorization_recommender.create(train, 'test_cas', 'species', target='score', max_iterations=1000, \
                            num_factors = factor, regularization=regulariz, linear_regularization=lin_regular, solver="sgd", verbose=False, 
                            user_data=X_cas, item_data=X_species, binary_target=singleclass, random_seed=seed)

                    # Find rmse and accuracy for the model
                    X_test_predict = model.predict(test)
                    
                    # Approx real values produced by factorization to integer
                    if (singleclass):
                        X_test_predict = np.where(X_test_predict.to_numpy()>=0.5, 1, 0)
                    else:
                        X_test_predict = np.rint(X_test_predict.to_numpy())
                    
                    # Compute accuracy and rmse
                    acc = accuracy_score(X.loc[test_ind].score.values, X_test_predict)
                    rmse = model.evaluate_rmse(test, 'score')['rmse_overall']
                    accs.append(acc)
                    rmses.append(rmse)

                # Check if RMSE is less than before, save params
                acc_avg = np.mean(accs)
                rmse_avg = np.mean(rmses)

                if (rmse_avg < best_rmse):
                    if verbose:
                        print("New best param combination found! RMSE:{}, Acc:{}, num_factors:{}, regularizer:{}, lin_reg:{}".format(rmse_avg, acc_avg, factor, regulariz, lin_regular))     
                    best_acc = acc_avg
                    best_rmse = rmse_avg
                    best_lin_regulariz = lin_regular
                    best_regulariz = regulariz
                    best_factor = factor

    return best_lin_regulariz, best_regulariz, best_factor

def run_mf(X_train, X_test, X_species, X_cas, lin_regulariz, regulariz, factor, singleclass, seed=13):
    '''Run KNN algorithm and return predictions, RMSE and accuracy.
    Inputs:
        - X_train, X_test (SFrames): score matrix divide in train and test, with side features for the experiments
        - y_train (numpy array): label for train to fit the algorithm
        - X_species (SFrame): SFrame with best side features for the species
        - X_cas (SFrame): SFrame with best side features for the chemicals
        - lin_regulariz(int): linear regularization parameter
        - regulariz (int): regularization parameter
        - factor (int): latent factors parameter
        - singleclass (boolean): flag to indicate if perform a binary of multiclass classification
        - seed (int): seed to use (fixed)
    Output: 
        - y_pred_train (numpy array): predictions done on the train set
        - y_pred_test (numpy array): predictions done on the test set
        - rmse (float): rmse on the test set
        - acc (float): accuracy on the test set
    ''' 
    
    # Train algorithm
    model = tc.recommender.factorization_recommender.create(X_train, user_id='test_cas', item_id='species', target='score', user_data=X_cas, item_data=X_species, solver="sgd", \
                                                            max_iterations=1000, verbose=True, \
                                                            num_factors=factor, regularization=regulariz, linear_regularization=lin_regulariz, \
                                                            binary_target=singleclass, random_seed=seed)    
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    if (singleclass):
        y_pred_test_int = np.where(y_pred_test.to_numpy()>=0.5, 1, 0)
    else:
        y_pred_test_int = np.rint(y_pred_test_int.to_numpy())
        
    # Compute scores
    rmse = model.evaluate_rmse(X_test, 'score')['rmse_overall']
    y_test = X_test.select_columns('score')
    acc = accuracy_score(y_test, y_pred_test_int)
    
    return y_pred_train, y_pred_test