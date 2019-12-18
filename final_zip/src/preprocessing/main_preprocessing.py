# -*- coding: utf-8 -*-
'''Main function to run preprocessing'''

import pandas as pd
import numpy as np

from importlib import import_module


def load_and_process_data(DATA_RESULTS_PATH, DATA_TEST_PATH, DATA_SPECIES_PATH, DATA_CAS_TO_SMILES_PATH, DATA_PREPROC_PATH):
    '''Load data and compute all the necessary preprocessing on raw data.
    Inputs:
            - DATA_RESULTS_PATH (string): path to the results.txt dataset.
            - DATA_TEST_PATH (string): path to the test.txt dataset.
            - DATA_SPECIES_PATH (string): path to the species.txt dataset.
            - DATA_CAS_TO_SMILES_PATH (string): path to .csv file containing information to match the CAS number to the SMILES
            - DATA_PREPROC_PATH (string): path in which save the processed dataset
    Outputs:
        - final_db (Pandas dataframe): preprocessed dataset to be used in the ML implementations.'''
           
    # Import needed module
    hp = import_module("src.preprocessing.helper_preprocessing")
        
    # Loading datasets
    print("Loading original datasets (this may take a while) ....\n")
    results, test, species = hp.load_original_data(DATA_RESULTS_PATH, DATA_TEST_PATH, DATA_SPECIES_PATH)
    
    # Prefilter on LC50, Mortality, Fish
    print("Prefiltering phase...")
    results_prefilter = hp.prefilter(results, test, species)
    print("Data prefiltered! Dataset dimension: {} datapoints x {} possible features\n"\
        .format(results_prefilter.index.size, results_prefilter.columns.size))
    
    # Filtering on the final dataset from 
    print("Filtering on best features for the dataset...")
    results_best_col = hp.filter_best_features(results_prefilter)
    print("Features selected and imputed! Dataset dimension:{} datapoints x {} features\n"\
        .format(results_best_col.index.size, results_best_col.columns.size - 1)) # not considering concentration (label)
 
    # Add SMILES features using CAS
    print("Adding new features using SMILES from CAS...")
    base_db = hp.add_smiles_features(results_best_col, DATA_CAS_TO_SMILES_PATH)
    print("SMILES features added! Dataset dimension: {} datapoints x {} features\n"\
        .format(base_db.index.size, base_db.columns.size - 2)) # no concentration and cas
    
    # Process SMILES features (numerical)
    print ("Preprocessing numerical features...")
    base_db_processed = hp.process_smiles_features(base_db)
    print ("Numerical features processed!\n")
    
    # Merging repeated experiments
    print("Starting repeated experiments analysis")
    final_db = hp.repeated_exp(base_db_processed)
    print("Repeated Experiment merged! Final dataset dimension: {} datapoints x {} features"\
        .format(final_db.index.size, final_db.columns.size-2)) # no concentration and cas
    
    return final_db
