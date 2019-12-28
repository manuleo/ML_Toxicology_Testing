import argparse
import pandas as pd
import numpy as np

import src.general_helper as genH
from src.knn.main_knn import knn_algorithm
from src.mf.main_mf import mf_algorithm
from src.preprocessing.main_preprocessing import load_and_process_data


def main(preproc, knn, mf, singleclass, multiclass, feat_sel, cross):
    
    # Set random seed
    seed = 13
    np.random.seed(seed)
    
    # Defining paths for the data
    DATA_RESULTS_PATH = "data/raw/results.txt"
    DATA_TEST_PATH = "data/raw/tests.txt"
    DATA_SPECIES_PATH = "data/raw/species.txt"
    
    DATA_CAS_TO_SMILES_PATH = "data/processed/cas_to_smiles.csv"
    DATA_PREPROC_PATH = "data/processed/final_db_processed.csv"
    
    KNN_SAVE_TRAIN_PATH = "output/knn/knn_train_pred"
    KNN_SAVE_TEST_PATH = "output/knn/knn_test_pred"
    
    MF_SAVE_TRAIN_PATH = "output/mf/mf_train_pred"
    MF_SAVE_TEST_PATH = "output/mf/mf_test_pred"
    
    
    # Data loading
    if (preproc):
        # Cleaning data
        print("Starting cleaning process...")
        final_db = load_and_process_data(DATA_RESULTS_PATH, DATA_TEST_PATH, DATA_SPECIES_PATH, DATA_CAS_TO_SMILES_PATH, DATA_PREPROC_PATH)
        
        # Saving dataset
        final_db.to_csv(DATA_PREPROC_PATH, index=False)
        print()
        
    else:
        # Loading preprocessed db
        print("Loading already processed dataset")
        final_db = pd.read_csv(DATA_PREPROC_PATH)
        print()
    
    # Assign score based on the problem to be solved
    if (singleclass):
        print("Binary classification algorithm chosen. Evaulating scores..\n")
        final_db = genH.binary_score(final_db)
    elif (multiclass):
        print("Multi classification algorithm chosen. Evaulating scores..\n")
        final_db = genH.multi_score(final_db)
    else:
        raise Exception('Flag problem with single/multiclass classification')
    
    if (knn):
        print("KNN algorithm chosen!\n")
        
        X_final_train, X_final_test = knn_algorithm(final_db, feat_sel, cross, singleclass, seed)
        
        # Saving predictions
        if (singleclass):
            save_train = KNN_SAVE_TRAIN_PATH + "_binary.csv"
            save_test = KNN_SAVE_TEST_PATH + "_binary.csv"
        else:
            save_train = KNN_SAVE_TRAIN_PATH + "_multi.csv"
            save_test = KNN_SAVE_TEST_PATH + "_multi.csv"  
        
        print("Saving train predicted dataset at {}".format(save_train))
        print("Saving test predicted dataset at {}".format(save_test))
        
        X_final_train.to_csv(save_train, index=False)
        X_final_test.to_csv(save_test, index=False)
        print("Dataset saved!\n")
        
    elif(mf):
        print("MF algorithm chosen!\n")
        
        X_final_train, X_final_test = mf_algorithm(final_db, feat_sel, cross, singleclass, seed)
        
        # Saving predictions
        if (singleclass):
            save_train = MF_SAVE_TRAIN_PATH + "_binary.csv"
            save_test = MF_SAVE_TEST_PATH + "_binary.csv"
        else:
            save_train = MF_SAVE_TRAIN_PATH + "_multi.csv"
            save_test = MF_SAVE_TEST_PATH + "_multi.csv"  
        
        print("Saving train predicted dataset at {}".format(save_train))
        print("Saving test predicted dataset at {}".format(save_test))
        
        X_final_train.to_csv(save_train, index=False)
        X_final_test.to_csv(save_test, index=False)
        print("Dataset saved!\n")
    
    else:
        raise Exception('Flag problem with algorithm selection!')
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the ML Toxicity Prediction implementation.')
    
    # Loading algorithm params
    parser.add_argument('-preproc', action='store_true', help="Execute all the preprocessing steps from the raw dataset. If not set, an already preprocessed dataset will be loaded.")
    parser.add_argument('-feature_sel', action='store_true', help="Run feature selection on the chosen algorithm. If not set, already selected features will be used.")
    parser.add_argument('-cross', action='store_true', help="Run Cross Validation on the chosen algorithm. If not set, already selected best parameters will be used.")

    title_algo = parser.add_argument_group('Possible algorithms', 'Select one of the two possible algorithms')
    group_algo = title_algo.add_mutually_exclusive_group(required=True)
    group_algo.add_argument('-knn', action='store_true', help="Run KNN algorithm")
    group_algo.add_argument('-mf', action='store_true', help="Run Matrix Factorization algorithm")
    
    title_class = parser.add_argument_group('Classification type', 'Select one of the two possible classification')
    group_class = title_class.add_mutually_exclusive_group(required=True)
    group_class.add_argument('-singleclass', action='store_true', help="Execute single class classification")
    group_class.add_argument('-multiclass', action='store_true', help="Execute multi class classification")
    
    args = parser.parse_args()
    main(args.preproc, args.knn, args.mf, args.singleclass, args.multiclass, args.feature_sel, args.cross)