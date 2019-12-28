# -*- coding: utf-8 -*-
'''Helper functions for the KNN algorithm'''

import pandas as pd
import numpy as np
import itertools

from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from scipy.spatial.distance import hamming
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def encode_categories(final_db):
    '''Take needed features from the dataset and encode string into categorical numbers
    Inputs:
        - final_db (Pandas dataframe): cleaned dataframe 
    Outputs:
        - X (Pandas dataframe): feature matrix dimension NxD, where N is the datapoints number and D the number of features
        - y (numpy array): labels array (binary or multiclass), dimension Nx1 
        - enc (sklearn OrdinalEncoder): ordinal encoder used (to be used in decoding after)
    '''
        
    # Loading data
    X = final_db.copy()
    X = X[[
    'exposure_type', 'obs_duration_mean', 'conc1_type', 'species', 'class','tax_order', 'family', 'genus',
    'atom_number', 'alone_atom_number', 'tripleBond','doubleBond', 'bonds_number', 'ring_number', 'Mol', 
    'MorganDensity', 'LogP']]
    y = final_db.score.copy().values
    
    # Encoding phase
    enc = OrdinalEncoder(dtype=int)
    enc.fit(X[['exposure_type', 'conc1_type', 'species', 'class', 'tax_order', 'family', 'genus']])
    X[['exposure_type', 'conc1_type', 'species', 'class', 'tax_order', 'family', 'genus']] = \
        enc.transform(X[['exposure_type', 'conc1_type', 'species', 'class', 'tax_order', 'family', 'genus']])
        
    return X, y, enc
        
def get_features():
    '''Get list of categorical and non categorical variables for KNN algorithm (see report for more information)
    Inputs:
        - Empty
    Outputs:
        - categorical (list): list of categorical features
        - non_categorical (list): list of numerical, non categorical features'''
    
    # Define variables
    categorical = [
    'ring_number', "exposure_type", "conc1_type", "species", 'tripleBond', 'obs_duration_mean',
    'doubleBond', 'alone_atom_number', 'class', 'tax_order', 'family', 'genus']
    
    non_categorical =['atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP']
    
    return categorical, non_categorical


def compute_distance_matrix(X, len_X_train, cat_features = [], num_features = [], alpha = 1):
    '''Compute distance matrix for the KNN algorithm (see report for detail)
    Inputs:
        - X (Pandas dataframe or numpy array-like): complete feature matrix X (shape NxD)
        - len_X_train (int): length of the X_train matrix previously computed (to correctly divide the train from the test)
        - cat_features (list): list of categorical features
        - num_features (list): list of numerical, non categorical features
        - alpha (int): weight for the categorical features loss (see report for detail)
    Output: 
        - dist_matr_train (Pandas dataframe): distance matrix to use for trainining
        - dist_matr_test (Pandas dataframe): distance matrix to use for testing
    '''

    # Select numerical and categorical features
    X_cat = X[cat_features]
    X_num = X[num_features]
    
    # Compute matrix for both categorical and numerical or just one (generalization for feature selection)
    if (len(cat_features)!=0 and len(num_features)):
        dist_matr = alpha * squareform(pdist(X_cat, metric = "hamming"))
        dist_matr += squareform(pdist(X_num, metric = "euclidean"))
    elif(len(cat_features)!=0):
        dist_matr = alpha * squareform(pdist(X_cat, metric = "hamming"))
    else:
        dist_matr = squareform(pdist(X_num, metric = "euclidean"))

    # Extract train and test matrices
    dist_matr_train = dist_matr[:len_X_train,:len_X_train]
    dist_matr_test = dist_matr[len_X_train:,:len_X_train]

    return dist_matr_train, dist_matr_test

def feature_selection_knn(X_train, X_test, y_train, y_test, categorical, non_categorical):
    '''Perform online feature selection for the KNN algorithm. Default params are used due for time constraints
    Inputs:
        - categorical (list): list of categorical features
        - non_categorical (list): list of numerical, non categorical features
        - X_train, X_test (Pandas dataframe): feature matrix splitted in train and test
        - y_train, y_test (numpy array): label for train and test
    Output: 
        - best_cat (list): list of best categorical features
        - best_non_cat (list): list of best non categorical features
    '''
 
    # Best parameter to search
    best_acc = 0
    best_cat = []
    best_non_cat = []

    # Define array of total possible features
    poss_features = np.array(categorical + non_categorical)
    
    # Do all the possible combinations of features (from 1 features to all)
    for k in range(1, 18):
        print("Starting k =",k)
        
        # All combinations with this fixed k
        poss_comb = list(itertools.combinations(range(0,17),k))
        for c in poss_comb:
            cat = []
            non_cat = []
        
        # First 12 features are categorical
        for i in list(c):
            if i in range(0, 12):
                cat.append(poss_features[i])
            else:
                non_cat.append(poss_features[i])

        # Compute distance matrix and KNN
        len_X_train = len(X_train)
        X_train_new, X_test_new = compute_distance_matrix(X_train.append(X_test), len_X_train, cat,non_cat, alpha = 1)
        
        neigh = KNeighborsClassifier(metric = 'precomputed')
        neigh.fit(X_train_new, y_train.ravel())
        y_pred = neigh.predict(X_test_new)
        
        # Find accuracy
        acc = accuracy_score(y_test, y_pred)

        # If improvement, save parameters
        if acc>best_acc:
            best_acc = acc
            best_cat = cat
            best_non_cat = non_cat
            print("Best combination found! Acc: {}, features: cat: {}, non_cat:{}".format(best_acc, best_cat, best_non_cat))
        
        # Clean variables
        del X_train_new, X_test_new
    
    return best_cat, best_non_cat


def cv_knn(X, y, cat_features = [], num_features = [], alphas = [], ks = [], leafs=[], seed=13, cv=3):
    '''Perform Cross Validation on KNN algorithm
    Inputs:
        - X (Pandas dataframe): feature matrix 
        - y (numpy array): label for X, either binary or multiclass
        - cat_features (list): list of categorical features
        - num_features (list): list of numerical, non categorical features
        - alphas (list): list of alpha to try for the distance matrix
        - ks (list): list of Ks to try for the neighbours number of the classifier
        - leafs (list): list of leaf_size to try for the classifier
        - seed (int): seed to use (fixed)
        - cv (int): number of fold to use in the CV
    Output: 
        - best_alpha (int): best alpha parameter
        - best_k (int): best K parameter
        - best_leaf (int): best leaf size parameter
    '''
     
    # Set seed and best initial params
    np.random.seed(seed)
    best_accuracy = 0
    best_alpha = 0
    best_k = 0
    best_leaf = 0
    
    # Compute distance matrix for numerical features (fixed)
    X_cat = X[cat_features]
    X_num = X[num_features]
    dist_matr_num = squareform(pdist(X_num, metric = "euclidean"))

    # Grid search on best params
    for alpha in alphas:
        
        # Compute distance matrix on categorical features (depends on alpha)
        dist_matr = alpha * squareform(pdist(X_cat, metric = "hamming"))
        dist_matr += dist_matr_num
        dist_matr = pd.DataFrame(dist_matr)
        
        for k in ks:
            for leaf in leafs:

                kf = KFold(n_splits=cv, shuffle=True)
                accs = []
                for train_index, test_index in kf.split(dist_matr):
                    
                    # Split in train and test
                    X_train = dist_matr.iloc[train_index, train_index]
                    X_test = dist_matr.iloc[test_index, train_index]
                    y_train = y[train_index]
                    y_test = y[test_index]

                    # KNN on the train, compute accuracy on test
                    neigh = KNeighborsClassifier(metric = 'precomputed', n_neighbors=k, n_jobs=-2, leaf_size=leaf)
                    neigh.fit(X_train, y_train.ravel())
                    y_pred = neigh.predict(X_test)

                    accs.append(accuracy_score(y_test, y_pred))

                # Compute average accuracy and save best parameters if actual best
                avg_acc = np.mean(accs)
                if (avg_acc > best_accuracy):
                    print("New best params found! alpha:{}, k:{}, leaf:{}, acc:{}".format(alpha, k, leaf, avg_acc))
                    best_alpha = alpha
                    best_k = k
                    best_accuracy = avg_acc
                    best_leaf = leaf

    return best_alpha, best_k, best_leaf

        
def run_knn(X_train, y_train, X_test, categorical, non_categorical, alpha, k, leaf_size):
    '''Run KNN algorithm and return predictions.
    Inputs:
        - X_train, X_test (Pandas dataframe): feature matrix splitted in train and test
        - y_train (numpy array): label for train to fit the algorithm
        - categorical (list): list of categorical features
        - non_categorical (list): list of numerical, non categorical features
        - alpha (int): alpha parameter for the distance matrix
        - k (int): K parameter for the KNN
        - leaf (int): leaf size parameter for KNN
    Output: 
        - y_pred_train (numpy array): predictions done on the train set
        - y_pred_test (numpy array): predictions done on the test set
    ''' 
    
    # Compute Distance Matrix
    len_X_train = len(X_train)
    X_train_distance, X_test_distance = compute_distance_matrix(X_train.append(X_test), len_X_train, categorical, non_categorical, alpha)
    
    # Run KNN
    neigh = KNeighborsClassifier(metric = 'precomputed', n_neighbors=k, leaf_size=leaf_size)
    neigh.fit(X_train_distance, y_train.ravel())
    
    # Make predictions
    y_pred_train = neigh.predict(X_train_distance)
    y_pred_test = neigh.predict(X_test_distance)
    
    return y_pred_train, y_pred_test


def decode_categories(X_final_train, X_final_test, enc):
    '''Decode categorical features from numbers to strings
    Inputs:
        - X_final_train (Pandas dataframe): final train dataset with prediction (categories as numbers)
        - X_final_test (Pandas dataframe): final train dataset with prediction (categories as numbers)
    Outputs:
        - X_final_train (Pandas dataframe): final train dataset with prediction (categories as strings)
        - X_final_test (Pandas dataframe): final train dataset with prediction (categories as strings)
    '''
         
    X_final_train[['exposure_type', 'conc1_type', 'species', 'class', 'tax_order', 'family', 'genus']] = \
        enc.inverse_transform(X_final_train[['exposure_type', 'conc1_type', 'species', 'class', 'tax_order', 'family', 'genus']])
        
    X_final_test[['exposure_type', 'conc1_type', 'species', 'class', 'tax_order', 'family', 'genus']] = \
        enc.inverse_transform(X_final_test[['exposure_type', 'conc1_type', 'species', 'class', 'tax_order', 'family', 'genus']])
        
    return X_final_train, X_final_test
    