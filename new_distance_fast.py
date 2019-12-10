import numpy as np
from scipy.spatial.distance import hamming
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def new_distance_matrix(X, len_X_train, cat_features = [], num_features = [], alpha = 1):
    ''' inputs: matrix X [num_samples, num_features], the list of the categorical features, the list of the numerical features, weight alpha
        output: distance matrix
    '''

    # Training
    X_cat = X[cat_features]
    X_num = X[num_features]
    dist_matr = alpha * squareform(pdist(X_cat, metric = "hamming")) + squareform(pdist(X_num, metric = "euclidean"))

    dist_matr_train = dist_matr[:len_X_train,:len_X_train]
    dist_matr_test = dist_matr[len_X_train:,:len_X_train]

    return dist_matr_train, dist_matr_test
