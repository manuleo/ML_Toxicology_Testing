import numpy as np
from scipy.spatial.distance import hamming

def new_distance_matrix(X_train, X_test = False, cat_features = [], num_features = [], alpha = 1):
    ''' inputs: matrix X [num_samples, num_features], the list of the categorical features, the list of the numerical features, weight alpha
        output: distance matrix
    '''

    # Training
    X_cat_train = X_train[cat_features]
    X_num_train = X_train[num_features]
    dist_matr_train = np.zeros((X_train.shape[0],X_train.shape[0]))

    for i in range(len(X_train)):
        for j in range(i):
            dij1 = d1(X_cat_train.iloc[i].values,X_cat_train.iloc[j].values)
            dij2 = d2(X_num_train.iloc[i].values,X_num_train.iloc[j].values)
            dij = new_distance(dij1,dij2,alpha)
            dist_matr_train[i,j] = dij
            dist_matr_train[j,i] = dij
    

    # Test
    X_cat_test = X_test[cat_features]
    X_num_test = X_test[num_features]
    dist_matr_test = np.zeros((X_test.shape[0],X_train.shape[0]))

    for i in range(len(X_test)):
        for j in range(len(X_train)):
            dij1 = d1(X_cat_test.iloc[i].values,X_cat_train.iloc[j].values)
            dij2 = d2(X_num_test.iloc[i].values,X_num_train.iloc[j].values)
            dij = new_distance(dij1,dij2,alpha)
            dist_matr_test[i,j] = dij

    return dist_matr_train, dist_matr_test
    

def d1(x,y):
    '''return the humming distance
    '''
    return hamming(x,y)

def d2(x,y):
    '''return the euclidean distance
    '''
    return np.linalg.norm(x - y)

def new_distance(d1,d2, alpha):
    '''  returns the new defined distance between vector x and y
        alpha * d1 + d2
        where d1 is the humming distance between the categorical features and d2 is the euclidean distance between the numerical ones
    '''
    return alpha * d1 + d2