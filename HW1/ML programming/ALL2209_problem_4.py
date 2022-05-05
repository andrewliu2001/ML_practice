import numpy as np
import pandas as pd

#Load dataset
file_train = 'compas_dataset/propublicaTrain.csv'
file_test = 'compas_dataset/propublicaTest.csv'

df_train = pd.read_csv(file_train)
df_test = pd.read_csv(file_test)

Y_train = np.array(df_train['two_year_recid'])
Y_test = np.array(df_test['two_year_recid'])
X_train = np.array(df_train.iloc[:, df_train.columns != 'two_year_recid'])
X_test = np.array(df_test.iloc[:, df_test.columns != 'two_year_recid'])

n_train = Y_train.shape[0]
n_test = Y_test.shape[0]
d = X_train.shape[1]

#We will be varying training size
def gen_data(percent_train):
    """
    Input: percentage of train data (between 0 and 1)
    Returns: arrays new_X_train, new_Y_train
    """
    new_X_train = X_train[:int(percent_train*n_train)]
    new_Y_train = Y_train[:int(percent_train*n_train)]
    return new_X_train, new_Y_train

"""

Part 1:    MLE based method

"""


def calculate_class_priors(Y_train):
    """
    Input: Y_train ((n_train,1) dimensional array)
    Returns: array of class priors (2-dimensional array)
    """
    class_priors = np.empty(2)
    for i in range(2):
      class_priors[i] = np.count_nonzero(Y_train == i) / Y_train.shape[0]
      
    return class_priors

def calculate_means(X_train, Y_train):
    """
    Input:
        - X_train ((n_train,d) array)
        - Y_train (n_train-dimensional array)
    Returns: mu_MLE (d-dimensional array)
    """
    mu_MLE = np.empty((2, d))
    
    for i in range(2):
        X_filtered = X_train[Y_train==i]
        mu_MLE[i] = np.mean(X_filtered, axis=0)
      
    return mu_MLE

def calculate_cov_det_pinv(X_train, Y_train):
    """
    Input: X_train, Y_train, mu_MLE
    Returns: covariance matrices (2,d,d), offsetted log of determinant of 
    covariance matrices (2,), psuedo-inverse of covariance matrices (2,d,d)
    """
    
    
    covariance_matrices = np.empty((2, d, d))
    log_determinants_of_cov_matrices = np.empty(2)
    pinv_covariance_matrices = np.empty((2,d,d))
    
    for i in range(2):
        X_filtered = X_train[Y_train==i]
        covariance_matrices[i] = np.cov(np.transpose(X_filtered))
        
        log_determinants_of_cov_matrices[i] = np.log(
                np.linalg.det(covariance_matrices[i])+0.001)
        
        pinv_covariance_matrices[i] = np.linalg.pinv(covariance_matrices[i])

    return (covariance_matrices, log_determinants_of_cov_matrices,
            pinv_covariance_matrices)


def MLE_model(x, mu, pinv_cov, log_det, class_prior):   
    """
    Input: 
        - x (d-dimensional)
        - mu (float)
        - pinv_cov (pseudoinverse covariance matrix (d,d))
        - log_det (logarithm of determinant (scalar))
        - class_prior (scalar)
    Returns: log probabilities (float)
    """
    diff = np.reshape(x-mu, (d,1))
    exponent = np.matmul(np.transpose(diff), pinv_cov)
    exponent = -0.5 * np.matmul(exponent, diff)
    return exponent - 0.5*log_det-(d/2)*np.log(2*np.pi) + np.log(class_prior)

def MLE_classifier(x, mu_MLE, pinv_covariance_matrices,
                            log_determinants_of_cov_matrices, class_priors):
    """
    Input:
        - x: d-dimensional array of one test sample
        - mu_MLE: array storing the mean x of each class
        - pinv_covariance_matrices: pseudoinverse of cov matrices
        - log_determinants_of_cov_matrices: offsetted log of determinants
        - class_priors: list of class priors
            
    Returns: classification result (integer from 0 to 9)
    """
    
    log_probabilities = np.empty(2)
    for i in range(2):
        log_probabilities[i] = MLE_model(x, mu_MLE[i],
                                         pinv_covariance_matrices[i],
                                         log_determinants_of_cov_matrices[i],
                                         class_priors[i])
    return np.argmax(log_probabilities)


"""

Part 2: kNN

"""

def kNN(x, k, X_train, Y_train, order=2):
    """
    Input: 
        - x (d-dimensional array)
        - k (number of nearest neighbors)
        - X_train ((n_train, d) array)
        - Y_train (n-dimensional array)
        - order (value of p for p-norm)
    Returns: majority (result of kNN classification)
    """
    index_to_distance = np.empty((X_train.shape[0], 2))
  
    for i in range(X_train.shape[0]):
        index_to_distance[i] = np.array([i, np.linalg.norm(X_train[i]-x, ord=order)])
    
    sorted_array = index_to_distance[np.argsort(index_to_distance[:,1])]
    k_sorted = sorted_array[:k]
 
    labels = Y_train[k_sorted[:,0].astype(int)]
    majority = np.argmax(np.bincount(labels))
    return majority


"""

Part 3: Naive Bayes classifier

"""


def naive_bayes_classifier(x, class_priors): 
    """
    Input: 
        - x (d-dimensional array)
        - class_priors (2-dimensional array)
    Returns: result of naive bayes classification
    """
    conditional_prob = np.empty(2)
    
    for i in range(2):
        X_modified = X_train[Y_train==i]
        
        prob = 1
        for j in range(d):
            prob = prob * np.count_nonzero(X_modified[:, j] == x[j])/X_modified.shape[0]
        prob = prob * class_priors[i]
        
        conditional_prob[i] = prob
    return np.argmax(conditional_prob)


