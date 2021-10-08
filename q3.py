import numpy as np
import matplotlib.pyplot as plt
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
    class_priors = np.empty(2)
    for i in range(2):
      class_priors[i] = np.count_nonzero(Y_train == i) / Y_train.shape[0]
      
    return class_priors

def calculate_means(X_train, Y_train):
    mu_MLE = np.empty((2, d))
    
    for i in range(2):
        X_filtered = X_train[Y_train==i]
        mu_MLE[i] = np.mean(X_filtered, axis=0)
      
    return mu_MLE

def calculate_cov_det_pinv(X_train, Y_train):
    """
    Input: X_train, Y_train, mu_MLE
    Returns: covariance matrices, offsetted log of determinant of covariance
    matrices, psuedo-inverse of covariance matrices.
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
    Return log probabilities instead of probabilities
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
    conditional_prob = np.empty(2)
    
    for i in range(2):
        X_modified = X_train[Y_train==i]
        
        prob = 1
        for j in range(d):
            prob = prob * np.count_nonzero(X_modified[:, j] == x[j])/X_modified.shape[0]
        prob = prob * class_priors[i]
        
        conditional_prob[i] = prob
    return np.argmax(conditional_prob)
        

"""
Part 4: Compare classifiers
"""


def compare_classifiers():
    
    percentages = np.array([0.5, 0.7, 0.9, 1.0])
    train_sizes = percentages * n_train
    accuracy_MLE = []
    accuracy_naive = []
    accuracy_two_NN_one_norm = []
    accuracy_two_NN_two_norm = []
    accuracy_two_NN_inf_norm = []
    accuracy_ten_NN_one_norm = []
    accuracy_ten_NN_two_norm = []
    accuracy_ten_NN_inf_norm = []
    accuracy_hundred_NN_one_norm = []
    accuracy_hundred_NN_two_norm = []
    accuracy_hundred_NN_inf_norm = []
    
    demographic_parity_diff = {}
    demographic_parity_diff['MLE'] = []
    demographic_parity_diff['naive'] = []
    demographic_parity_diff['L1 2-NN'] = []
    demographic_parity_diff['L2 2-NN'] = []
    demographic_parity_diff['Linf 2-NN'] = []
    demographic_parity_diff['L1 10-NN'] = []
    demographic_parity_diff['L2 10-NN'] = []
    demographic_parity_diff['Linf 10-NN'] = []
    demographic_parity_diff['L1 100-NN'] = []
    demographic_parity_diff['L2 100-NN'] = []
    demographic_parity_diff['Linf 100-NN'] = []
    
    races = X_test[:,2]
    X_test0 = X_test[(races==0)]
    Y_test0 = Y_test[(races==0)]
    X_test1 = X_test[(races==1)]
    Y_test1 = Y_test[(races==1)]
    print(X_test1.shape)
    n_test = X_test.shape[0]
    n_test0 = X_test0.shape[0]
    n_test1 = X_test1.shape[0]
    
    for percent_train in percentages:
        X_train, Y_train = gen_data(percent_train)
        
        test_race0_preds = np.empty(n_test0)
        test_race1_preds = np.empty(n_test1)
        
        #Relevant calculations for MLE
        class_priors = calculate_class_priors(Y_train)
        mu_MLE = calculate_means(X_train, Y_train)
        (covariance_matrices, log_determinants_of_cov_matrices,
        pinv_covariance_matrices) = calculate_cov_det_pinv(X_train, Y_train)
        
        for i in range(n_test0):
            pred = MLE_classifier(X_test0[i], mu_MLE, pinv_covariance_matrices,
                           log_determinants_of_cov_matrices, class_priors)
            test_race0_preds[i] = pred
        
        conditional = Y_test0[test_race0_preds==0]
        m = conditional.shape[0]
        P0_1 = np.sum(conditional)/m

        for i in range(n_test1):
            pred = MLE_classifier(X_test1[i], mu_MLE, pinv_covariance_matrices,
                           log_determinants_of_cov_matrices, class_priors)
            test_race1_preds[i] = pred
        conditional = Y_test1[test_race1_preds==0]
        m = conditional.shape[0]
        P1_1 = np.sum(conditional)/m
        demographic_parity_diff['MLE'].append(np.abs(P0_1-P1_1))
        
        
        
        for i in range(n_test0):
            pred = naive_bayes_classifier(X_test0[i], class_priors)
            test_race0_preds[i] = pred
        conditional = Y_test0[test_race0_preds==0]
        m = conditional.shape[0]
        P0_1 = np.sum(conditional)/m
        
        for i in range(n_test1):
            pred = naive_bayes_classifier(X_test1[i], class_priors)
            test_race1_preds[i] = pred
        conditional = Y_test1[test_race1_preds==0]
        m = conditional.shape[0]
        P1_1 = np.sum(conditional)/m
        
        demographic_parity_diff['naive'].append(np.abs(P0_1-P1_1))
        
        
        
        
        for i in range(n_test0):
            pred = kNN(X_test0[i], 2, X_train, Y_train, 1)
            test_race0_preds[i] = pred
        conditional = Y_test0[test_race0_preds==0]
        m = conditional.shape[0]
        P0_1 = np.sum(conditional)/m
        
        for i in range(n_test1):
            pred = kNN(X_test1[i], 2, X_train, Y_train, 1)
            test_race1_preds[i] = pred
        conditional = Y_test1[test_race1_preds==0]
        m = conditional.shape[0]
        P1_1 = np.sum(conditional)/m
        
        demographic_parity_diff['L1 2-NN'].append(np.abs(P0_1-P1_1))
        
        
        for i in range(n_test0):
            pred = kNN(X_test0[i], 2, X_train, Y_train, 2)
            test_race0_preds[i] = pred
        conditional = Y_test0[test_race0_preds==0]
        m = conditional.shape[0]
        P0_1 = np.sum(conditional)/m
        
        for i in range(n_test1):
            pred = kNN(X_test1[i], 2, X_train, Y_train, 2)
            test_race1_preds[i] = pred
        conditional = Y_test1[test_race1_preds==0]
        m = conditional.shape[0]
        P1_1 = np.sum(conditional)/m
        
        demographic_parity_diff['L2 2-NN'].append(np.abs(P0_1-P1_1))
        

        
        
        
        for i in range(n_test0):
            pred = kNN(X_test0[i], 2, X_train, Y_train, np.inf)
            test_race0_preds[i] = pred
        conditional = Y_test0[test_race0_preds==0]
        m = conditional.shape[0]
        P0_1 = np.sum(conditional)/m
        
        for i in range(n_test1):
            pred = kNN(X_test1[i], 2, X_train, Y_train, np.inf)
            test_race1_preds[i] = pred
        conditional = Y_test1[test_race1_preds==0]
        m = conditional.shape[0]
        P1_1 = np.sum(conditional)/m
        
        demographic_parity_diff['Linf 2-NN'].append(np.abs(P0_1-P1_1))
        
        
        for i in range(n_test0):
            pred = kNN(X_test0[i], 10, X_train, Y_train, 1)
            test_race0_preds[i] = pred
        conditional = Y_test0[test_race0_preds==0]
        m = conditional.shape[0]
        P0_1 = np.sum(conditional)/m
        
        for i in range(n_test1):
            pred = kNN(X_test1[i], 10, X_train, Y_train, 1)
            test_race1_preds[i] = pred
        conditional = Y_test1[test_race1_preds==0]
        m = conditional.shape[0]
        P1_1 = np.sum(conditional)/m
        
        demographic_parity_diff['L1 10-NN'].append(np.abs(P0_1-P1_1))
        
        
        for i in range(n_test0):
            pred = kNN(X_test0[i], 10, X_train, Y_train, 2)
            test_race0_preds[i] = pred
        conditional = Y_test0[test_race0_preds==0]
        m = conditional.shape[0]
        P0_1 = np.sum(conditional)/m
        
        for i in range(n_test1):
            pred = kNN(X_test1[i], 10, X_train, Y_train, 2)
            test_race1_preds[i] = pred
        conditional = Y_test1[test_race1_preds==0]
        m = conditional.shape[0]
        P1_1 = np.sum(conditional)/m
        
        demographic_parity_diff['L2 10-NN'].append(np.abs(P0_1-P1_1))
        
        
        for i in range(n_test0):
            pred = kNN(X_test0[i], 10, X_train, Y_train, np.inf)
            test_race0_preds[i] = pred
        conditional = Y_test0[test_race0_preds==0]
        m = conditional.shape[0]
        P0_1 = np.sum(conditional)/m
        
        for i in range(n_test1):
            pred = kNN(X_test1[i], 10, X_train, Y_train, np.inf)
            test_race1_preds[i] = pred
        conditional = Y_test1[test_race1_preds==0]
        m = conditional.shape[0]
        P1_1 = np.sum(conditional)/m
        
        demographic_parity_diff['Linf 10-NN'].append(np.abs(P0_1-P1_1))
        
        
        for i in range(n_test0):
            pred = kNN(X_test0[i], 100, X_train, Y_train, 1)
            test_race0_preds[i] = pred
        conditional = Y_test0[test_race0_preds==0]
        m = conditional.shape[0]
        P0_1 = np.sum(conditional)/m
        
        for i in range(n_test1):
            pred = kNN(X_test1[i], 100, X_train, Y_train, 1)
            test_race1_preds[i] = pred
        conditional = Y_test1[test_race1_preds==0]
        m = conditional.shape[0]
        P1_1 = np.sum(conditional)/m
        
        demographic_parity_diff['L1 100-NN'].append(np.abs(P0_1-P1_1))
        
        
        for i in range(n_test0):
            pred = kNN(X_test0[i], 100, X_train, Y_train, 2)
            test_race0_preds[i] = pred
        conditional = Y_test0[test_race0_preds==0]
        m = conditional.shape[0]
        P0_1 = np.sum(conditional)/m
        
        for i in range(n_test1):
            pred = kNN(X_test1[i], 100, X_train, Y_train, 2)
            test_race1_preds[i] = pred
        conditional = Y_test1[test_race1_preds==0]
        m = conditional.shape[0]
        P1_1 = np.sum(conditional)/m
        
        demographic_parity_diff['L2 100-NN'].append(np.abs(P0_1-P1_1))
        
        
        for i in range(n_test0):
            pred = kNN(X_test0[i], 100, X_train, Y_train, np.inf)
            test_race0_preds[i] = pred
        conditional = Y_test0[test_race0_preds==0]
        m = conditional.shape[0]
        P0_1 = np.sum(conditional)/m
        
        for i in range(n_test1):
            pred = kNN(X_test1[i], 100, X_train, Y_train, np.inf)
            test_race1_preds[i] = pred
        conditional = Y_test1[test_race1_preds==0]
        m = conditional.shape[0]
        P1_1 = np.sum(conditional)/m
        
        demographic_parity_diff['Linf 100-NN'].append(np.abs(P0_1-P1_1))
        
        
        
    
    plt.title('Absolute demographic parity discrepancy for classifiers')
    plt.plot(train_sizes, demographic_parity_diff['MLE'], label='MLE', c='r')
    plt.plot(train_sizes, demographic_parity_diff['naive'], label='Naive bayes', c='g')
    plt.plot(train_sizes, demographic_parity_diff['L1 2-NN'], label='L1 2-NN', c='b')
    plt.plot(train_sizes, demographic_parity_diff['L2 2-NN'], label='L2 2-NN', c='k')
    plt.plot(train_sizes, demographic_parity_diff['Linf 2-NN'], label='Linf 2-NN', c='cyan')
    plt.plot(train_sizes, demographic_parity_diff['L1 10-NN'], 'k--', label='L1 10-NN')
    plt.plot(train_sizes, demographic_parity_diff['L2 10-NN'], 'b--', label='L2 10-NN')
    plt.plot(train_sizes, demographic_parity_diff['Linf 10-NN'], 'r--', label='Linf 10-NN')
    plt.plot(train_sizes, demographic_parity_diff['L1 100-NN'], 'g--', label='L1 100-NN')
    plt.plot(train_sizes, demographic_parity_diff['L2 100-NN'], 'c--', label='L2 100-NN')
    plt.plot(train_sizes, demographic_parity_diff['Linf 100-NN'], 'm--', label='Linf 100-NN')
    plt.xlabel('Training size')
    plt.ylabel('Absolute demographic parity discrepancy')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    
compare_classifiers()