import matplotlib.pyplot as plt
import scipy.io
import numpy as np

#Load file
file = 'digits'
mat = scipy.io.loadmat(file)

X=mat['X']
Y=mat['Y']

n = X.shape[0]
d = X.shape[1]


def gen_data(percent_train):
    """
    Input: percentage of train data (between 0 and 1)
    Returns: arrays X_train, X_test, Y_train, Y_test
    """
    X_train, X_test = X[:int(percent_train*n)], X[int(percent_train*n):]
    Y_train, Y_test = Y[:int(percent_train*n)], Y[int(percent_train*n):]
    return X_train, X_test, Y_train, Y_test




"""

Part 1: MLE-based method

"""
def calculate_class_priors(Y_train):
    """
    Input: Y_train ((n_train,1) dimensional array)
    Returns: array of class priors (2-dimensional array)
    """
    class_priors = np.empty(10)
    for i in range(0,10):
      class_priors[i] = np.count_nonzero(Y_train == i) / Y_train.shape[0]
      
    return class_priors

def calculate_means(X_train, Y_train):
    """
    Input:
        - X_train ((n_train,d) array)
        - Y_train (n_train-dimensional array)
    Returns: mu_MLE (d-dimensional array)
    """
    mu_MLE = np.empty((10, d))
    
    for i in range(10):
      X_filtered = X_train[np.reshape(Y_train==i, (Y_train.shape[0],))]
      mu_MLE[i] = np.mean(X_filtered, axis=0)
      
    return mu_MLE

def calculate_cov_det_pinv(X_train, Y_train):
    """
    Input: X_train, Y_train, mu_MLE
    Returns: covariance matrices, offsetted log of determinant of covariance
    matrices, psuedo-inverse of covariance matrices.
    """
    
    
    covariance_matrices = np.empty((10, d, d))
    log_determinants_of_cov_matrices = np.empty(10)
    pinv_covariance_matrices = np.empty((10,d,d))
    
    for i in range(10):
        X_filtered = X_train[np.reshape(Y_train==i, (Y_train.shape[0],))]
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
    
    log_probabilities = np.empty(10)
    for i in range(10):
        log_probabilities[i] = MLE_model(x, mu_MLE[i],
                                         pinv_covariance_matrices[i],
                                         log_determinants_of_cov_matrices[i],
                                         class_priors[i])
    return np.argmax(log_probabilities)




"""

Part 2: kNN method

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
    labels = np.reshape(Y_train[k_sorted[:,0].astype(int)], (k,))
    majority = np.argmax(np.bincount(labels))
    return majority


def compare_classifiers():
    
    percentages = np.array([0.5,0.6,0.7,0.8,0.9])
    train_sizes = percentages * n
    accuracy_MLE = []
    accuracy_kNN = []
    
    
    for percent_train in percentages:
        X_train, X_test, Y_train, Y_test = gen_data(percent_train)
        n_test = X_test.shape[0]
        
        #Calculate MLE model parameters
        class_priors = calculate_class_priors(Y_train)
        mu_MLE = calculate_means(X_train, Y_train)
        (covariance_matrices, log_determinants_of_cov_matrices,
        pinv_covariance_matrices) = calculate_cov_det_pinv(X_train, Y_train)
        

        MLE_correct_test = 0
        
        for i in range(n_test):
            if Y_test[i][0] == MLE_classifier(X_test[i], mu_MLE,
                               pinv_covariance_matrices,
                               log_determinants_of_cov_matrices,
                               class_priors):
                MLE_correct_test += 1
        
        print(f'MLE classifier accuracy for percent_train = {percent_train}: {MLE_correct_test/n_test}')
        accuracy_MLE.append(MLE_correct_test/n_test)
        
        kNN_correct_test = 0
        for i in range(n_test):
            if Y_test[i][0] == kNN(X_test[i], 100, X_train, Y_train, order=2):
                kNN_correct_test += 1
        print(f'kNN classifier accuracy for percent_train = {percent_train}: {kNN_correct_test/n_test}')
        accuracy_kNN.append(kNN_correct_test/n_test)
        
    plt.plot(train_sizes, accuracy_MLE, label='MLE accuracy')
    plt.plot(train_sizes, accuracy_kNN, label='kNN accuracy')
    plt.xlabel('Train sample size')
    plt.ylabel('Test accuracy')
    plt.title('MLE vs 100-NN (L2 norm) test accuracy for various training sample sizes')


def compare_kNNs():
    """
    Compares kNN performance across different norms as sample size varies.
    """
    percentages = np.array([0.5,0.6,0.7,0.8,0.9])
    train_sizes = percentages * n
    one_norm_accuracy = []
    two_norm_accuracy = []
    inf_norm_accuracy = []

    for percent_train in percentages:
        X_train, X_test, Y_train, Y_test = gen_data(percent_train)
        
        one_NN_correct_test = 0
        two_NN_correct_test = 0
        inf_NN_correct_test = 0
        for i in range(100):
            if Y_test[i][0] == kNN(X_test[i], 2, X_train, Y_train, order=1):
                one_NN_correct_test += 1
        for i in range(100):
            if Y_test[i][0] == kNN(X_test[i], 2, X_train, Y_train, order=2):
                two_NN_correct_test += 1
        for i in range(100):
            if Y_test[i][0] == kNN(X_test[i], 2, X_train, Y_train, order=np.inf):
                inf_NN_correct_test += 1
                
        print(one_NN_correct_test/100, two_NN_correct_test/100, inf_NN_correct_test/100)
                
        one_norm_accuracy.append(one_NN_correct_test/100)
        two_norm_accuracy.append(two_NN_correct_test/100)
        inf_norm_accuracy.append(inf_NN_correct_test/100)


    plt.plot(train_sizes, one_norm_accuracy, label='1-norm accuracy')
    plt.plot(train_sizes, two_norm_accuracy, label='2-norm accuracy')
    plt.plot(train_sizes, inf_norm_accuracy, label='inf-norm accuracy')
    plt.xlabel('Train sample size')
    plt.ylabel('Test accuracy')
    plt.title('1-norm vs 2-norm vs inf-norm 2-NN test accuracy for various training sample sizes')
    plt.legend()


