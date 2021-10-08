import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def calculate_class_priors(Y_train):
    class_priors = np.empty(2)
    for i in range(2):
      class_priors[i] = np.count_nonzero(Y_train == i) / Y_train.shape[0]
      
    return class_priors


class_priors = calculate_class_priors(Y_train)


def naive_bayes_classifier(x): 
    conditional_prob = np.empty(2)
    
    for i in range(2):
        X_modified = X_train[Y_train==i]
        
        prob = 1
        for j in range(d):
            prob = prob * np.count_nonzero(X_modified[:, j] == x[j])/X_modified.shape[0]
        prob = prob * class_priors[i]
        
        conditional_prob[i] = prob
    return np.argmax(conditional_prob)

correct= 0
for i in range(n_test):
    if Y_test[i] == naive_bayes_classifier(X_test[i]):
        correct += 1
    
print(correct/n_test)
    