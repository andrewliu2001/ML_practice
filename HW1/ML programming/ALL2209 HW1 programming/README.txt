References/libraries used:
1. Numpy
2. Matplotlib
3. Pandas
4. Scipy.io

Notes:
I removed the code for generating plots, as the TA said this was not required.






Instructions for running ALL2209_problem_4.py and ALL2209_problem_5.py:


You are given X_train, X_test, Y_train, Y_test


For MLE classifier:
1. Calculate the class priors by running: class_priors = calculate_class_priors(Y_train)
2. Calculate means by running: mu_MLE = calculate_means(X_train, Y_train)
3. Calculate covariance matrices, determinants, and pseudo-inverses by running: (covariance_matrices, log_determinants_of_cov_matrices, pinv_covariance_matrices) = calculate_cov_det_pinv(X_train, Y_train)
4. Run the MLE classifier on the test data using a for loop:

for i in range(n_test):
	classification_result = MLE_classifier(X_test[i], mu_MLE, pinv_covariance_matrices, log_determinants_of_cov_matrices, class_priors)
	#Do whatever with classification_result



For kNN classifier:
1. Choose a value for k (number of nearest neighbors) and order (value of p for p-norm).

k=2
order=np.inf

2. imply run the kNN classifier on test data:

for i in range(n_test):
	classification_result = kNN(X_test[i], k, X_train, Y_train, order)
	#Do whatever with classification_result




For Naive Bayes classifier: (only available for ALL2209_problem_4.py)
1. Calculate class_priors
class_priors = calculate_class_priors(Y_train)

2. Run the Naive Bayes classifier on test data:

for i in range(n_test):
	classification_result = naive_bayes_classifier(X_test[i], class_priors) 
	#Do whatever with classification_result