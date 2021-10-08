       """
       MLE_correct_test = 0
       for i in range(n_test):
           if Y_test[i] == MLE_classifier(X_test[i], mu_MLE,
                              pinv_covariance_matrices,
                              log_determinants_of_cov_matrices,
                              class_priors):
               MLE_correct_test += 1
       print(f'MLE classifier accuracy for percent_train = {percent_train}: {MLE_correct_test/n_test}')
       accuracy_MLE.append(MLE_correct_test/n_test)
       """
       naive_correct_test = 0
       for i in range(n_test):
           if Y_test[i] == naive_bayes_classifier(X_test[i], class_priors):
               naive_correct_test += 1
       print(f'Naive Bayes classifier accuracy for percent_train = {percent_train}: {naive_correct_test/n_test}')
       accuracy_naive.append(naive_correct_test/n_test)
       
       
       """
       two_NN_one_norm_correct_test = 0
       for i in range(n_test):
           if Y_test[i] == kNN(X_test[i], 2, X_train, Y_train, order=1):
               two_NN_one_norm_correct_test += 1
       print(f'2-NN L1 classifier accuracy for percent_train = {percent_train}: {two_NN_one_norm_correct_test/n_test}')
       accuracy_two_NN_one_norm.append(two_NN_one_norm_correct_test/n_test)
       
       two_NN_two_norm_correct_test = 0
       for i in range(n_test):
           if Y_test[i] == kNN(X_test[i], 2, X_train, Y_train, order=2):
               two_NN_two_norm_correct_test += 1
       print(f'2-NN L2 classifier accuracy for percent_train = {percent_train}: {two_NN_two_norm_correct_test/n_test}')
       accuracy_two_NN_two_norm.append(two_NN_two_norm_correct_test/n_test) 
       
       two_NN_inf_norm_correct_test = 0
       for i in range(n_test):
           if Y_test[i] == kNN(X_test[i], 2, X_train, Y_train, order=np.inf):
               two_NN_inf_norm_correct_test += 1
       print(f'2-NN L-inf classifier accuracy for percent_train = {percent_train}: {two_NN_inf_norm_correct_test/n_test}')
       accuracy_two_NN_inf_norm.append(two_NN_inf_norm_correct_test/n_test) 
   
   
   
   
       ten_NN_one_norm_correct_test = 0
       for i in range(n_test):
           if Y_test[i] == kNN(X_test[i], 10, X_train, Y_train, order=1):
               ten_NN_one_norm_correct_test += 1
       print(f'10-NN L1 classifier accuracy for percent_train = {percent_train}: {ten_NN_one_norm_correct_test/n_test}')
       accuracy_ten_NN_one_norm.append(ten_NN_one_norm_correct_test/n_test)
       
       ten_NN_two_norm_correct_test = 0
       for i in range(n_test):
           if Y_test[i] == kNN(X_test[i], 10, X_train, Y_train, order=2):
               ten_NN_two_norm_correct_test += 1
       print(f'10-NN L2 classifier accuracy for percent_train = {percent_train}: {ten_NN_two_norm_correct_test/n_test}')
       accuracy_ten_NN_two_norm.append(ten_NN_two_norm_correct_test/n_test) 
       
       ten_NN_inf_norm_correct_test = 0
       for i in range(n_test):
           if Y_test[i] == kNN(X_test[i], 10, X_train, Y_train, order=np.inf):
               ten_NN_inf_norm_correct_test += 1
       print(f'10-NN L-inf classifier accuracy for percent_train = {percent_train}: {ten_NN_inf_norm_correct_test/n_test}')
       accuracy_ten_NN_inf_norm.append(ten_NN_inf_norm_correct_test/n_test) 
   """
   """
       hundred_NN_one_norm_correct_test = 0
       for i in range(n_test):
           if Y_test[i] == kNN(X_test[i], 100, X_train, Y_train, order=1):
               hundred_NN_one_norm_correct_test += 1
       print(f'100-NN L1 classifier accuracy for percent_train = {percent_train}: {hundred_NN_one_norm_correct_test/n_test}')
       accuracy_hundred_NN_one_norm.append(hundred_NN_one_norm_correct_test/n_test)
       
       hundred_NN_two_norm_correct_test = 0
       for i in range(n_test):
           if Y_test[i] == kNN(X_test[i], 100, X_train, Y_train, order=2):
               hundred_NN_two_norm_correct_test += 1
       print(f'100-NN L2 classifier accuracy for percent_train = {percent_train}: {hundred_NN_two_norm_correct_test/n_test}')
       accuracy_hundred_NN_two_norm.append(hundred_NN_two_norm_correct_test/n_test) 
       
       hundred_NN_inf_norm_correct_test = 0
       for i in range(n_test):
           if Y_test[i] == kNN(X_test[i], 100, X_train, Y_train, order=np.inf):
               hundred_NN_inf_norm_correct_test += 1
       print(f'100-NN L-inf classifier accuracy for percent_train = {percent_train}: {hundred_NN_inf_norm_correct_test/n_test}')
       accuracy_hundred_NN_inf_norm.append(hundred_NN_inf_norm_correct_test/n_test) 
   """
   print(accuracy_naive)
   

   plt.plot(train_sizes, accuracy_MLE, label='MLE accuracy')
   plt.plot(train_sizes, accuracy_naive, label='Naive Bayes accuracy')
   plt.plot(train_sizes, accuracy_two_NN_one_norm, label='2-NN L1 accuracy')
   plt.plot(train_sizes, accuracy_two_NN_two_norm, label='2-NN L2 accuracy')
   plt.plot(train_sizes, accuracy_two_NN_inf_norm, label='2-NN L-inf accuracy')
   plt.plot(train_sizes, accuracy_ten_NN_one_norm, label='10-NN L1 accuracy')
   plt.plot(train_sizes, accuracy_ten_NN_two_norm, label='10-NN L2 accuracy')
   plt.plot(train_sizes, accuracy_ten_NN_inf_norm, label='10-NN L-inf accuracy')
   plt.xlabel('Train sample size')
   plt.ylabel('Test accuracy')
   plt.title('Comparison of test accuracy for various training sample sizes')
   plt.legend()