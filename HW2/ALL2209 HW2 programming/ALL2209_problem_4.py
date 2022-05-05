import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt

"""
Load and reshape data
"""
X_train = np.load('train.npy')
Y_train = np.load('trainlabels.npy')
X_test = np.load('test.npy')
Y_test = np.load('testlabels.npy')
n_train = X_train.shape[0]
d = X_train.shape[1]
n_test = X_test.shape[0]

X_train = np.reshape(X_train, (n_train, d*d))
X_test = np.reshape(X_test, (n_test, d*d))

"""
Decision tree classifier (problem 4ii, 4iii)
"""
max_num_leaves = [50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
DT_classifiers = []
for max_n_leaves in max_num_leaves:
    DT_classifiers.append(DecisionTreeClassifier(max_leaf_nodes=max_n_leaves))
  
for clf in DT_classifiers:
    clf.fit(X_train, Y_train)

test_losses = []
train_losses = []
for clf in DT_classifiers:
    test_losses.append(zero_one_loss(Y_test, clf.predict(X_test)))
    train_losses.append(zero_one_loss(Y_train, clf.predict(X_train)))
  
plt.plot(np.log(max_num_leaves), train_losses, label='Train loss')
plt.plot(np.log(max_num_leaves), test_losses, label='Test loss')
plt.legend()
plt.xlabel('Log of max_num_leaves')
plt.ylabel('Zero-one loss')
plt.title('Zero-one loss vs maximum permitted number of leaves plot for decision tree classifiers')

"""
Random Forest classifier (problem 4v.a)
"""
RF_classifiers = []
for max_n_leaves in max_num_leaves:
    RF_classifiers.append(RandomForestClassifier(max_leaf_nodes=max_n_leaves))
for clf in RF_classifiers:
    clf.fit(X_train, Y_train)
train_errors = []
test_errors = []
for clf in RF_classifiers:
    test_errors.append(zero_one_loss(Y_test, clf.predict(X_test)))
    train_errors.append(zero_one_loss(Y_train, clf.predict(X_train)))
plt.plot(np.log(np.array(max_num_leaves)*100), train_errors, label='Train errors')
plt.plot(np.log(np.array(max_num_leaves)*100), test_errors, label='Test errors')
plt.title('Zero-one error vs log_max_num_leaves for Random Forest Classifier')
plt.xlabel('Log of maximum number of leaves')
plt.ylabel('Zero-one error')
plt.legend()

"""
Random Forest classifier (problem 4v.b)
"""
RF_classifiers = []
num_trees = [5, 50, 100,200, 400, 600, 800, 1000, 1200, 1400]
for num_tree in num_trees:
    RF_classifiers.append(RandomForestClassifier(n_estimators=num_tree, max_leaf_nodes=100))
for clf in RF_classifiers:
    clf.fit(X_train, Y_train)
train_losses = []
test_losses = []
for clf in RF_classifiers:
    train_losses.append(zero_one_loss(Y_train, clf.predict(X_train)))
    test_losses.append(zero_one_loss(Y_test, clf.predict(X_test)))
fig, axs = plt.subplots(2, )
axs[0].plot(np.log(np.array(num_trees)*100), (train_losses), label='Train loss')
axs[1].plot(np.log(np.array(num_trees)*100), (test_losses), label='Test loss', c='g')
axs[0].legend()
axs[1].legend()
fig.suptitle('Zero-one loss for Random Forest classifier (max_leaves=100) with various number of trees')
axs[0].set_xlabel('Log of total parameters')
axs[0].set_ylabel('Zero-one loss')
axs[1].set_xlabel('Log of total parameters')
axs[1].set_ylabel('Zero-one loss')

"""
Dual-phase Random Forest classifier (problem 4vi)
"""
"""
Phase 1
"""
max_nodes = [100, 200, 500, 1000, 1500, 2000, 3000, 4000, 4908]
classifiers = []
for max_n_nodes in max_nodes:
    classifiers.append(RandomForestClassifier(n_estimators=1, max_leaf_nodes=max_n_nodes))
for clf in classifiers:
    clf.fit(X_train, Y_train)
train_loss = []
test_loss = []
for clf in classifiers:
    train_loss.append(zero_one_loss(Y_train, clf.predict(X_train)))
    test_loss.append(zero_one_loss(Y_test, clf.predict(X_test)))
"""
Phase 2
"""
phase_2_classifiers = []
num_trees = [2, 4, 8, 16, 32, 64, 128]
for num_tree in num_trees:
    phase_2_classifiers.append(RandomForestClassifier(n_estimators=num_tree, max_leaf_nodes=4908))
for clf in phase_2_classifiers:
    clf.fit(X_train, Y_train)
phase_2_train_loss = []
phase_2_test_loss = []
for clf in phase_2_classifiers:
    phase_2_train_loss.append(zero_one_loss(Y_train, clf.predict(X_train)))
    phase_2_test_loss.append(zero_one_loss(Y_test, clf.predict(X_test)))
agg_train_loss = train_loss + phase_2_train_loss
agg_test_loss = test_loss + phase_2_test_loss
total_params = max_nodes + list(np.array(num_trees)*4908)
total_params = np.array(total_params)
plt.title('Zero-one loss vs log of total parameters for Random Forest classifier')
plt.plot(np.log(total_params), agg_train_loss, label='Train loss')
plt.plot(np.log(total_params), agg_test_loss, label='Test loss')
plt.legend()
plt.xlabel('Log of total parameters')
plt.ylabel('Zero-one loss')