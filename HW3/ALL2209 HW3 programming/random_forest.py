import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from google.colab import drive
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import csv

#Load data
drive.mount('/content/drive')
mat = scipy.io.loadmat('/content/drive/My Drive/ML hw3/MSdata.mat')


#Prepare data
x_test = mat['testx']

x_train = mat['trainx']

y_train = np.array(mat['trainy'], dtype=np.float)

n_train = x_train.shape[0]

n_test = x_test.shape[0]
d = x_train.shape[1]


#Normalization
mu = np.mean(x_train, 0)
sigma = np.var(x_train, 0)

x_test = (x_test-mu)/sigma
x_train = (x_train-mu)/sigma


#Model
RF = RandomForestRegressor(100, max_leaf_nodes=500)
RF.fit(x_train, y_train)
RF.predict(x_test)


#Create outputs
outputs = []
predictions = RF.predict(x_test)
for i in range(n_test):
  outputs.append([i+1, predictions[i]])

with open('/content/drive/My Drive/ML hw3/output.csv', 'w') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(['dataid', 'prediction'])
  writer.writerows(outputs)