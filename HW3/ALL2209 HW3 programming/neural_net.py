import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from google.colab import drive
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import csv

drive.mount('/content/drive')
mat = scipy.io.loadmat('/content/drive/My Drive/ML hw3/MSdata.mat')

percentage_train = 0.99

x_test = mat['testx']

orig_n_train = mat['trainx'].shape[0]

x_train = mat['trainx'][:int(orig_n_train*percentage_train)]
x_valid = mat['trainx'][int(orig_n_train*percentage_train):]

y_train = np.array(mat['trainy'][:int(orig_n_train*percentage_train)], dtype=np.float)
y_valid = np.array(mat['trainy'][int(orig_n_train*percentage_train):], dtype=np.float)

n_train = x_train.shape[0]
n_valid = x_valid.shape[0]
n_test = x_test.shape[0]
d = x_train.shape[1]


#Normalization
mu = np.mean(x_train, 0)
sigma = np.var(x_train, 0)

x_test = (x_test-mu)/sigma
x_train = (x_train-mu)/sigma
x_valid = (x_valid-mu)/sigma


#Cast as TensorDatasets
x_test = torch.Tensor(x_test)
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_valid = torch.Tensor(x_valid)
y_valid = torch.Tensor(y_valid)

ds = TensorDataset(x_train, y_train)
dataloader = DataLoader(ds, batch_size=1000)
valid_ds = TensorDataset(x_valid, y_valid)
valid_dataloader = DataLoader(valid_ds, batch_size=1000)

#Model
model = nn.Sequential(
    nn.Linear(d, 90),
    nn.ReLU(inplace=True),
    nn.Linear(90, 20),
    nn.ReLU(inplace=True),
    nn.Linear(20, 1),
)


#Random weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)


#Train loop
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    valid_loss = 0.0
    for i, data in enumerate(valid_dataloader):
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()

    print(f'Train loss: {running_loss/n_train}, Valid loss: {valid_loss/n_valid}')
print('Finished Training')


#Create outputs
outputs = []
predictions = model(x_test)
for i in range(n_test):
  outputs.append([i+1, predictions[i].item()])
  
with open('/content/drive/My Drive/ML hw3/output.csv', 'w') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(['dataid', 'prediction'])
  writer.writerows(outputs)

