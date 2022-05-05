import numpy as np
import matplotlib.pyplot as plt

"""
Data generation
"""

X = np.empty((100,2))
t = np.linspace(-5,5,50)
for i in range(50):
  theta = np.random.rand()*100
  x = np.array([np.cos(theta), np.sin(theta)])
  X[i] = x

for i in range(50):
  theta = np.random.rand()*100
  x = np.array([5*np.cos(theta), 5*np.sin(theta)])
  X[50+i] = x

n = X.shape[0]


"""
Initializing assignments to clusters
"""

k=2
assignments = {}

#Initialize with random assignments
for i in range(n):
  assignments[i] = np.random.randint(0, k)


"""
Helper functions
"""
def isAssignedTo(i, j):
  """
  Calculatez z_ij
  """
  return (assignments[i] == j)

def alpha(i, j, denom):
  """
  Calculates alpha_ij
  """
  zij = isAssignedTo(i, j)
  return zij/denom

def denominator(j):
  """
  Calculates denominator for alpha_ij
  """
  denom = 0
  for k in range(n):
    denom += isAssignedTo(k,j)
  return denom

def kernel(i, j, kernel_type):
  """
  Computes kernel inner product between X[i], X[j]
  """
  if kernel_type == 'rbf':
    return np.exp(-(np.linalg.norm(X[i]-X[j])**2)/9)
  if kernel_type == 'quadratic':
    return (1+np.dot(X[i],X[j]))**2
  if kernel_type == 'linear':
    return np.dot(X[i],X[j])

def distances_between_data(X, kernel_type):
  """
  Creates table (numpy array) of kernelized distances between data points
  """
  distances_between_data = np.zeros((n, n))
  for i in range(n):
    for j in range(n):
      distances_between_data[i][j] = kernel(i, j, kernel_type)

  return distances_between_data

def distance_to_cluster(dphis,i,j):
  """
  Returns kernelized distance between ith data point and jth cluster center.
  """
  dist = dphis[i,i]
  denom = denominator(j)
  for l in range(n):
    for m in range(n):
      dist += alpha(l,j, denom)*alpha(m,j, denom)*dphis[l,m]
  for l in range(n):
    dist -= 2*alpha(l,j, denom)*dphis[i,l]
  return dist



"""
Training loop
"""
dphis = distances_between_data(X, 'rbf')


for itr in range(10):
  for i in range(n):
    min_dist = np.inf
    closest_center = 0
    for j in range(k):
      dist = distance_to_cluster(dphis, i,j)
      if dist < min_dist:
        min_dist = dist
        closest_center = j
    assignments[i] = closest_center
    
"""
Plotting
"""
color_map = {0:'green', 1:'red'}
for i in range(n):
  plt.scatter(X[i][0], X[i][1], c=color_map[assignments[i]])