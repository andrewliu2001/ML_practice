import numpy as np
import matplotlib.pyplot as plt


"""
    Create the distance matrix
"""
distances = [206, 429, 1504, 963, 2976, 3095, 2979, 1949, 233, 1308, 802, 2815, 2934, 2786, 1771, 1075, 671, 2684, 2799, 2631, 1616, 1329, 3273, 3053, 2687, 2037, 2013, 2142, 2054, 996, 808, 1131, 1307, 379, 1235, 1059]

D_matrix = np.zeros((9,9))

counter = 0
for i in range(0,9):
  for j in range(i,9):
    if(i==j):
      D_matrix[i][j]=0
    else:
      D_matrix[i][j] = distances[counter]
      D_matrix[j][i] = distances[counter]
      counter += 1

"""
    Random initialization of coordinate values
"""

xs = np.random.randn(9,2)

"""
    Function for computing the derivative
"""

def compute_derivatives(xs, D_matrix):
  derivatives = np.zeros((9,2))
  for i in range(9):
    for j in range(9):
      if j != i:
        derivatives[i] += 2*(1-D_matrix[i][j]/np.linalg.norm(xs[i]-xs[j]))*(xs[i]-xs[j])
  
  return derivatives


"""
    Training loop
"""

lr = 0.01
while True:
  derivatives = compute_derivatives(xs, D_matrix)
  xs = xs - derivatives*lr

  if np.linalg.norm(derivatives) < 1e-10:
    break

"""
    Plotting
"""

annotations=['BOS', 'NYC', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN']
plt.rcParams["figure.figsize"] = (10,10)

plt.scatter(xs[:,0], xs[:,1])
for i, label in enumerate(annotations):
  plt.annotate(label, (xs[:,0][i], xs[:,1][i]))

plt.title('Optimized locations of cities')
