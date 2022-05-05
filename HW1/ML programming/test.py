import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

accuracy_MLE = [0.6785, 0.6765, 0.6785, 0.681, 0.676, 0.6785]
accuracy_naive = [0.6805, 0.6805, 0.6785, 0.6785, 0.678, 0.6785]
accuracy_two_NN_one_norm = [0.6165, 0.6135, 0.6005, 0.6055, 0.611, 0.617]
accuracy_two_NN_two_norm = [0.6085, 0.608, 0.611, 0.606, 0.607, 0.6285]
accuracy_two_NN_inf_norm = [0.5855, 0.602, 0.607, 0.5935, 0.615, 0.601]
accuracy_ten_NN_one_norm = [0.664, 0.6595, 0.664, 0.6525, 0.654, 0.658]
accuracy_ten_NN_two_norm = [0.6625, 0.665, 0.6505, 0.654, 0.664, 0.6495]
accuracy_ten_NN_inf_norm = [0.652, 0.639, 0.654, 0.6535, 0.6475, 0.6485]
accuracy_hundred_NN_one_norm = [0.6885, 0.6935, 0.6985, 0.6915, 0.6875, 0.6955]
accuracy_hundred_NN_two_norm = [0.6915, 0.6875, 0.6905, 0.682, 0.689, 0.6875]
accuracy_hundred_NN_inf_norm = [0.6905, 0.687, 0.683, 0.688, 0.6935, 0.684]

train_sizes = [2083.5, 2500.2, 2916.9, 3333.6, 3750.3, 4167.]
plt.figure(figsize=(8, 6), dpi=80)

plt.plot(train_sizes, accuracy_MLE, label='MLE accuracy', c='r')
plt.plot(train_sizes, accuracy_naive, label='Naive Bayes accuracy', c='g')
plt.plot(train_sizes, accuracy_two_NN_one_norm, label='2-NN L1 accuracy', c='b')
plt.plot(train_sizes, accuracy_two_NN_two_norm, label='2-NN L2 accuracy', c='k')
plt.plot(train_sizes, accuracy_two_NN_inf_norm, label='2-NN L-inf accuracy', c='cyan')
plt.plot(train_sizes, accuracy_ten_NN_one_norm, 'k--', label='10-NN L1 accuracy')
plt.plot(train_sizes, accuracy_ten_NN_two_norm, 'b--', label='10-NN L2 accuracy')
plt.plot(train_sizes, accuracy_ten_NN_inf_norm, 'r--', label='10-NN L-inf accuracy', )
plt.plot(train_sizes, accuracy_hundred_NN_one_norm, 'g--', label='100-NN L1 accuracy')
plt.plot(train_sizes, accuracy_hundred_NN_two_norm, 'c--', label='100-NN L2 accuracy')
plt.plot(train_sizes, accuracy_hundred_NN_inf_norm, 'm--', label='100-NN L-inf accuracy')
plt.xlabel('Train sample size')
plt.ylabel('Test accuracy')
plt.title('Comparison of test accuracy for various training sample sizes')
plt.legend(bbox_to_anchor=(1.1, 1.05))