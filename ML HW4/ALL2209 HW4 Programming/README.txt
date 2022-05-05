Requirements:
1. numpy
2. matplotlib

Note:
For the question 4 implementation, I currently have the dataset of two concentric circles loaded. Simply change the the two for loops under the data generation section to generate a different dataset (e.g. quadratic curves/sinusoids). You can also adjust the number of clustering iterations under the training loop. Note that I found that the clusters always converged in <= 10 iterations, which is why I set the default max_itr to 10 and didn't bother with a convergence checking condition.