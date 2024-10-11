import numpy as np

def dirichlet_variance(vector):
    alpha_0 = sum(vector)  # Sum of all alphas (alpha_0)
    variances = np.zeros_like(vector)
    
    for i, p_i in enumerate(vector):
        alpha_i = p_i * alpha_0  # alpha_i is proportional to the mean p_i
        variances[i] = (alpha_i * (alpha_0 - alpha_i)) / (alpha_0 ** 2 * (alpha_0 + 1))
    
    return variances

# Simulating model logits (before softmax)
output = np.random.uniform(0, 1, 10)
print(output)

variance = dirichlet_variance(output)
print(variance)
print(sum(variance))