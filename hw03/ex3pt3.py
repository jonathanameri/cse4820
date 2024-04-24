import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

def calculate_r_values(n_dimensions_list):
    n_samples = 101
    results = []

    for n_dimensions in n_dimensions_list:
        # Generate 101 samples of X in n dimensions
        X = np.random.uniform(0, 1, (n_samples, n_dimensions))
        
        # Choose one sample (e.g., the first one)
        chosen_sample = X[0, :]
        
        # Calculate pairwise Euclidean distances from the chosen sample to all other samples
        distances = np.array([euclidean(chosen_sample, X[i, :]) for i in range(1, n_samples)])
        
        # Calculate Dmax and Dmin
        Dmax = np.max(distances)
        Dmin = np.min(distances)
        
        # Calculate r
        r = (Dmax - Dmin) / Dmin
        
        results.append((n_dimensions, r))
    
    # Plotting
    dimensions, rs = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, rs, marker='o')
    plt.xscale('log')
    plt.xlabel('Dimensionality (n)')
    plt.ylabel('r value')
    plt.title('Behavior of distances in high dimensions')
    plt.grid(True)
    plt.show()
    return results

n_dimensions_list = [1, 10, 100, 10**3, 10**4, 10**5]
calculate_r_values(n_dimensions_list)
