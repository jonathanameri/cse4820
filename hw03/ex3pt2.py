import numpy as np
from scipy.spatial.distance import euclidean

# Function to calculate distances for n = 10
def calculate_distances_for_n_10():
    n_samples = 101
    n_dimensions = 10

    # Generate 101 samples of X in 10 dimensions
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
    
    return Dmax, Dmin, r

# Execute the function and display the results
Dmax, Dmin, r = calculate_distances_for_n_10()
print(f"Dmax: {Dmax}")
print(f"Dmin: {Dmin}")
print(f"r: {r}")
