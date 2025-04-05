

# code from Faezeh and Andy

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd



def quicksort(arr, pivot_function):
    if len(arr) < 2:
        return arr
    else:
        pivot_index = pivot_function(arr)
        pivot = arr[pivot_index]
        less = [i for j, i in enumerate(arr) if i <= pivot and j != pivot_index]
        greater = [i for j, i in enumerate(arr) if i > pivot and j != pivot_index]
        return quicksort(less, pivot_function) + [pivot] + quicksort(greater, pivot_function)

# Example pivot selection functions
def first_element_pivot(arr):
    return 0

def last_element_pivot(arr):
    return len(arr) - 1

def random_element_pivot(arr):
    import random
    return random.randint(0, len(arr) - 1)

def median_of_three_pivot(arr):
    mid = len(arr) // 2
    if len(arr) > 2:
        a = arr[0]
        b = arr[mid]
        c = arr[-1]
        if a <= b <= c or c <= b <= a:
            return mid
        elif b <= a <= c or c <= a <= b:
            return 0
        else:
            return len(arr) - 1
    else:
        return 0


# Define a function to generate data for different distributions.
def generate_data(distribution, size, **params):
    if distribution == 'uniform':
        low = params.get('low', 0)
        high = params.get('high', 100)
        return np.random.uniform(low, high, size)
    elif distribution == 'normal':
        loc = params.get('loc', 50)
        scale = params.get('scale', 10)
        return np.random.normal(loc, scale, size)
    elif distribution == 'exponential':
        scale = params.get('scale', 20)
        return np.random.exponential(scale, size)
    elif distribution == 'sorted':
        # Generating a nearly sorted array: sort a uniform array then perturb it slightly.
        arr = np.sort(np.random.uniform(0, 100, size))
        perturb = np.random.uniform(-0.5, 0.5, size)
        return arr + perturb
    else:
        raise ValueError("Unknown distribution")

# Function to measure quicksort runtime using NumPy's quicksort.
def measure_quicksort_time(arr):
    start = time.perf_counter()
    # Using NumPy's sort with the quicksort algorithm.
    #_ = np.sort(arr, kind='quicksort')
    _ = quicksort(arr, first_element_pivot)
    end = time.perf_counter()
    return end - start

# Run experiments for each distribution and input size, averaging over multiple trials.
def run_experiments(distributions, sizes, trials):
    results = []
    for dist in distributions:
        for size in sizes:
            times = []
            for t in range(trials):
                # Generate the array based on the current distribution.
                if dist == 'uniform':
                    arr = generate_data('uniform', size, low=0, high=1000)
                elif dist == 'normal':
                    arr = generate_data('normal', size, loc=500, scale=50)
                elif dist == 'exponential':
                    arr = generate_data('exponential', size, scale=100)
                elif dist == 'sorted':
                    arr = generate_data('sorted', size)
                else:
                    arr = generate_data('uniform', size)
                # Measure the quicksort runtime.
                time_taken = measure_quicksort_time(arr)
                times.append(time_taken)
            avg_time = np.mean(times)
            results.append({
                'Distribution': dist,
                'Size': size,
                'AverageTime': avg_time
            })
            print(f"Distribution: {dist}, Size: {size}, Avg Time: {avg_time:.6f} seconds")
    return pd.DataFrame(results)

# Define experiment parameters.
distributions = ['uniform', 'normal', 'exponential', 'sorted']
sizes = [100, 500, 1000, 5000] # [12500, 25000, 50000, 100000]  # You can adjust the sizes based on your needs.
trials = 10  # Number of trials to average the timing.

# Run the experiments.
df_results = run_experiments(distributions, sizes, trials)

# Step 8: Visualize the results using matplotlib.
plt.figure(figsize=(10, 6))
for dist in distributions:
    df_subset = df_results[df_results['Distribution'] == dist]
    plt.plot(df_subset['Size'], df_subset['AverageTime'], marker='o', label=dist)
plt.xlabel("Input Size")
plt.ylabel("Average Quicksort Time (seconds)")
plt.title("Quicksort Performance Across Different Distributions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optionally, save the results to a CSV file.
df_results.to_csv("quicksort_experiment_results.csv", index=False)
print("Results saved to quicksort_experiment_results.csv")


# chatGPT:


# Example usage
"""
arr = [64, 34, 25, 12, 22, 11, 90]

sorted_arr_first = quicksort(arr.copy(), first_element_pivot)
print("Sorted array (first element pivot):", sorted_arr_first)

sorted_arr_last = quicksort(arr.copy(), last_element_pivot)
print("Sorted array (last element pivot):", sorted_arr_last)

sorted_arr_random = quicksort(arr.copy(), random_element_pivot)
print("Sorted array (random element pivot):", sorted_arr_random)

sorted_arr_median = quicksort(arr.copy(), median_of_three_pivot)
print("Sorted array (median of three pivot):", sorted_arr_median)


"""