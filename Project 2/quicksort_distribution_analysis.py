# --- File: quicksort_distribution_analysis.py ---

import numpy as np
import pandas as pd
import time # Keep time for potential other uses, though timing is in imported func
import matplotlib.pyplot as plt # Keep for plotting results

# --- Import Quicksort from Part 1 ---
try:
    # Assuming the Part 1 file is named quicksort_pivots.py and is in the same directory
    # We need the main quicksort function which handles strategy selection,
    # comparison counting, and timing internally.
    from quicksort_pivots import quicksort as quicksort_part1
    print("Successfully imported quicksort from quicksort_pivots.py")
except ImportError:
    print("ERROR: Could not import quicksort from quicksort_pivots.py.")
    print("Ensure 'quicksort_pivots.py' exists in the same directory and contains the quicksort function.")
    exit() # Exit if the required function can't be imported

# --- Data Generation (using distributions) ---

def generate_data(distribution, size, **params):
    """Generates a NumPy array with the specified distribution and size."""
    if distribution == 'uniform':
        low = params.get('low', 0)
        high = params.get('high', size * 10) # Adjusted default high
        return np.random.uniform(low, high, size)
    elif distribution == 'normal':
        loc = params.get('loc', size / 2) # Adjusted default loc
        scale = params.get('scale', size / 6) # Adjusted default scale
        return np.random.normal(loc, scale, size)
    elif distribution == 'exponential':
        # Exponential distribution values start at 0 and decrease.
        # Scale parameter (lambda) is rate; scale = 1/lambda in numpy.
        scale = params.get('scale', size / 5) # Example scale parameter
        return np.random.exponential(scale, size)
    elif distribution == 'sorted_nearly':
        # Generate sorted data and slightly perturb it
        arr = np.linspace(0, size * 10, size) # Create perfectly sorted data
        # Add small random noise (adjust magnitude as needed)
        noise = np.random.normal(0, size*0.1, size)
        arr += noise
        # Optional: shuffle a small percentage of elements
        swap_indices = np.random.choice(size, size // 20, replace=False) # Swap 5%
        np.random.shuffle(arr[swap_indices]) # Shuffle selected elements
        return arr
    elif distribution == 'sorted_reversed':
        # Generate data sorted in reverse order
        return np.linspace(size * 10, 0, size) # Create perfectly reverse sorted data
    else:
        raise ValueError(f"Unknown distribution type: {distribution}")

# --- Experiment Runner ---

def run_distribution_experiments(distributions, sizes, trials, pivot_strategy):
    """
    Runs Quicksort experiments for different data distributions and sizes.

    Args:
        distributions (list): List of distribution names (strings) to test.
        sizes (list): List of array sizes (integers) to test.
        trials (int): Number of random arrays to generate per size/distribution.
        pivot_strategy (str): The pivot strategy to use (must match one in Part 1).

    Returns:
        pandas.DataFrame: A DataFrame containing the results.
    """
    results_list = [] # Store results as a list of dictionaries

    print("-" * 70)
    print(f"Running Quicksort Analysis with Pivot Strategy: '{pivot_strategy}'")
    print(f"Distributions: {distributions}")
    print(f"Sizes: {sizes}")
    print(f"Trials per setting: {trials}")
    print("-" * 70)
    print("{:<18} {:<10} {:<18} {:<18}".format(
        "Distribution", "Size", "Avg Compares", "Avg Time (s)"
        ))
    print("-" * 70)

    for dist in distributions:
        for size in sizes:
            run_times = []
            run_comparisons = []

            # print(f"  Testing: Distribution={dist}, Size={size}...") # Verbose

            for t in range(trials):
                # 1. Generate data
                try:
                    # Provide specific parameters for distributions if needed, otherwise use defaults
                    if dist == 'uniform':
                        arr_np = generate_data(dist, size, high=size*10)
                    elif dist == 'normal':
                        arr_np = generate_data(dist, size, loc=size/2, scale=size/6)
                    else: # Use defaults for others or add specific params
                         arr_np = generate_data(dist, size)

                    # Convert numpy array to Python list for Part 1's quicksort
                    arr_list = arr_np.tolist()

                    # 2. Run Quicksort (imported from Part 1) and get results
                    comparisons, exec_time = quicksort_part1(arr_list, pivot_strategy=pivot_strategy)

                    # 3. Record results for this trial
                    run_times.append(exec_time)
                    run_comparisons.append(comparisons)

                except RecursionError:
                    print(f"    WARNING: Recursion depth exceeded for {dist}, size {size}, trial {t+1}. Skipping trial.")
                    # Optionally append NaN or skip trial for averages
                    run_times.append(np.nan)
                    run_comparisons.append(np.nan)
                except Exception as e:
                    print(f"    ERROR: An error occurred for {dist}, size {size}, trial {t+1}: {e}")
                    run_times.append(np.nan)
                    run_comparisons.append(np.nan)

            # Calculate averages for this size/distribution combination, ignoring NaNs
            avg_time = np.nanmean(run_times) if run_times else 0
            avg_comparisons = np.nanmean(run_comparisons) if run_comparisons else 0

            # Store results
            results_list.append({
                'Distribution': dist,
                'Size': size,
                'Trials': trials, # Store number of trials attempted
                'SuccessfulTrials': len(run_times) - np.isnan(run_times).sum(), # Store successful trials
                'PivotStrategy': pivot_strategy,
                'AverageComparisons': avg_comparisons,
                'AverageTime': avg_time
            })

            print("{:<18} {:<10} {:<18.1f} {:<18.6f}".format(
                dist, size, avg_comparisons, avg_time
            ))

    print("-" * 70)
    return pd.DataFrame(results_list)

# --- Main Execution ---

if __name__ == "__main__":
    # --- Experiment Parameters ---
    # Choose distributions to test
    distributions_to_test = ['uniform', 'normal', 'exponential', 'sorted_nearly', 'sorted_reversed']
    # Choose array sizes
    array_sizes = [100, 500, 1000, 2500, 5000] # Add larger sizes cautiously
    # Number of trials per setting
    num_trials = 15 # e.g., 10-20 as suggested
    # CHOOSE PIVOT STRATEGY: Select one strategy implemented in quicksort_pivots.py
    # Options likely: 'first', 'last', 'middle', 'median_of_three'
    chosen_pivot_strategy = 'median_of_three' # Example: choosing median-of-three

    # --- Run Experiments ---
    df_results = run_distribution_experiments(
        distributions=distributions_to_test,
        sizes=array_sizes,
        trials=num_trials,
        pivot_strategy=chosen_pivot_strategy
    )

    # --- Save Results ---
    results_filename = f"quicksort_distribution_results_{chosen_pivot_strategy}.csv"
    try:
        df_results.to_csv(results_filename, index=False)
        print(f"\nResults saved to {results_filename}")
    except Exception as e:
        print(f"\nERROR: Could not save results to CSV: {e}")

    # --- Visualize Results (Time vs Size for different distributions) ---
    plt.figure(figsize=(12, 7))
    for dist in distributions_to_test:
        df_subset = df_results[df_results['Distribution'] == dist]
        if not df_subset.empty:
            plt.plot(df_subset['Size'], df_subset['AverageTime'], marker='o', linestyle='-', label=f"{dist} (Time)")

            # Optional: Plot comparisons on a secondary axis if scales differ wildly
            # ax2 = plt.twinx()
            # ax2.plot(df_subset['Size'], df_subset['AverageComparisons'], marker='x', linestyle='--', label=f"{dist} (Comps)")
            # ax2.set_ylabel("Average Comparisons", color='tab:red')
            # ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.xlabel("Input Size (N)")
    plt.ylabel("Average Execution Time (seconds)")
    plt.title(f"Quicksort Performance ({chosen_pivot_strategy} pivot) Across Distributions")
    plt.legend(loc='upper left') # Adjust legend location if needed
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xscale('log') # Consider log scale for size if range is large
    plt.yscale('log') # Consider log scale for time if it grows rapidly
    plt.tight_layout()
    plot_filename = f"quicksort_distribution_plot_{chosen_pivot_strategy}.png"
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"ERROR: Could not save plot: {e}")
    plt.show()

    print("\nAnalysis complete.")