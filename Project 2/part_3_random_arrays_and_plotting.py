

# code from Faezeh and Andy

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import part_1_quicksort_pivots as p1


# Define a function to generate data for different distributions.
def generate_data(distribution, size, **params):
    """Generates numpy arrays with specified distributions."""
    if distribution == 'uniform':
        low = params.get('low', 0)
        high = params.get('high', 100)
        # Ensure high > low for uniform distribution
        if high <= low:
            high = low + 100 # Default range if invalid params given
        return np.random.uniform(low, high, size)
    elif distribution == 'normal':
        loc = params.get('loc', 50)
        scale = params.get('scale', 10)
        # Ensure scale is positive
        if scale <= 0:
            scale = 10 # Default scale
        return np.random.normal(loc, scale, size)
    elif distribution == 'exponential':
        scale = params.get('scale', 20)
        # Ensure scale is positive
        if scale <= 0:
            scale = 20 # Default scale
        return np.random.exponential(scale, size)
    elif distribution == 'sorted':
        # Generating a nearly sorted array: sort a uniform array then perturb it slightly.
        arr = np.sort(np.random.uniform(0, 100, size))
        # Adding small random noise to make it 'nearly' sorted
        perturb_scale = 0.5 # Adjust noise level if needed
        perturb = np.random.uniform(-perturb_scale, perturb_scale, size)
        return arr + perturb
    else:
        raise ValueError("Unknown distribution")

# Function to measure quicksort runtime using the imported quicksort from part_1
def measure_quicksort_time(arr, pivot_strategy="median_of_three"):
    """
    Measures the execution time of the quicksort algorithm from part_1.
    Converts numpy array to list before sorting.
    Returns only the execution time.
    """
    # Convert numpy array to Python list, as part_1's quicksort expects a list
    arr_list = arr.tolist()

    # Call the quicksort function from part_1
    # It returns (comparisons, execution_time)
    # We only need the execution_time for this script's purpose
    try:
        # Use the specified pivot strategy from part_1's options
        _, time_taken, _, _ = p1.quicksort(arr_list, pivot_strategy=pivot_strategy)
        return time_taken
    except RecursionError:
        print(f"    WARNING: Recursion depth exceeded for size {len(arr_list)} with pivot '{pivot_strategy}'. Returning NaN.")
        return float('nan') # Indicate failure
    except Exception as e:
        print(f"    ERROR during quicksort: {e}. Returning NaN.")
        return float('nan') # Indicate failure

# Run experiments for each distribution and input size, averaging over multiple trials.
def run_experiments(distributions, sizes, trials, pivot_strategy):
    """Runs timing experiments for quicksort using the specified pivot strategy."""
    results = []
    print(f"--- Running Experiments ---")
    print(f"Distributions: {distributions}")
    print(f"Sizes: {sizes}")
    print(f"Trials per setting: {trials}")
    print(f"Using Pivot Strategy: '{pivot_strategy}' (from part_1)")
    print("-" * 30)

    for dist in distributions:
        print(f"\nDistribution: {dist}")
        for size in sizes:
            times = []
            print(f"  Size: {size}")
            for t in range(trials):
                # Generate the array based on the current distribution.
                # Using parameters suitable for the distributions
                if dist == 'uniform':
                    arr = generate_data('uniform', size, low=0, high=size*10) # Wider range
                elif dist == 'normal':
                    arr = generate_data('normal', size, loc=size*5, scale=size) # Scale with size
                elif dist == 'exponential':
                    arr = generate_data('exponential', size, scale=size*2) # Scale with size
                elif dist == 'sorted':
                    arr = generate_data('sorted', size)
                else: # Default to uniform if unknown, though generate_data would raise error
                     arr = generate_data('uniform', size, low=0, high=size*10)

                # Measure the quicksort runtime using the function from part_1
                time_taken = measure_quicksort_time(arr, pivot_strategy=pivot_strategy)
                if not np.isnan(time_taken): # Only append valid times
                     times.append(time_taken)
                # Small delay can sometimes help with performance counter precision if runs are very fast
                # time.sleep(0.01)

            if times: # Calculate average only if there were successful runs
                avg_time = np.mean(times)
                print(f"    Avg Time: {avg_time:.6f} seconds over {len(times)} successful trials")
                results.append({
                    'Distribution': dist,
                    'Size': size,
                    'AverageTime': avg_time
                })
            else: # Handle case where all trials failed (e.g., RecursionError)
                 print(f"    Avg Time: N/A (All {trials} trials failed)")
                 results.append({
                    'Distribution': dist,
                    'Size': size,
                    'AverageTime': float('nan') # Store NaN for failed averages
                })

    return pd.DataFrame(results)

# Define experiment parameters.
distributions_to_test = ['uniform', 'normal', 'exponential', 'sorted']
# Note: Larger sizes might hit recursion limits depending on data and pivot strategy
# If you encounter RecursionError, you might need to uncomment `sys.setrecursionlimit`
# in part_1_quicksort_pivots.py (use with caution).
sizes_to_test = [100, 500, 1000, 2500, 5000] # Adjusted for potential recursion depth
trials_per_setting = 5  # Reduced trials slightly for faster feedback during testing
# Choose the pivot strategy from part_1 to use for this experiment
# Options: "first", "last", "middle", "median_of_three"
selected_pivot_strategy = "middle"

# Run the experiments using the chosen pivot strategy
df_results = run_experiments(distributions_to_test, sizes_to_test, trials_per_setting, selected_pivot_strategy)

# Filter out rows where AverageTime is NaN before plotting
df_plot = df_results.dropna(subset=['AverageTime'])


if not df_plot.empty:
    # Visualize the results using matplotlib.
    plt.figure(figsize=(12, 7)) # Slightly larger figure
    for dist in df_plot['Distribution'].unique():
        df_subset = df_plot[df_plot['Distribution'] == dist]
        plt.plot(df_subset['Size'], df_subset['AverageTime'], marker='o', linestyle='-', label=dist)

    plt.xlabel("Input Size (N)")
    plt.ylabel("Average Quicksort Time (seconds)")
    plt.title(f"Quicksort Performance using '{selected_pivot_strategy}' Pivot\nAcross Different Input Distributions")
    plt.legend(title="Distribution")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xscale('log') # Often helpful to see trends across orders of magnitude
    plt.yscale('log') # Time complexity is often visualized on log-log scale
    # Add specific tick marks if needed for log scale clarity
    # from matplotlib.ticker import LogLocator, NullFormatter
    # plt.gca().xaxis.set_major_locator(LogLocator(base=10.0))
    # plt.gca().yaxis.set_major_locator(LogLocator(base=10.0))
    # Use ScalarFormatter for non-scientific notation if preferred for ticks
    from matplotlib.ticker import ScalarFormatter
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.xticks(sizes_to_test) # Ensure original sizes are marked

    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()
else:
    print("\nNo valid results to plot. Experiments might have failed (e.g., recursion depth).")


# Optionally, save the results to a CSV file.
if not df_results.empty:
    try:
        output_filename = f"quicksort_{selected_pivot_strategy}_experiment_results.csv"
        df_results.to_csv(output_filename, index=False)
        print(f"\nResults saved to {output_filename}")
    except Exception as e:
        print(f"\nCould not save results to CSV: {e}")
