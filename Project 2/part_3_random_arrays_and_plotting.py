# code from Faezeh and Andy
# Modified to include comparison counts and recursion depth analysis

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import part_1_quicksort_pivots as p1
from matplotlib.ticker import ScalarFormatter
import sys  # Needed for potential recursion limit adjustments

# Optional: Increase recursion depth if needed for larger arrays in part_1
# Consider adding: sys.setrecursionlimit(10000) in part_1_quicksort_pivots.py
# Or adjust here, but better in the source file if possible.


# Define a function to generate data for different distributions.
def generate_data(distribution, size, **params):
    """Generates numpy arrays with specified distributions."""
    if distribution == "uniform":
        low = params.get("low", 0)
        high = params.get("high", 100)
        if high <= low:
            high = low + 100
        return np.random.uniform(low, high, size)
    elif distribution == "normal":
        loc = params.get("loc", 50)
        scale = params.get("scale", 10)
        if scale <= 0:
            scale = 10
        return np.random.normal(loc, scale, size)
    elif distribution == "exponential":
        scale = params.get("scale", 20)
        if scale <= 0:
            scale = 20
        return np.random.exponential(scale, size)
    elif distribution == "sorted":
        arr = np.sort(np.random.uniform(0, 100, size))
        perturb_scale = 0.5
        perturb = np.random.uniform(-perturb_scale, perturb_scale, size)
        return arr + perturb
    else:
        raise ValueError("Unknown distribution")


# Function to measure quicksort runtime using the imported quicksort from part_1
def run_quicksort_and_get_metrics(arr, pivot_strategy="median_of_three"):
    """
    Runs the quicksort algorithm from part_1 and returns key metrics.
    Converts numpy array to list before sorting.
    Returns: comparisons, execution_time, max_depth, avg_balance (or NaNs on error)
    """
    arr_list = arr.tolist()
    try:
        # Capture all return values from p1.quicksort
        comparisons, time_taken, max_depth, avg_balance = p1.quicksort(
            arr_list, pivot_strategy=pivot_strategy
        )
        return (
            comparisons,
            time_taken,
            max_depth,
        )  # , avg_balance # avg_balance currently unused but available
    except RecursionError:
        print(
            f"    WARNING: Recursion depth exceeded for size {len(arr_list)} with pivot '{pivot_strategy}'. Returning NaNs."
        )
        return float("nan"), float("nan"), float("nan")  # , float('nan')
    except Exception as e:
        print(f"    ERROR during quicksort: {e}. Returning NaNs.")
        return float("nan"), float("nan"), float("nan")  # , float('nan')


# Run experiments for each distribution and input size, averaging over multiple trials.
def run_experiments(distributions, sizes, trials, pivot_strategy):
    """
    Runs quicksort experiments, collecting time, comparisons, and depth.
    Uses the specified pivot strategy.
    """
    results = []

    for dist in distributions:
        print(f"\n  Distribution: {dist}")
        for size in sizes:
            # Lists to store metrics for each trial
            times = []
            comparisons_list = []
            depths = []
            print(f"    Size: {size}")
            for t in range(trials):
                # Generate data
                if dist == "uniform":
                    arr = generate_data("uniform", size, low=0, high=size * 10)
                elif dist == "normal":
                    arr = generate_data("normal", size, loc=size * 5, scale=size)
                elif dist == "exponential":
                    arr = generate_data("exponential", size, scale=size * 2)
                elif dist == "sorted":
                    arr = generate_data("sorted", size)
                else:
                    arr = generate_data("uniform", size, low=0, high=size * 10)

                # Run quicksort and get all metrics
                comps, time_taken, depth = run_quicksort_and_get_metrics(
                    arr, pivot_strategy=pivot_strategy
                )

                # Append metrics if the run was successful (not NaN)
                # Check time_taken for NaN as an indicator of failure
                if not np.isnan(time_taken):
                    times.append(time_taken)
                    comparisons_list.append(comps)
                    depths.append(depth)

            # Calculate averages only if there were successful runs
            num_successful_trials = len(times)
            if num_successful_trials > 0:
                avg_time = np.mean(times)
                avg_comparisons = np.mean(comparisons_list)
                avg_depth = np.mean(depths)
                print(f"      Avg Time: {avg_time:.6f} sec")
                print(f"      Avg Compares: {avg_comparisons:.1f}")
                print(f"      Avg Max Depth: {avg_depth:.1f}")
                print(
                    f"      (Based on {num_successful_trials}/{trials} successful trials)"
                )

                # Append results including new metrics
                results.append(
                    {
                        "Distribution": dist,
                        "Size": size,
                        "AverageTime": avg_time,
                        "AverageComparisons": avg_comparisons,
                        "AverageDepth": avg_depth,
                        "SuccessfulTrials": num_successful_trials,
                    }
                )
            else:  # Handle case where all trials failed
                print(f"      Avg Time: N/A (All {trials} trials failed)")
                results.append(
                    {
                        "Distribution": dist,
                        "Size": size,
                        "AverageTime": float("nan"),
                        "AverageComparisons": float("nan"),
                        "AverageDepth": float("nan"),
                        "SuccessfulTrials": 0,
                    }
                )

    return pd.DataFrame(results)


# Define experiment parameters.
distributions_to_test = ["uniform", "normal", "exponential", "sorted"]
sizes_to_test = [
    100,
    500,
    1000,
    2500,
    5000,
    7500,
]  # Added a size, adjust if recursion errors occur
trials_per_setting = 5  # Keep trials lower for faster testing

# Define the list of pivot strategies from part_1 to loop over
pivot_strategies_to_test = ["first", "last", "middle", "median_of_three"]

# --- Main Loop over Pivot Strategies ---
for current_pivot_strategy in pivot_strategies_to_test:
    print(f"\n{'='*60}")
    print(f"Starting Experiment for Pivot Strategy: '{current_pivot_strategy}'")
    print(f"{'='*60}")
    print(f"Distributions: {distributions_to_test}")
    print(f"Sizes: {sizes_to_test}")
    print(f"Trials per setting: {trials_per_setting}")
    print("-" * 30)

    # Run the experiments using the current pivot strategy
    df_results = run_experiments(
        distributions_to_test, sizes_to_test, trials_per_setting, current_pivot_strategy
    )

    # --- Plotting Section ---

    # Plot 1: Average Runtime
    df_plot_time = df_results.dropna(subset=["AverageTime"])
    if not df_plot_time.empty:
        plt.figure(figsize=(12, 7))
        for dist in df_plot_time["Distribution"].unique():
            df_subset = df_plot_time[df_plot_time["Distribution"] == dist]
            plot_sizes = pd.to_numeric(df_subset["Size"])
            plot_values = pd.to_numeric(df_subset["AverageTime"])
            plt.plot(plot_sizes, plot_values, marker="o", linestyle="-", label=dist)

        plt.xlabel("Input Size (N)")
        plt.ylabel("Average Quicksort Time (seconds)")
        plt.title(f"Quicksort Runtime using '{current_pivot_strategy}' Pivot")
        plt.legend(title="Distribution")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.xscale("log")
        plt.yscale("log")
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        plt.xticks(sizes_to_test)
        plt.tight_layout()
        plt.show()
    else:
        print(
            f"\nNo valid Runtime results to plot for pivot strategy '{current_pivot_strategy}'."
        )

    # Plot 2: Average Comparisons
    df_plot_comps = df_results.dropna(subset=["AverageComparisons"])
    if not df_plot_comps.empty:
        plt.figure(figsize=(12, 7))
        for dist in df_plot_comps["Distribution"].unique():
            df_subset = df_plot_comps[df_plot_comps["Distribution"] == dist]
            plot_sizes = pd.to_numeric(df_subset["Size"])
            plot_values = pd.to_numeric(df_subset["AverageComparisons"])
            plt.plot(
                plot_sizes, plot_values, marker="s", linestyle="--", label=dist
            )  # Different marker/style

        plt.xlabel("Input Size (N)")
        plt.ylabel("Average Number of Comparisons")
        plt.title(f"Quicksort Comparisons using '{current_pivot_strategy}' Pivot")
        plt.legend(title="Distribution")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.xscale("log")
        plt.yscale("log")  # Comparisons also expected to be O(n log n)
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        plt.xticks(sizes_to_test)
        plt.tight_layout()
        plt.show()
    else:
        print(
            f"\nNo valid Comparison results to plot for pivot strategy '{current_pivot_strategy}'."
        )

    # Plot 3: Average Max Recursion Depth
    df_plot_depth = df_results.dropna(subset=["AverageDepth"])
    if not df_plot_depth.empty:
        plt.figure(figsize=(12, 7))
        for dist in df_plot_depth["Distribution"].unique():
            df_subset = df_plot_depth[df_plot_depth["Distribution"] == dist]
            plot_sizes = pd.to_numeric(df_subset["Size"])
            plot_values = pd.to_numeric(df_subset["AverageDepth"])
            plt.plot(
                plot_sizes, plot_values, marker="^", linestyle=":", label=dist
            )  # Different marker/style

        plt.xlabel("Input Size (N)")
        plt.ylabel("Average Max Recursion Depth")
        plt.title(
            f"Quicksort Max Recursion Depth using '{current_pivot_strategy}' Pivot"
        )
        plt.legend(title="Distribution")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        # Depth might grow logarithmically or linearly (worst case)
        plt.xscale("log")
        plt.yscale("linear")  # Let's start with linear for depth
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        # plt.gca().yaxis.set_major_formatter(ScalarFormatter()) # Use default for linear
        plt.xticks(sizes_to_test)
        plt.tight_layout()
        plt.show()
    else:
        print(
            f"\nNo valid Depth results to plot for pivot strategy '{current_pivot_strategy}'."
        )

    if not df_results.empty:
        try:
            output_filename = (
                f"quicksort_{current_pivot_strategy}_experiment_results.csv"
            )
            df_results.to_csv(output_filename, index=False)
            print(
                f"\nResults (Time, Comparisons, Depth) for '{current_pivot_strategy}' saved to {output_filename}"
            )
        except Exception as e:
            print(
                f"\nCould not save results for '{current_pivot_strategy}' to CSV: {e}"
            )

print(f"\n{'='*60}")
print("All experiments complete.")
print(f"{'='*60}")
