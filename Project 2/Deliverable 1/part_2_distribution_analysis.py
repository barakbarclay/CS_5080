import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

from part_1_quicksort_pivots import quicksort

# Code from Faezeh


# Define a function to generate data for different distributions.
def generate_data(distribution, size, **params):
    if distribution == "uniform":
        low = params.get("low", 0)
        high = params.get("high", 100)
        return np.random.uniform(low, high, size)
    elif distribution == "normal":
        loc = params.get("loc", 50)
        scale = params.get("scale", 10)
        return np.random.normal(loc, scale, size)
    elif distribution == "exponential":
        scale = params.get("scale", 20)
        return np.random.exponential(scale, size)
    elif distribution == "sorted":
        # Generating a nearly sorted array: sort a uniform array then perturb it slightly.
        arr = np.sort(np.random.uniform(0, 100, size))
        perturb = np.random.uniform(-0.5, 0.5, size)
        # Ensure no negative values if the original range was non-negative
        arr = np.maximum(0, arr + perturb)
        return arr
    else:
        raise ValueError("Unknown distribution")


# Run experiments for each distribution and input size, averaging over multiple trials.
def run_experiments(distributions, sizes, trials, pivot_strategy_to_use):
    results = []
    print(f"\n--- Running Experiments for Pivot Strategy: {pivot_strategy_to_use} ---")

    for dist in distributions:
        for size in sizes:
            times = []
            comparisons_list = []
            print(f"  Testing Distribution: {dist}, Size: {size}...")
            for t in range(trials):
                # Generate the array based on the current distribution.
                if dist == "uniform":
                    arr_np = generate_data("uniform", size, low=0, high=1000)
                elif dist == "normal":
                    arr_np = generate_data("normal", size, loc=500, scale=50)
                elif dist == "exponential":
                    arr_np = generate_data("exponential", size, scale=100)
                elif dist == "sorted":
                    arr_np = generate_data("sorted", size)
                else:  # Default case or handle specific other distributions
                    arr_np = generate_data("uniform", size, low=0, high=1000)

                arr_list = arr_np.tolist()
                arr_copy_for_sort = arr_list[:]  # Work on a copy

                try:
                    # Call quicksort with the passed pivot strategy
                    comps, time_taken, _, _ = quicksort(
                        arr_copy_for_sort, pivot_strategy=pivot_strategy_to_use
                    )
                    times.append(time_taken)
                    comparisons_list.append(comps)
                except RecursionError:
                    print(
                        f"      WARNING: Recursion depth exceeded for {dist}, size {size}, trial {t+1}. Skipping trial."
                    )
                    times.append(float("nan"))
                    comparisons_list.append(float("nan"))
                except Exception as e:
                    print(
                        f"      ERROR occurred during quicksort: {e}. Skipping trial."
                    )
                    times.append(float("nan"))
                    comparisons_list.append(float("nan"))

            avg_time = (
                np.nanmean(times) if not np.all(np.isnan(times)) else float("nan")
            )
            avg_comparisons = (
                np.nanmean(comparisons_list)
                if not np.all(np.isnan(comparisons_list))
                else float("nan")
            )

            results.append(
                {
                    "Distribution": dist,
                    "Size": size,
                    "AverageTime": avg_time,
                    "AverageComparisons": avg_comparisons,
                }
            )
            print(
                f"    Distribution: {dist}, Size: {size}, Avg Time: {avg_time:.6f}s, Avg Compares: {avg_comparisons:.1f}"
            )

    return pd.DataFrame(results)


# Define experiment parameters.
distributions = ["uniform", "normal", "exponential", "sorted"]
sizes = [100, 500, 1000, 2500, 5000]
trials = 10  # Number of trials to average the timing.

# Define the pivot strategies to loop over
pivot_strategies = ["first", "last", "middle", "median_of_three"]

# --- Main Loop ---
for strategy in pivot_strategies:
    # Run the experiments for the current strategy
    df_results = run_experiments(distributions, sizes, trials, strategy)

    # --- Plotting for the current strategy ---
    plt.figure(figsize=(12, 7))  # Slightly larger figure

    # Plot Time
    plt.subplot(1, 2, 1)  # Create subplot for time
    for dist in distributions:
        df_subset = df_results[df_results["Distribution"] == dist]
        plt.plot(
            df_subset["Size"],
            df_subset["AverageTime"],
            marker="o",
            linestyle="-",
            label=dist,
        )
    plt.xlabel("Input Size (n)")
    plt.ylabel("Average Quicksort Time (seconds)")
    plt.title(f"Time ({strategy} pivot)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.yscale("log")  # Use log scale for time if it varies widely
    plt.xscale("log")  # Use log scale for size if needed

    # Plot Comparisons
    plt.subplot(1, 2, 2)  # Create subplot for comparisons
    for dist in distributions:
        df_subset = df_results[df_results["Distribution"] == dist]
        plt.plot(
            df_subset["Size"],
            df_subset["AverageComparisons"],
            marker="x",
            linestyle="--",
            label=dist,
        )
    plt.xlabel("Input Size (n)")
    plt.ylabel("Average Comparisons")
    plt.title(f"Comparisons ({strategy} pivot)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.yscale("log")  # Use log scale for comparisons often helps visualization
    plt.xscale("log")  # Use log scale for size if needed

    # Overall figure title and layout adjustment
    plt.suptitle(f"Quicksort Performance - Pivot Strategy: {strategy}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    # Save the plot
    plot_filename = f"quicksort_performance_{strategy}_pivot.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()  # Display the plot
    plt.close()  # Close the figure to free memory before the next loop iteration

    # --- Save results to a CSV file for the current strategy ---
    csv_filename = f"quicksort_results_{strategy}_pivot.csv"
    try:
        df_results.to_csv(csv_filename, index=False, float_format="%.8f")
        print(f"Results for {strategy} pivot saved to {csv_filename}")
    except Exception as e:
        print(f"Error saving CSV file {csv_filename}: {e}")

print("\nAll experiments completed.")
