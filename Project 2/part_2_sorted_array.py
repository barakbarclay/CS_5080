import sys

sys.setrecursionlimit(10000)  # Increase recursion limit to allow deeper recursions

import random
import time
import matplotlib.pyplot as plt
import numpy as np

# Code from Faezeh

# ===============================
# Step 1: Create and perturb arrays
# ===============================
def generate_sorted_array(n):
    """Generates a sorted array of integers from 0 to n-1."""
    return list(range(n))


def perturb_array(arr, percent):
    """
    Perturbs the array by swapping a number of randomly chosen element pairs.
    'percent' is the fraction of the total number of elements to swap.
    """
    arr = arr.copy()
    n = len(arr)
    num_swaps = int(n * percent)
    for _ in range(num_swaps):
        i, j = random.sample(range(n), 2)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


# ===============================
# Step 2: Instrumented Quicksort
# ===============================
global_metrics = {}


def reset_metrics():
    global global_metrics
    global_metrics = {
        "max_depth": 0,  # maximum recursion depth reached
        "pivot_balance": []  # store balance ratios for each partition call
    }


def instrumented_quicksort(arr, depth=0):
    """
    A quicksort implementation using the first element as the pivot.
    Records:
      - Maximum recursion depth reached.
      - For each partition, records the balance:
        ratio = min(len(left), len(right)) / (len(arr)-1)
    """
    global_metrics["max_depth"] = max(global_metrics["max_depth"], depth)

    if len(arr) <= 1:
        return arr

    pivot = arr[0]
    left = [x for x in arr[1:] if x < pivot]
    right = [x for x in arr[1:] if x >= pivot]

    if len(arr) > 1:
        balance = min(len(left), len(right)) / (len(arr) - 1)
        global_metrics["pivot_balance"].append(balance)

    return instrumented_quicksort(left, depth + 1) + [pivot] + instrumented_quicksort(right, depth + 1)


# ===============================
# Step 3: Running Experiments
# ===============================
def run_experiment(n, noise_levels, trials=10):
    results = {"noise": [], "runtime": [], "max_depth": [], "avg_balance": []}

    for noise in noise_levels:
        trial_runtimes = []
        trial_depths = []
        trial_balances = []

        for _ in range(trials):
            sorted_arr = generate_sorted_array(n)
            test_arr = perturb_array(sorted_arr, noise)

            reset_metrics()

            start_time = time.time()
            instrumented_quicksort(test_arr)
            end_time = time.time()

            trial_runtimes.append(end_time - start_time)
            trial_depths.append(global_metrics["max_depth"])

            if global_metrics["pivot_balance"]:
                avg_balance = sum(global_metrics["pivot_balance"]) / len(global_metrics["pivot_balance"])
            else:
                avg_balance = 0
            trial_balances.append(avg_balance)

        results["noise"].append(noise * 100)
        results["runtime"].append(np.mean(trial_runtimes))
        results["max_depth"].append(np.mean(trial_depths))
        results["avg_balance"].append(np.mean(trial_balances))

        print(f"Noise: {noise * 100:.1f}%, Avg. Runtime: {results['runtime'][-1]:.5f}s, "
              f"Avg. Max Depth: {results['max_depth'][-1]:.2f}, "
              f"Avg. Pivot Balance: {results['avg_balance'][-1]:.3f}")

    return results


# ===============================
# Step 4: Plot the Results
# ===============================
def plot_results(results):
    noise = results["noise"]

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(noise, results["runtime"], marker='o')
    plt.xlabel("Noise Level (%)")
    plt.ylabel("Average Runtime (s)")
    plt.title("Runtime vs. Noise Level")

    plt.subplot(1, 3, 2)
    plt.plot(noise, results["max_depth"], marker='o', color='green')
    plt.xlabel("Noise Level (%)")
    plt.ylabel("Average Max Recursion Depth")
    plt.title("Recursion Depth vs. Noise Level")

    plt.subplot(1, 3, 3)
    plt.plot(noise, results["avg_balance"], marker='o', color='red')
    plt.xlabel("Noise Level (%)")
    plt.ylabel("Average Pivot Balance")
    plt.title("Pivot Balance vs. Noise Level")

    plt.tight_layout()
    plt.show()


# ===============================
# Main: Set parameters and run the experiment
# ===============================
if __name__ == "__main__":
    n = 1000  # Size of the array
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    trials = 10

    results = run_experiment(n, noise_levels, trials)
    plot_results(results)
