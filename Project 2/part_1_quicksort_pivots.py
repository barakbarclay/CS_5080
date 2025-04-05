import random
import time
import sys
import numpy as np

# Increase recursion depth limit for larger arrays (use with caution)
# sys.setrecursionlimit(2000) # You might need this for sizes >= 5000


# --- Comparison Counter ---
# Using a class to make the counter mutable and easily passed around.
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self, amount=1):
        self.count += amount

    def get_count(self):
        return self.count

    def reset(self):
        self.count = 0


# --- Pivot Selection Strategies ---


def get_pivot_index_first(arr, low, high):
    """Selects the first element as the pivot."""
    return low


def get_pivot_index_last(arr, low, high):
    """Selects the last element as the pivot."""
    return high


def get_pivot_index_middle(arr, low, high):
    """Selects the middle element as the pivot."""
    return (low + high) // 2


def get_pivot_index_median_of_three(arr, low, high, comparison_counter):
    """Selects the median of the first, middle, and last elements."""
    mid = (low + high) // 2
    # Use indices and values to avoid index out of bounds on small arrays
    candidates = {}
    if low <= high:
        candidates[low] = arr[low]
    if mid >= low and mid <= high and mid != low:
        candidates[mid] = arr[mid]
    if high >= low and high != low and high != mid:
        candidates[high] = arr[high]

    # If fewer than 3 distinct elements/indices, default to a simpler strategy (e.g., middle or first)
    if len(candidates) < 3:
        # comparison_counter.increment(0) # No comparisons if < 3 elements considered
        # Defaulting to middle or first if middle isn't available
        if mid >= low and mid <= high:
            return mid
        else:
            return low  # Should only happen if low == high

    # Identify the three values and their original indices
    first_val = arr[low]
    mid_val = arr[mid]
    last_val = arr[high]

    # Perform comparisons to find the median
    # Counting comparisons explicitly here
    comparison_counter.increment(3)  # Max 3 comparisons needed

    if (first_val <= mid_val <= last_val) or (last_val <= mid_val <= first_val):
        return mid
    elif (mid_val <= first_val <= last_val) or (last_val <= first_val <= mid_val):
        return low
    else:
        return high


# --- Partition Function (Lomuto partition scheme adapted) ---


def partition(arr, low, high, pivot_index, comparison_counter):
    """
    Partitions the array around the pivot element.
    Moves the chosen pivot to the end first for consistent partitioning logic.
    Returns the final index of the pivot after partitioning.
    """
    pivot_value = arr[pivot_index]
    # Move pivot to end
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]

    i = low - 1  # Index of smaller element

    for j in range(low, high):
        # --- Comparison happens here ---
        comparison_counter.increment()
        if arr[j] < pivot_value:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    # Place pivot in its correct final position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# --- Quicksort Implementation ---


def quicksort_recursive(
    arr, low, high, pivot_strategy, comparison_counter, metrics_tracker, depth
):
    """
    Recursive Quicksort function.
    Tracks depth and calculates pivot balance.
    """
    # Update max depth reached
    metrics_tracker["max_depth"] = max(metrics_tracker.get("max_depth", 0), depth)

    if low < high:
        # 1. Choose pivot index based on strategy
        if pivot_strategy == "first":
            pivot_index = get_pivot_index_first(arr, low, high)
        elif pivot_strategy == "last":
            pivot_index = get_pivot_index_last(arr, low, high)
        elif pivot_strategy == "middle":
            pivot_index = get_pivot_index_middle(arr, low, high)
        elif pivot_strategy == "median_of_three":
            pivot_index = get_pivot_index_median_of_three(
                arr, low, high, comparison_counter
            )
        else:
            raise ValueError("Invalid pivot strategy")

        # 2. Partition the array
        pi = partition(arr, low, high, pivot_index, comparison_counter)

        # 3. Calculate and record pivot balance for this step
        left_size = pi - low
        right_size = high - pi
        current_size_minus_1 = high - low  # Denominator for balance calc

        if (
            current_size_minus_1 > 0
        ):  # Avoid division by zero for subarrays of size <= 1
            balance = min(left_size, right_size) / current_size_minus_1
            metrics_tracker.setdefault("balance_list", []).append(
                balance
            )  # Use setdefault for safety

        # 4. Recursively sort the sub-arrays
        quicksort_recursive(
            arr,
            low,
            pi - 1,
            pivot_strategy,
            comparison_counter,
            metrics_tracker,
            depth + 1,
        )
        quicksort_recursive(
            arr,
            pi + 1,
            high,
            pivot_strategy,
            comparison_counter,
            metrics_tracker,
            depth + 1,
        )


# --- Main Quicksort Function Wrapper ---


def quicksort(arr, pivot_strategy="last"):
    """
    Sorts the array using Quicksort with the specified pivot strategy.
    Returns:
        - comparisons (int): Total number of comparisons.
        - execution_time (float): Wall clock time for the sort.
        - max_depth (int): Maximum recursion depth reached.
        - avg_balance (float): Average pivot balance across all partitions.
                                Returns 0 if no partitions occurred (e.g., array size < 2).
    """
    if not arr:
        return 0, 0.0, 0, 0.0  # Comparisons, time, depth, balance

    if len(arr) == 1:
        return (
            0,
            0.0,
            0,
            0.0,
        )  # No comparisons, time, depth, or balance for single element

    arr_copy = arr[:]  # Work on a copy
    comparison_counter = Counter()
    # Initialize metrics tracker for this run
    metrics_tracker = {"max_depth": 0, "balance_list": []}

    start_time = time.perf_counter()
    quicksort_recursive(
        arr_copy,
        0,
        len(arr_copy) - 1,
        pivot_strategy,
        comparison_counter,
        metrics_tracker,
        depth=0,
    )
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    comparisons = comparison_counter.get_count()
    max_depth = metrics_tracker.get("max_depth", 0)

    # Calculate average balance
    balance_list = metrics_tracker.get("balance_list", [])
    if balance_list:
        avg_balance = sum(balance_list) / len(balance_list)
    else:
        avg_balance = (
            0.0  # Define avg balance as 0 if no partitions happened (size < 2)
        )
        # Alternatively, you could return float('nan') here if preferred
        # avg_balance = float('nan')

    # Sanity check: verify if sorted (optional, adds overhead)
    # assert all(arr_copy[i] <= arr_copy[i+1] for i in range(len(arr_copy)-1)), "Array not sorted!"

    return comparisons, execution_time, max_depth, avg_balance


# --- Experiment Setup ---


def run_experiment():
    """
    Runs the Quicksort experiment for different array sizes and pivot strategies.
    Reports comparisons, time, max depth, and average balance.
    """
    array_sizes = [100, 500, 1000, 2500, 5000]  # Adjust sizes as needed
    # Add 10000, but might require increasing recursion depth limit `sys.setrecursionlimit`
    # array_sizes = [100, 500, 1000, 2500, 5000, 10000]
    pivot_strategies = ["first", "last", "middle", "median_of_three"]
    num_runs = 3  # Number of runs per size/strategy pair for averaging (optional)

    print("-" * 88)
    print("Quicksort Performance Comparison")
    print(f"Array Sizes: {array_sizes}")
    print(f"Pivot Strategies: {pivot_strategies}")
    print(f"Number of runs per setting: {num_runs}")
    print("-" * 88)
    print(
        "{:<10} {:<20} {:<15} {:<15} {:<12} {:<15}".format(
            "Size",
            "Pivot Strategy",
            "Avg Compares",
            "Avg Time (s)",
            "Avg Depth",
            "Avg Balance",
        )
    )
    print("-" * 88)

    results = {}  # Store detailed results if needed

    for size in array_sizes:
        results[size] = {}
        # Generate base random array for this size
        # Using range 0 to 10*size to allow duplicates but still have variety
        try:
            # Generate a list of random integers.
            # Using a large range to reduce the chance of all elements being the same
            # in small arrays, which can be a worst-case scenario for some pivots.
            base_random_array = [random.randint(0, size * 10) for _ in range(size)]
            print(f"\n--- Array Size: {size} ---")

            for strategy in pivot_strategies:
                # Store lists for all metrics across runs
                results[size][strategy] = {
                    "comparisons": [],
                    "times": [],
                    "depths": [],
                    "balances": [],
                }
                total_comparisons = 0
                total_time = 0.0
                total_depth = 0
                total_balance = 0.0  # Sum of averages to compute final average
                failed_run = False

                print(f"  Running Strategy: {strategy}...")
                for run in range(num_runs):
                    # Important: Use a fresh copy for each run and strategy!
                    array_to_sort = base_random_array[:]

                    try:
                        # Capture all returned metrics
                        comparisons, exec_time, max_depth, avg_balance = quicksort(
                            array_to_sort, pivot_strategy=strategy
                        )
                        results[size][strategy]["comparisons"].append(comparisons)
                        results[size][strategy]["times"].append(exec_time)
                        results[size][strategy]["depths"].append(max_depth)
                        results[size][strategy]["balances"].append(avg_balance)

                        total_comparisons += comparisons
                        total_time += exec_time
                        total_depth += max_depth
                        # Check avg_balance is not NaN before adding if using NaN option
                        # if not np.isnan(avg_balance):
                        #    total_balance += avg_balance
                        total_balance += avg_balance  # Assuming 0.0 for size < 2 runs

                    except RecursionError:
                        print(
                            f"    Run {run+1}/{num_runs}: FAILED (Recursion Depth Exceeded)"
                        )
                        # Store NaN or None to indicate failure
                        results[size][strategy]["comparisons"].append(float("nan"))
                        results[size][strategy]["times"].append(float("nan"))
                        results[size][strategy]["depths"].append(float("nan"))
                        results[size][strategy]["balances"].append(float("nan"))
                        failed_run = True
                        break  # Stop runs for this strategy/size if one fails
                    # print(f"    Run {run+1}/{num_runs}: Compares={comparisons}, Time={exec_time:.6f}s") # Verbose output

                if failed_run:
                    # Print failure indication for the average results
                    print(
                        "{:<10} {:<20} {:<15} {:<15} {:<12} {:<15}".format(
                            size, strategy, "N/A (Failed)", "N/A", "N/A", "N/A"
                        )
                    )
                elif num_runs > 0:
                    avg_comparisons = total_comparisons / num_runs
                    avg_time = total_time / num_runs
                    avg_depth = total_depth / num_runs
                    # Average of the average balances from each run
                    avg_avg_balance = total_balance / num_runs

                    # Print the averaged results
                    print(
                        "{:<10} {:<20} {:<15.1f} {:<15.6f} {:<12.1f} {:<15.3f}".format(
                            size,
                            strategy,
                            avg_comparisons,
                            avg_time,
                            avg_depth,
                            avg_avg_balance,
                        )
                    )

        except MemoryError:
            print(f"\n--- Array Size: {size} ---")
            print("  FAILED: MemoryError generating the array. Skipping larger sizes.")
            break  # Stop if we run out of memory

    print("-" * 88)
    print("Experiment Complete.")
    print("-" * 88)

    # Example: Access detailed results: results[1000]['median_of_three']['depths']


# --- Main Execution ---
if __name__ == "__main__":
    run_experiment()
