import random
import time
import sys

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
      if mid >= low and mid <=high:
          return mid
      else:
          return low # Should only happen if low == high


  # Identify the three values and their original indices
  first_val = arr[low]
  mid_val = arr[mid]
  last_val = arr[high]

  # Perform comparisons to find the median
  # Counting comparisons explicitly here
  comparison_counter.increment(3) # Max 3 comparisons needed

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

  i = low - 1 # Index of smaller element

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

def quicksort_recursive(arr, low, high, pivot_strategy, comparison_counter):
  """
  Recursive Quicksort function.
  """
  if low < high:
    # 1. Choose pivot index based on strategy
    if pivot_strategy == 'first':
      pivot_index = get_pivot_index_first(arr, low, high)
    elif pivot_strategy == 'last':
      pivot_index = get_pivot_index_last(arr, low, high)
    elif pivot_strategy == 'middle':
      pivot_index = get_pivot_index_middle(arr, low, high)
    elif pivot_strategy == 'median_of_three':
      pivot_index = get_pivot_index_median_of_three(arr, low, high, comparison_counter)
    else:
      raise ValueError("Invalid pivot strategy")

    # 2. Partition the array
    pi = partition(arr, low, high, pivot_index, comparison_counter)

    # 3. Recursively sort the sub-arrays
    quicksort_recursive(arr, low, pi - 1, pivot_strategy, comparison_counter)
    quicksort_recursive(arr, pi + 1, high, pivot_strategy, comparison_counter)

# --- Main Quicksort Function Wrapper ---

def quicksort(arr, pivot_strategy='last'):
    """
    Sorts the array using Quicksort with the specified pivot strategy.
    Returns the number of comparisons and the execution time.
    """
    if not arr:
        return 0, 0.0

    arr_copy = arr[:] # Work on a copy to allow multiple runs on same initial data
    comparison_counter = Counter()

    start_time = time.perf_counter()
    quicksort_recursive(arr_copy, 0, len(arr_copy) - 1, pivot_strategy, comparison_counter)
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    comparisons = comparison_counter.get_count()

    # Sanity check: verify if sorted (optional, adds overhead)
    # assert all(arr_copy[i] <= arr_copy[i+1] for i in range(len(arr_copy)-1)), "Array not sorted!"

    return comparisons, execution_time

# --- Experiment Setup ---

def run_experiment():
    """
    Runs the Quicksort experiment for different array sizes and pivot strategies.
    """
    array_sizes = [100, 500, 1000, 2500, 5000] # Adjust sizes as needed
    # Add 10000, but might require increasing recursion depth limit `sys.setrecursionlimit`
    # array_sizes = [100, 500, 1000, 2500, 5000, 10000]
    pivot_strategies = ['first', 'last', 'middle', 'median_of_three']
    num_runs = 3 # Number of runs per size/strategy pair for averaging (optional)

    print("-" * 60)
    print("Quicksort Performance Comparison")
    print(f"Array Sizes: {array_sizes}")
    print(f"Pivot Strategies: {pivot_strategies}")
    print(f"Number of runs per setting: {num_runs}")
    print("-" * 60)
    print("{:<12} {:<20} {:<15} {:<15}".format("Size", "Pivot Strategy", "Avg Compares", "Avg Time (s)"))
    print("-" * 60)

    results = {} # Store detailed results if needed

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
                results[size][strategy] = {'comparisons': [], 'times': []}
                total_comparisons = 0
                total_time = 0.0

                print(f"  Running Strategy: {strategy}...")
                for run in range(num_runs):
                    # Important: Use a fresh copy for each run and strategy!
                    array_to_sort = base_random_array[:]

                    try:
                        comparisons, exec_time = quicksort(array_to_sort, pivot_strategy=strategy)
                        results[size][strategy]['comparisons'].append(comparisons)
                        results[size][strategy]['times'].append(exec_time)
                        total_comparisons += comparisons
                        total_time += exec_time
                    except RecursionError:
                         print(f"    Run {run+1}/{num_runs}: FAILED (Recursion Depth Exceeded)")
                         # Store NaN or None to indicate failure
                         results[size][strategy]['comparisons'].append(float('nan'))
                         results[size][strategy]['times'].append(float('nan'))
                         total_comparisons = float('nan') # Mark average as NaN if any run fails
                         total_time = float('nan')
                         break # Stop runs for this strategy/size if one fails
                    # print(f"    Run {run+1}/{num_runs}: Compares={comparisons}, Time={exec_time:.6f}s") # Verbose output


                if num_runs > 0 and not (isinstance(total_comparisons, float) and total_comparisons != total_comparisons): # Check if not NaN
                     avg_comparisons = total_comparisons / num_runs
                     avg_time = total_time / num_runs
                     print("{:<12} {:<20} {:<15.1f} {:<15.6f}".format(size, strategy, avg_comparisons, avg_time))
                else:
                     print("{:<12} {:<20} {:<15} {:<15}".format(size, strategy, "N/A (Failed)", "N/A (Failed)"))


        except MemoryError:
            print(f"\n--- Array Size: {size} ---")
            print("  FAILED: MemoryError generating the array. Skipping larger sizes.")
            break # Stop if we run out of memory

    print("-" * 60)
    print("Experiment Complete.")
    print("-" * 60)

    # You can access detailed results from the 'results' dictionary if needed
    # Example: results[1000]['median_of_three']['times'] would give a list of times for that run

# --- Main Execution ---
if __name__ == "__main__":
    run_experiment()