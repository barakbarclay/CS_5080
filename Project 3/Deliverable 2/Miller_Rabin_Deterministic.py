import random
import time
# Removed: from tabulate import tabulate
import pandas as pd # For CSV and Excel output
import matplotlib.pyplot as plt # For graphs

# --- Core Miller-Rabin Logic (largely unchanged) ---
def _is_strong_probable_prime(n, a, d, s):
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = pow(x, 2, n)
        if x == n - 1:
            return True
        if x == 1:
            return False
    return False

def miller_rabin_deterministic_jaeschke(n):
    if n <= 1: return "composite"
    if n in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}:
        return "prime"
    if n % 2 == 0 or n % 3 == 0 or n % 5 == 0:
        return "composite"
    if n >= 4759123141:
        return "out of range for Jaeschke set"

    bases = [2, 7, 61]
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in bases:
        if n == a: return "prime"
        if n % a == 0 and n != a: return "composite"
        if not _is_strong_probable_prime(n, a, d, s):
            return "composite"
    return "prime"

def miller_rabin_probabilistic(n, k):
    if n <= 1: return "composite"
    if n == 2 or n == 3: return "probably prime"
    if n % 2 == 0: return "composite"
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(k):
        if n <= 4: a = 2
        elif n == 5: a = random.choice([2,3])
        else: a = random.randrange(2, n - 1)
        if not _is_strong_probable_prime(n, a, d, s):
            return "composite"
    return "probably prime"

# --- Benchmarking Function ---
def run_benchmark(test_limit=100000):
    print(f"\nRunning benchmarks for numbers up to {test_limit-1}...")
    numbers_to_test = list(range(test_limit))
    benchmark_results_list_of_dicts = [] # Changed variable name for clarity

    # Deterministic Jaeschke
    start_time = time.time()
    for num in numbers_to_test:
        miller_rabin_deterministic_jaeschke(num)
    time_taken = time.time() - start_time
    benchmark_results_list_of_dicts.append({
        "Test Type": "Deterministic (Jaeschke {2,7,61})",
        "N (Up to)": test_limit -1, # Clarify N refers to numbers tested up to N-1
        "Time (s)": float(f"{time_taken:.4f}") # Store as float for easier processing
    })

    # Probabilistic Miller-Rabin with varying k
    for k_prob in [3, 5, 10]:
        start_time = time.time()
        for num in numbers_to_test:
            miller_rabin_probabilistic(num, k_prob)
        time_taken = time.time() - start_time
        benchmark_results_list_of_dicts.append({
            "Test Type": f"Probabilistic (k={k_prob})",
            "N (Up to)": test_limit -1,
            "Time (s)": float(f"{time_taken:.4f}")
        })
    return benchmark_results_list_of_dicts

# --- Main Execution ---
if __name__ == "__main__":
    print("Miller-Rabin Primality Test Output Demonstration")
    print("=" * 50)
    print("Note: Output will be saved to CSV and Excel files.")
    print("Ensure you have 'pandas' and 'openpyxl' installed: pip install pandas openpyxl")


    # 1. Deterministic Test Examples
    print("\n1. Processing Deterministic Miller-Rabin Test Examples...")
    deterministic_test_cases = [
        {"Number": 1, "Known Status": "Composite (Not Prime)"},
        {"Number": 2, "Known Status": "Prime"},
        {"Number": 17, "Known Status": "Prime"},
        {"Number": 22, "Known Status": "Composite (2*11)"},
        {"Number": 61, "Known Status": "Prime (a base)"},
        {"Number": 97, "Known Status": "Prime"},
        {"Number": 2047, "Known Status": "Composite (23*89)"},
        {"Number": 4759123123, "Known Status": "Composite (17*279948419)"},
        {"Number": 4759123140, "Known Status": "Composite (Even)"},
        {"Number": 4759123141, "Known Status": "N/A (Test Limit Boundary)"}
    ]

    results_for_df_det = []
    for case in deterministic_test_cases:
        num = case["Number"]
        result = miller_rabin_deterministic_jaeschke(num)
        results_for_df_det.append({
            "Number": num,
            "Known Status": case["Known Status"],
            "Test Result (Deterministic Jaeschke)": result
        })
    
    df_deterministic_results = pd.DataFrame(results_for_df_det)
    
    # Save deterministic results to CSV
    csv_det_filename = "deterministic_primality_tests_results.csv"
    df_deterministic_results.to_csv(csv_det_filename, index=False)
    print(f"Deterministic test results saved to '{csv_det_filename}'")

    # 2. Efficiency Comparison
    print("\n2. Processing Efficiency Comparison Benchmark...")
    # You can change test_limit for benchmark if needed, e.g., 10000 for faster run during dev
    benchmark_data_list_of_dicts = run_benchmark(test_limit=100000) 
    df_benchmark_results = pd.DataFrame(benchmark_data_list_of_dicts)

    # Save benchmark results to CSV
    csv_bench_filename = "benchmark_results.csv"
    df_benchmark_results.to_csv(csv_bench_filename, index=False)
    print(f"Benchmark results saved to '{csv_bench_filename}'")

    # Save both results to a single Excel file with different sheets
    excel_filename = "miller_rabin_all_results.xlsx"
    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df_deterministic_results.to_excel(writer, sheet_name='Deterministic Tests', index=False)
            df_benchmark_results.to_excel(writer, sheet_name='Benchmark Comparison', index=False)
        print(f"All results saved to '{excel_filename}'")
    except Exception as e:
        print(f"Could not write to Excel file '{excel_filename}': {e}")
        print("Make sure 'openpyxl' is installed: pip install openpyxl")


    # 3. Benchmark Graph (remains the same, saves to PNG)
    print("\n3. Generating benchmark graph (benchmark_comparison.png)...")
    labels = [item["Test Type"] for item in benchmark_data_list_of_dicts]
    times = [item["Time (s)"] for item in benchmark_data_list_of_dicts] # Already float

    plt.figure(figsize=(12, 7)) # Adjusted figure size
    bars = plt.bar(labels, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']) # Example colors
    plt.ylabel('Time Taken (seconds)')
    plt.xlabel('Miller-Rabin Test Variant')
    plt.title(f'Miller-Rabin Performance Comparison (Numbers up to {benchmark_data_list_of_dicts[0]["N (Up to)"]})')
    plt.xticks(rotation=15, ha="right")
    
    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}s', va='bottom', ha='center', fontsize=9)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Add a light grid for y-axis
    plt.tight_layout()

    try:
        plt.savefig("miller_rabin_benchmark_comparison.png")
        print("Benchmark graph saved as 'miller_rabin_benchmark_comparison.png'")
        # plt.show() # Uncomment to display the plot interactively
    except Exception as e:
        print(f"Could not save or show plot: {e}")
        print("Matplotlib might require a GUI backend or specific configuration.")

    print("\n=" * 50)
    print("Script finished. Check the generated CSV, Excel, and PNG files.")