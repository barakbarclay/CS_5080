import random
import time
from tabulate import tabulate # For tables
import matplotlib.pyplot as plt # For graphs

# --- Core Miller-Rabin Logic (largely unchanged) ---
def _is_strong_probable_prime(n, a, d, s):
    """
    Checks if n is a strong probable prime to base a.
    (n - 1) = 2^s * d where d is odd.
    Helper function for Miller-Rabin.
    """
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True

    for _ in range(s - 1):
        x = pow(x, 2, n)
        if x == n - 1:
            return True
        if x == 1: # Non-trivial square root of 1
            return False
    return False # Composite if loop finishes

def miller_rabin_deterministic_jaeschke(n):
    """
    Deterministic Miller-Rabin test for n < 4,759,123,141
    using bases {2, 7, 61}.
    Returns "prime" or "composite" or "out of range".
    """
    if n <= 1: return "composite"
    # Bases themselves and other small primes
    if n in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}:
        return "prime"
    
    if n % 2 == 0 or n % 3 == 0 or n % 5 == 0: # Quick check for divisibility by small primes
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
        if n == a: # Should have been caught by the 'in' check above, but for safety.
            return "prime" 
        # If a base divides n, and n is not that base itself (n > a), then n is composite.
        # This is implicitly handled if _is_strong_probable_prime is robust for a|n,
        # or by the fact that for these small prime bases, n%a==0 would have been caught
        # by the earlier n%2, n%3, n%5 checks if a was 2,3,5.
        # For a=7 or a=61, if n is a multiple, it's composite.
        # A direct check `if n % a == 0: return "composite"` could be added here if n wasn't already filtered.
        # The _is_strong_probable_prime test assumes gcd(a,n)=1 for its guarantees usually,
        # but practically if pow(a,d,n) works, it means 'a' is not 0 mod n.
        # If n%a == 0, and a < n, then n is composite. The initial checks cover a=2,3,5.
        # For a=7 or a=61, if n % a == 0, it's composite.
        # Let's ensure this is robust. A simple way:
        if n % a == 0 and n != a : # If a base divides n and n is larger
             return "composite"

        if not _is_strong_probable_prime(n, a, d, s):
            return "composite"
    return "prime"

def miller_rabin_probabilistic(n, k):
    """
    Probabilistic Miller-Rabin test.
    Returns "probably prime" or "composite".
    """
    if n <= 1: return "composite"
    if n == 2 or n == 3: return "probably prime"
    if n % 2 == 0: return "composite"

    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    for _ in range(k):
        # Ensure 'a' is chosen correctly for small n if they reach here
        if n <= 4: # Should not happen due to prior checks, but defensive
            a = 2
        elif n == 5: # randrange(2,4)
            a = random.choice([2,3])
        else:
            a = random.randrange(2, n - 1)
            
        if not _is_strong_probable_prime(n, a, d, s):
            return "composite"
    return "probably prime"

# --- Benchmarking Function ---
def run_benchmark(test_limit=100000):
    """
    Runs benchmarks and returns results for tabulation and graphing.
    """
    print(f"\nRunning benchmarks for numbers up to {test_limit-1}...")
    numbers_to_test = list(range(test_limit))
    benchmark_results = []

    # Deterministic Jaeschke
    start_time = time.time()
    for num in numbers_to_test:
        miller_rabin_deterministic_jaeschke(num)
    time_taken = time.time() - start_time
    benchmark_results.append({
        "Test Type": "Deterministic (Jaeschke {2,7,61})",
        "N": test_limit,
        "Time (s)": f"{time_taken:.4f}"
    })

    # Probabilistic Miller-Rabin with varying k
    for k_prob in [3, 5, 10]:
        start_time = time.time()
        for num in numbers_to_test:
            miller_rabin_probabilistic(num, k_prob)
        time_taken = time.time() - start_time
        benchmark_results.append({
            "Test Type": f"Probabilistic (k={k_prob})",
            "N": test_limit,
            "Time (s)": f"{time_taken:.4f}"
        })
    return benchmark_results

# --- Main Execution ---
if __name__ == "__main__":
    print("Miller-Rabin Primality Test Output Demonstration")
    print("=" * 50)

    # 1. Deterministic Test Examples Table
    print("\n1. Deterministic Miller-Rabin Test Examples (Jaeschke for n < 4,759,123,141)")
    deterministic_test_cases = [
        {"Number": 1, "Known Status": "Composite (Not Prime)"},
        {"Number": 2, "Known Status": "Prime"},
        {"Number": 17, "Known Status": "Prime"},
        {"Number": 22, "Known Status": "Composite (2*11)"},
        {"Number": 61, "Known Status": "Prime (a base)"},
        {"Number": 97, "Known Status": "Prime"},
        {"Number": 2047, "Known Status": "Composite (23*89)"},
        {"Number": 4759123123, "Known Status": "Composite (17*279948419)"}, # Corrected
        {"Number": 4759123140, "Known Status": "Composite (Even)"},
        {"Number": 4759123141, "Known Status": "N/A (Test Limit Boundary)"}
    ]

    table_data_det = []
    headers_det = ["Number", "Known Status", "Test Result (Deterministic Jaeschke)"]
    for case in deterministic_test_cases:
        num = case["Number"]
        result = miller_rabin_deterministic_jaeschke(num)
        table_data_det.append([num, case["Known Status"], result])
    
    print(tabulate(table_data_det, headers=headers_det, tablefmt="grid"))

    # 2. Efficiency Comparison (Table and Graph)
    print("\n2. Efficiency Comparison")
    # You can change test_limit for benchmark if needed, e.g., 10000 for faster run during dev
    benchmark_data = run_benchmark(test_limit=100000) 
    
    # Benchmark Table
    print("\nBenchmark Results Table:")
    # benchmark_data is already a list of dicts, tabulate can handle that directly
    print(tabulate(benchmark_data, headers="keys", tablefmt="grid"))

    # Benchmark Graph
    print("\nGenerating benchmark graph (benchmark_comparison.png)...")
    labels = [item["Test Type"] for item in benchmark_data]
    times = [float(item["Time (s)"]) for item in benchmark_data]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, times, color=['blue', 'green', 'orange', 'red'])
    plt.ylabel('Time Taken (seconds)')
    plt.xlabel('Miller-Rabin Test Variant')
    plt.title(f'Miller-Rabin Performance Comparison (N < {benchmark_data[0]["N"]})')
    plt.xticks(rotation=15, ha="right") # Rotate labels for better readability
    plt.tight_layout() # Adjust layout to make room for labels

    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}s', va='bottom', ha='center')


    try:
        plt.savefig("miller_rabin_benchmark_comparison.png")
        print("Benchmark graph saved as 'miller_rabin_benchmark_comparison.png'")
        # plt.show() # Uncomment to display the plot interactively if a GUI environment is available
    except Exception as e:
        print(f"Could not save or show plot: {e}")
        print("Matplotlib might require a GUI backend or specific configuration to display plots directly.")

    print("\n=" * 50)
    print("Note: To run this script, you may need to install 'tabulate' and 'matplotlib'.")
    print("You can install them using: pip install tabulate matplotlib")