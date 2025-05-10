import random
import time
import pandas as pd
import matplotlib.pyplot as plt

# --- Core Miller-Rabin Logic ---
def _is_strong_probable_prime(n, a, d, s):
    """Checks if n is a strong probable prime to base a. (n - 1) = 2^s * d."""
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

# --- Deterministic Variant Definitions ---
DETERMINISTIC_VARIANTS = [
    {
        "name": "Det. Base {2}",
        "bases": [2],
        "limit": 2047, # Smallest composite spsp to base 2 is 2047
        "short_name": "Det.{2}"
    },
    {
        "name": "Det. Bases {2,3}",
        "bases": [2, 3],
        "limit": 1373653,
        "short_name": "Det.{2,3}"
    },
    {
        "name": "Det. Bases {2,3,5}",
        "bases": [2, 3, 5],
        "limit": 25326001,
        "short_name": "Det.{2,3,5}"
    },
    {
        "name": "Det. Jaeschke {2,7,61}",
        "bases": [2, 7, 61],
        "limit": 4759123141, # Correct for n < limit
        "short_name": "Det.Jaeschke"
    },
    # Example: A common set for 64-bit numbers (first 12 primes), limit approx 2^64
    # For benchmark purposes, we'll test over a smaller common range.
    # Actual smallest k-th prime spsp values grow very quickly.
    # This one is more for showing a larger set of bases rather than its full limit in benchmark.
    {
        "name": "Det. First 7 Primes",
        "bases": [2, 3, 5, 7, 11, 13, 17],
        "limit": 341550071728321,
        "short_name": "Det.First7"
    }
]

def miller_rabin_deterministic_variant(n, variant_config):
    """
    Tests n using a specific deterministic Miller-Rabin variant configuration.
    variant_config is a dictionary: {"name": str, "bases": list, "limit": int}
    """
    bases_to_test = variant_config["bases"]
    variant_upper_limit = variant_config["limit"]
    variant_name = variant_config["name"]

    # Universal initial checks
    if n <= 1: return "composite"
    if n == 2 or n == 3: return "prime" # All variants should classify these as prime
    if n % 2 == 0: return "composite"
    
    if n >= variant_upper_limit:
        return f"out of range for {variant_name.split(' ')[1]}" # e.g. "Det. Base {2}" -> "{2}"

    # Check if n is one of the (prime) bases for this specific variant
    if n in bases_to_test: # Assuming bases are prime
        # Small prime bases like 2,3,5,7,61 are prime.
        # We need a list of known small primes to verify if a base itself is prime,
        # or assume bases provided in DETERMINISTIC_VARIANTS are indeed prime.
        # For this implementation, we assume bases listed are primes.
        return "prime"

    # Pre-check divisibility by prime bases (optimization)
    for b in bases_to_test:
        if b < n and n % b == 0: # If n is divisible by one of its prime bases (and not the base itself)
            return "composite"

    # Miller-Rabin core logic
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    for a in bases_to_test:
        if n == a: continue # Already handled if n is a base
        if not _is_strong_probable_prime(n, a, d, s):
            return "composite"
    return "prime"

def miller_rabin_probabilistic(n, k):
    """Probabilistic Miller-Rabin test."""
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
def run_benchmark(benchmark_test_limit=10000): # Reduced for quicker demonstration
    """
    Runs benchmarks for multiple deterministic variants and probabilistic,
    returns results for tabulation and graphing.
    """
    print(f"\nRunning benchmarks for numbers up to {benchmark_test_limit-1}...")
    numbers_to_test = list(range(benchmark_test_limit))
    benchmark_results_list = []

    # Test each deterministic variant
    for variant_config in DETERMINISTIC_VARIANTS:
        # Ensure benchmark_test_limit is valid for this variant's deterministic guarantee
        # For comparing base set overhead, we test on a common range.
        # The 'limit' in variant_config is its max deterministic range.
        # Here, we are testing performance on numbers_to_test, assuming it's within reasonable bounds.
        
        start_time = time.time()
        for num in numbers_to_test:
            # We pass the full config so it can handle "out of range" if num exceeds its specific limit
            # However, for the benchmark itself, we are testing up to benchmark_test_limit
            if num < variant_config["limit"]: # Only test if num is within this variant's deterministic range
                 miller_rabin_deterministic_variant(num, variant_config)
            # else: # Or count how many were out of range for this variant's guarantee
                 # pass # Or just skip, affecting total operations if limits differ vastly
        time_taken = time.time() - start_time
        
        # Note for N: if benchmark_test_limit > variant_config["limit"],
        # the variant didn't test all numbers up to benchmark_test_limit deterministically.
        # For fair comparison of base set overhead, benchmark_test_limit should be <= smallest variant limit.
        # Or, report effective N tested for each.
        # Here, we assume benchmark_test_limit is chosen s.t. it's meaningful for most variants.
        effective_n_tested = min(benchmark_test_limit, variant_config["limit"])

        benchmark_results_list.append({
            "Test Type": variant_config["name"],
            "Bases Used": len(variant_config["bases"]),
            "Deterministic Limit": f"< {variant_config['limit']}",
            "N Benchmarked Up To": benchmark_test_limit -1, # The common range chosen for benchmark
            "Time (s)": float(f"{time_taken:.4f}")
        })

    # Probabilistic Miller-Rabin for comparison
    for k_prob in [3, 7]: # k=3 for few bases, k=7 to match "First 7 Primes"
        start_time = time.time()
        for num in numbers_to_test:
            miller_rabin_probabilistic(num, k_prob)
        time_taken = time.time() - start_time
        benchmark_results_list.append({
            "Test Type": f"Probabilistic (k={k_prob})",
            "Bases Used": k_prob,
            "Deterministic Limit": "N/A",
            "N Benchmarked Up To": benchmark_test_limit-1,
            "Time (s)": float(f"{time_taken:.4f}")
        })
    return benchmark_results_list

# --- Main Execution ---
if __name__ == "__main__":
    print("Miller-Rabin Primality Test - Deterministic Variants Comparison")
    print("=" * 60)
    print("Note: Output will be saved to CSV and Excel files.")
    print("Ensure 'pandas' and 'openpyxl' (for Excel) are installed: pip install pandas openpyxl matplotlib")

    # 1. Example Results for Specific Numbers from Different Deterministic Variants
    print("\n1. Processing Test Examples for Deterministic Variants...")
    example_numbers_to_test = [1, 2, 17, 22, 61, 97, 2046, 2047, 1373652, 1373653]
    example_results_data = []
    
    header_row = ["Number"] + [v["short_name"] for v in DETERMINISTIC_VARIANTS]
    example_results_data.append(header_row)

    for num in example_numbers_to_test:
        row = [num]
        for variant_config in DETERMINISTIC_VARIANTS:
            # Only run if num is reasonably within the variant's intended scope for this example table
            # The function itself handles n >= limit, but for table readability:
            if num < variant_config["limit"] or num < 5000: # Show for smaller numbers, or if clearly in range
                row.append(miller_rabin_deterministic_variant(num, variant_config))
            else:
                row.append("skip/too large") # To keep example table focused
        example_results_data.append(row)

    # Convert to DataFrame for saving (using first row as header)
    df_example_results = pd.DataFrame(example_results_data[1:], columns=example_results_data[0])

    csv_examples_filename = "deterministic_variant_examples.csv"
    df_example_results.to_csv(csv_examples_filename, index=False)
    print(f"Example test results saved to '{csv_examples_filename}'")


    # 2. Efficiency Comparison Benchmark
    print("\n2. Processing Efficiency Comparison Benchmark...")
    # BENCHMARK_LIMIT should ideally be less than the smallest 'limit' of variants you want to fully compare.
    # 2046 is the largest number < Det.{2} limit. Let's use a slightly larger common range for more tests.
    # e.g., 10,000. All listed variants are deterministic well beyond this.
    BENCHMARK_UPPER_BOUND_N = 20000 # Test numbers 0 to 19999
    benchmark_data_list = run_benchmark(benchmark_test_limit=BENCHMARK_UPPER_BOUND_N)
    df_benchmark_results = pd.DataFrame(benchmark_data_list)

    csv_bench_filename = "deterministic_benchmark_comparison.csv"
    df_benchmark_results.to_csv(csv_bench_filename, index=False)
    print(f"Benchmark results saved to '{csv_bench_filename}'")

    # Save both to a single Excel file
    excel_filename = "miller_rabin_deterministic_analysis.xlsx"
    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df_example_results.to_excel(writer, sheet_name='Variant Examples', index=False)
            df_benchmark_results.to_excel(writer, sheet_name='Benchmark Comparison', index=False)
        print(f"All tabular results saved to '{excel_filename}'")
    except Exception as e:
        print(f"Could not write to Excel file '{excel_filename}': {e}")


    # 3. Benchmark Graph
    print("\n3. Generating benchmark graph (deterministic_benchmark_graph.png)...")
    
    # Filter for graph: only include variants fully tested up to BENCHMARK_UPPER_BOUND_N - 1
    # Or simply plot all collected benchmark data. The "N Benchmarked Up To" helps clarify.
    graph_labels = [item["Test Type"] for item in benchmark_data_list]
    graph_times = [item["Time (s)"] for item in benchmark_data_list]
    graph_bases_used = [str(item["Bases Used"]) for item in benchmark_data_list]

    fig, ax = plt.subplots(figsize=(14, 8)) # Adjusted figure size
    bars = ax.bar(graph_labels, graph_times, color=plt.cm.viridis(range(len(graph_labels))))
    
    ax.set_ylabel('Time Taken (seconds)')
    ax.set_xlabel('Miller-Rabin Test Variant')
    ax.set_title(f'Miller-Rabin Performance (Numbers up to {BENCHMARK_UPPER_BOUND_N-1})')
    plt.xticks(rotation=25, ha="right", fontsize=9)
    
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}s\n({graph_bases_used[i]} bases)', 
                 va='bottom', ha='center', fontsize=8)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    try:
        plt.savefig("deterministic_benchmark_graph.png")
        print("Benchmark graph saved as 'deterministic_benchmark_graph.png'")
    except Exception as e:
        print(f"Could not save benchmark graph: {e}")

    print("\n=" * 60)
    print("Script finished. Check the generated CSV, Excel, and PNG files.")