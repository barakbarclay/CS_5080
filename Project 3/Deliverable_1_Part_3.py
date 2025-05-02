import math
import random
# import matplotlib.pyplot as plt # No longer needed for static plots
import numpy as np
import time
import plotly.graph_objects as go # Using Plotly for interactive plots
import os # To save HTML file

# --- Prime Number Generation (Reused) ---

def primes_sieve(limit):
    """
    Generates prime numbers up to a given limit using the Sieve of Eratosthenes.
    Args:
        limit (int): The upper bound (exclusive) for prime generation.
    Returns:
        list: A list of prime numbers less than limit.
    """
    # Ensure limit is at least 2 for sieve logic
    limit = max(2, int(limit))
    prime = [True] * limit
    prime[0] = prime[1] = False # 0 and 1 are not prime

    # Optimization: only need to check up to sqrt(limit)
    for i in range(2, int(math.sqrt(limit)) + 1):
        if prime[i]:
            # Mark multiples of i as not prime, starting from i*i
            for multiple in range(i*i, limit, i):
                prime[multiple] = False

    prime_numbers = [i for i, is_prime in enumerate(prime) if is_prime]
    return prime_numbers

def get_primes_in_range(n_min, n_max):
    """
    Finds prime numbers within a specific range [n_min, n_max).
    Args:
        n_min (int): The lower bound (inclusive).
        n_max (int): The upper bound (exclusive).
    Returns:
        list: A list of prime numbers p such that n_min <= p < n_max.
            Returns an empty list if n_max <= n_min.
    """
    if n_max <= n_min:
        return []
    # Generate primes up to n_max using the sieve
    all_primes_up_to_max = primes_sieve(n_max)
    # Filter primes that are >= n_min
    primes_in_range = [p for p in all_primes_up_to_max if p >= n_min]
    return primes_in_range

# --- Adversary K and Theoretical Rate (Reused for baseline) ---

def calculate_adversary_k_and_theoretical_rate(n, primes_between_n_n2):
    """
    Calculates the adversary's number K and the theoretical false positive rate
    for the *original* scheme (primes in (n, n^2)). This serves as the baseline.
    Args:
        n (int): The number of bits.
        primes_between_n_n2 (list): List of primes p such that n < p < n^2.
    Returns:
        tuple: (K, theoretical_rate, num_adversary_primes)
               K (int): The adversary's number (product of smallest primes in (n, n^2) < 2^n - 1).
               theoretical_rate (float): The theoretical false positive rate (adversary primes / total primes in (n, n^2)).
               num_adversary_primes (int): Number of primes used to construct K.
    """
    if not primes_between_n_n2:
        return 1, 0.0, 0 # No primes, K=1, rate=0

    limit = pow(2, n) - 1 # The maximum value for x or y
    if limit <= 0: # Handle n=0 or n=1, or potential overflow for very large n
        # print(f"Warning: 2^{n}-1 calculation resulted in non-positive limit.")
        limit = (1 << n) - 1 # Bit shift alternative, safer for large n

    K = 1
    num_adversary_primes = 0
    # Primes are assumed sorted if coming directly from sieve filtering
    sorted_primes = primes_between_n_n2

    for p in sorted_primes:
        # Check for potential overflow before multiplication, though Python handles large ints
        if p <= 0: continue # Should not happen for primes > n
        # Check if K * p would exceed limit
        # Use multiplication check that avoids large intermediate product if K or p is huge
        if K > limit // p:
             break
        next_K = K * p
        # Ensure next_K is still positive and less than limit
        if 0 < next_K < limit:
            K = next_K
            num_adversary_primes += 1
        else:
            # Stop if the next product exceeds the limit or causes overflow issues
            break

    total_primes = len(primes_between_n_n2)
    # Theoretical rate based on the original full range of primes
    theoretical_rate = num_adversary_primes / total_primes if total_primes > 0 else 0.0

    # Return K=1 if no adversary primes could be found (e.g., if limit is very small)
    if num_adversary_primes == 0:
        K = 1

    return K, theoretical_rate, num_adversary_primes

# --- Original Empirical Rate Calculation (Reused) ---

def calculate_empirical_rate_original(K, primes_between_n_n2, num_trials=10000):
    """
    Calculates the empirical false positive rate for the original scheme.
    Uses primes from the full range (n, n^2).
    Args:
        K (int): The adversary's number (calculated based on original range).
        primes_between_n_n2 (list): List of primes p such that n < p < n^2.
        num_trials (int): The number of random trials to run.
    Returns:
        float: The empirical false positive rate.
    """
    if not primes_between_n_n2 or num_trials <= 0:
        return 0.0

    false_positives = 0
    x = 0 # Alice's number
    y = K # Bob's number (adversary's choice)

    for _ in range(num_trials):
        # Alice chooses one random prime from the original range
        p = random.choice(primes_between_n_n2)
        h = x % p # Alice computes hash (always 0 for x=0)
        g = y % p # Bob computes hash

        if g == h: # Check for collision (false positive when x != y)
            false_positives += 1

    empirical_rate = false_positives / num_trials
    return empirical_rate

# --- Countermeasure 1: Two-Fingerprint Scheme ---

def calculate_empirical_rate_two_fingerprints(K, primes_between_n_n2, num_trials=10000):
    """
    Calculates the empirical false positive rate for the two-fingerprint countermeasure.
    Uses primes from the full range (n, n^2).
    Args:
        K (int): The adversary's number.
        primes_between_n_n2 (list): List of primes p such that n < p < n^2.
        num_trials (int): The number of random trials to run.
    Returns:
        float: The empirical false positive rate for the countermeasure.
    """
    # Need at least two primes to choose from for the countermeasure
    if len(primes_between_n_n2) < 2 or num_trials <= 0:
        return 0.0

    false_positives = 0
    x = 0 # Alice's number
    y = K # Bob's number (adversary's choice)

    for _ in range(num_trials):
        # Alice chooses two *distinct* random primes from the original range
        p1, p2 = random.sample(primes_between_n_n2, 2)

        # Alice computes hashes
        h1 = x % p1 # Always 0 for x=0
        h2 = x % p2 # Always 0 for x=0

        # Bob computes hashes
        g1 = y % p1
        g2 = y % p2

        # False positive only if BOTH match
        if g1 == h1 and g2 == h2:
            false_positives += 1

    empirical_rate = false_positives / num_trials
    return empirical_rate

# --- Countermeasure 2: Restricted Prime Range Scheme ---

def calculate_empirical_rate_restricted_primes(K, primes_restricted, num_trials=10000):
    """
    Calculates the empirical false positive rate for the restricted prime range countermeasure.
    Alice only chooses primes p > n*ln(n).
    Args:
        K (int): The adversary's number (calculated based on original range).
        primes_restricted (list): List of primes p such that n*ln(n) < p < n^2.
        num_trials (int): The number of random trials to run.
    Returns:
        float: The empirical false positive rate for this countermeasure.
    """
    # Need at least one prime in the restricted range
    if not primes_restricted or num_trials <= 0:
        return 0.0

    false_positives = 0
    x = 0 # Alice's number
    y = K # Bob's number (adversary's choice)

    for _ in range(num_trials):
        # Alice chooses one random prime from the *restricted* range
        p = random.choice(primes_restricted)
        h = x % p # Alice computes hash (always 0 for x=0)
        g = y % p # Bob computes hash

        if g == h: # Check for collision (false positive when x != y)
            false_positives += 1

    empirical_rate = false_positives / num_trials
    return empirical_rate


# --- Main Experiment Runner ---

def run_comparison_experiment(n_start=6, n_end=200, num_trials_empirical=10000):
    """
    Runs the comparison experiment for n from n_start to n_end,
    calculating theoretical original rate, empirical original rate,
    and empirical rates for both countermeasures.

    Args:
        n_start (int): The starting value of n.
        n_end (int): The ending value of n (inclusive).
        num_trials_empirical (int): Number of trials for empirical calculations.

    Returns:
        tuple: (n_values, theoretical_rates, empirical_rates_orig,
                empirical_rates_cm1, empirical_rates_cm2)
               Lists containing the results for plotting.
    """
    n_values = []
    theoretical_rates = []
    empirical_rates_orig = []
    empirical_rates_cm1 = [] # Countermeasure 1: Two fingerprints
    empirical_rates_cm2 = [] # Countermeasure 2: Restricted primes

    print(f"Running comparison experiment for n from {n_start} to {n_end}...")
    start_time = time.time()

    # Pre-calculate primes up to the maximum possible n^2
    max_n_squared = n_end * n_end
    print(f"Pre-calculating primes up to {max_n_squared}...")
    # Ensure sieve limit is reasonable
    sieve_limit = max(max_n_squared, n_start + 2) # Need at least n+1 primes
    all_primes = primes_sieve(sieve_limit)
    print(f"Prime calculation finished. Found {len(all_primes)} primes.")

    # Use a set for faster prime lookups if needed, but list comprehension is often fine
    # prime_set = set(all_primes)

    for n in range(n_start, n_end + 1):
        if n <= 1: continue # Skip n=0, 1 as log(n) is problematic

        n_squared = n * n
        # Original range lower bound (exclusive): n
        lower_bound_orig = n + 1
        # Restricted range lower bound (exclusive): n*ln(n)
        # Use ceiling to get the first integer strictly greater than n*ln(n)
        try:
            lower_bound_restricted = math.ceil(n * math.log(n)) + 1
        except ValueError: # math.log(1) is 0, handle n=1 if not skipped
             print(f"n={n}: Skipping due to log calculation issue.")
             continue

        upper_bound = n_squared # Upper bound (exclusive) for both

        # Filter primes for the original full range (n, n^2)
        primes_n_n2 = [p for p in all_primes if lower_bound_orig <= p < upper_bound]

        # Filter primes for the restricted range (n*ln(n), n^2)
        primes_restricted = [p for p in primes_n_n2 if p >= lower_bound_restricted]

        # --- Checks for sufficient primes ---
        # Need at least 1 prime for original and restricted schemes
        if not primes_n_n2:
             print(f"n={n}: No primes found in ({n}, {n_squared}). Skipping.")
             continue
        # Need at least 2 primes for the two-fingerprint scheme
        can_run_cm1 = len(primes_n_n2) >= 2
        # Need at least 1 prime for the restricted prime scheme
        can_run_cm2 = len(primes_restricted) >= 1

        # --- Calculations ---
        # 1. Theoretical Calculation (Baseline using original range)
        K, theo_rate, num_adv_primes = calculate_adversary_k_and_theoretical_rate(n, primes_n_n2)

        # 2. Empirical Calculation (Original Scheme)
        emp_rate_orig = calculate_empirical_rate_original(K, primes_n_n2, num_trials=num_trials_empirical)

        # 3. Empirical Calculation (Countermeasure 1: Two Fingerprints)
        emp_rate_cm1 = calculate_empirical_rate_two_fingerprints(K, primes_n_n2, num_trials=num_trials_empirical) if can_run_cm1 else float('nan')

        # 4. Empirical Calculation (Countermeasure 2: Restricted Primes)
        emp_rate_cm2 = calculate_empirical_rate_restricted_primes(K, primes_restricted, num_trials=num_trials_empirical) if can_run_cm2 else float('nan')

        # Store results
        n_values.append(n)
        theoretical_rates.append(theo_rate)
        empirical_rates_orig.append(emp_rate_orig)
        empirical_rates_cm1.append(emp_rate_cm1)
        empirical_rates_cm2.append(emp_rate_cm2)

        # Print progress periodically
        if n == 6 or n == 10 or n % 20 == 0 or n == n_end:
             cm1_str = f"{emp_rate_cm1:.4f}" if not np.isnan(emp_rate_cm1) else "N/A"
             cm2_str = f"{emp_rate_cm2:.4f}" if not np.isnan(emp_rate_cm2) else "N/A"
             print(f"n={n}: Primes(orig)={len(primes_n_n2)}, Primes(restr)={len(primes_restricted)}, Adv Primes={num_adv_primes}, Theo Rate={theo_rate:.4f}, Emp Orig={emp_rate_orig:.4f}, Emp CM1={cm1_str}, Emp CM2={cm2_str}")

    end_time = time.time()
    print(f"\nExperiment finished in {end_time - start_time:.2f} seconds.")

    return n_values, theoretical_rates, empirical_rates_orig, empirical_rates_cm1, empirical_rates_cm2

# --- Interactive Visualization (Plotly) ---

def create_interactive_plot(n_values, theoretical_rates, empirical_rates_orig,
                            empirical_rates_cm1, empirical_rates_cm2,
                            filename="fingerprint_comparison_multi_cm.html"):
    """
    Creates an interactive Plotly plot comparing the rates and saves it as HTML.
    Includes traces for original, theoretical, and both countermeasures.

    Args:
        n_values (list): List of n values.
        theoretical_rates (list): List of theoretical rates (original scheme baseline).
        empirical_rates_orig (list): List of empirical rates (original scheme).
        empirical_rates_cm1 (list): List of empirical rates (countermeasure 1: two prints).
        empirical_rates_cm2 (list): List of empirical rates (countermeasure 2: restricted primes).
        filename (str): Name of the HTML file to save the plot.
    """
    if not n_values:
        print("No data to plot.")
        return

    fig = go.Figure()

    # Add Theoretical Rate (Original Baseline) trace
    fig.add_trace(go.Scatter(
        x=n_values, y=theoretical_rates, mode='lines', name='Theoretical Rate (Original Baseline)',
        line=dict(dash='solid', color='grey'),
        hovertemplate='n=%{x}<br>Theoretical Rate=%{y:.6f}<extra></extra>'
    ))

    # Add Empirical Rate (Original) trace
    fig.add_trace(go.Scatter(
        x=n_values, y=empirical_rates_orig, mode='lines+markers', name='Empirical Rate (Original)',
        marker=dict(symbol='x', size=5, color='blue'), line=dict(dash='dash', color='blue'),
        hovertemplate='n=%{x}<br>Empirical Rate (Orig)=%{y:.6f}<extra></extra>'
    ))

    # Add Empirical Rate (Countermeasure 1: Two Fingerprints) trace
    # Use connectgaps=False to handle potential NaNs if too few primes existed for some n
    fig.add_trace(go.Scatter(
        x=n_values, y=empirical_rates_cm1, mode='lines+markers', name='Empirical Rate (CM1: Two Prints)',
        marker=dict(symbol='diamond', size=5, color='red'), line=dict(dash='dot', color='red'),
        connectgaps=False,
        hovertemplate='n=%{x}<br>Empirical Rate (CM1)=%{y:.6f}<extra></extra>'
    ))

    # Add Empirical Rate (Countermeasure 2: Restricted Primes) trace
    # Use connectgaps=False
    fig.add_trace(go.Scatter(
        x=n_values, y=empirical_rates_cm2, mode='lines+markers', name='Empirical Rate (CM2: Restricted Primes p>n*ln(n))',
        marker=dict(symbol='star', size=5, color='green'), line=dict(dash='dashdot', color='green'),
        connectgaps=False,
        hovertemplate='n=%{x}<br>Empirical Rate (CM2)=%{y:.6f}<extra></extra>'
    ))

    # Update layout for clarity and interactivity
    fig.update_layout(
        title='Fingerprinting False Positive Rates vs. n (Adversarial Case with Countermeasures)',
        xaxis_title='n (Number of Bits)',
        yaxis_title='False Positive Rate (Log Scale)',
        yaxis_type="log",  # Use log scale for y-axis
        hovermode="x unified", # Show hover info for all traces at once
        legend_title_text='Scheme Type',
        template='plotly_white' # Use a clean template
    )

    # Save the plot to an HTML file
    try:
        fig.write_html(filename)
        print(f"Interactive plot saved to '{os.path.abspath(filename)}'")
        # Depending on the environment, you might open the file automatically:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(filename))
    except Exception as e:
        print(f"Error saving plot to HTML: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # Run the comparison experiment (adjust n_end and num_trials as needed)
    n_vals, theo_rates, emp_rates_o, emp_rates_c1, emp_rates_c2 = run_comparison_experiment(
        n_start=6,   # Start from 6 as n*ln(n) grows
        n_end=1000,   # Keep n_end moderate for reasonable runtime
        num_trials_empirical=10000 # Adjust for accuracy/speed
    )

    # Create and save the interactive plot with both countermeasures
    create_interactive_plot(n_vals, theo_rates, emp_rates_o, emp_rates_c1, emp_rates_c2)
