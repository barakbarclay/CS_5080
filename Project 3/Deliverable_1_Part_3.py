import math
import random
import matplotlib.pyplot as plt
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
    if limit < 2:
        return []
    prime = [True] * limit
    prime[0] = prime[1] = False # 0 and 1 are not prime

    for i in range(2, int(math.sqrt(limit)) + 1):
        if prime[i]:
            # Mark multiples of i as not prime
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

# --- Adversary K and Theoretical Rate (Reused) ---

def calculate_adversary_k_and_theoretical_rate(n, primes_between_n_n2):
    """
    Calculates the adversary's number K and the theoretical false positive rate
    for the *original* scheme.
    Args:
        n (int): The number of bits.
        primes_between_n_n2 (list): List of primes p such that n < p < n^2.
    Returns:
        tuple: (K, theoretical_rate, num_adversary_primes)
               K (int): The adversary's number (product of smallest primes < 2^n - 1).
               theoretical_rate (float): The theoretical false positive rate.
               num_adversary_primes (int): Number of primes used to construct K.
    """
    if not primes_between_n_n2:
        return 1, 0.0, 0 # No primes, K=1, rate=0

    limit = pow(2, n) - 1 # The maximum value for x or y
    if limit < 0: # Handle potential overflow for very large n
        print(f"Warning: 2^{n}-1 might be too large, relying on Python's arbitrary precision.")
        limit = (1 << n) - 1 # Bit shift alternative

    K = 1
    num_adversary_primes = 0
    sorted_primes = primes_between_n_n2 # Assumes input is sorted

    for p in sorted_primes:
        if K > limit // p: # Check if K * p would exceed limit
             break
        next_K = K * p
        if next_K < limit:
            K = next_K
            num_adversary_primes += 1
        else:
            break # Stop if the next product exceeds the limit

    total_primes = len(primes_between_n_n2)
    theoretical_rate = num_adversary_primes / total_primes if total_primes > 0 else 0.0

    return K, theoretical_rate, num_adversary_primes

# --- Original Empirical Rate Calculation (Reused) ---

def calculate_empirical_rate_original(K, primes_between_n_n2, num_trials=10000):
    """
    Calculates the empirical false positive rate for the original scheme.
    Args:
        K (int): The adversary's number.
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
        # Alice chooses one random prime
        p = random.choice(primes_between_n_n2)
        h = x % p # Alice computes hash (always 0 for x=0)
        g = y % p # Bob computes hash

        if g == h: # Check for collision (false positive when x != y)
            false_positives += 1

    empirical_rate = false_positives / num_trials
    return empirical_rate

# --- Countermeasure: Two-Fingerprint Scheme ---

def calculate_empirical_rate_two_fingerprints(K, primes_between_n_n2, num_trials=10000):
    """
    Calculates the empirical false positive rate for the two-fingerprint countermeasure.
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
        # Alice chooses two *distinct* random primes
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

# --- Main Experiment Runner ---

def run_comparison_experiment(n_start=6, n_end=200, num_trials_empirical=10000):
    """
    Runs the comparison experiment for n from n_start to n_end,
    calculating theoretical original rate, empirical original rate,
    and empirical countermeasure rate.

    Args:
        n_start (int): The starting value of n.
        n_end (int): The ending value of n (inclusive).
        num_trials_empirical (int): Number of trials for empirical calculations.

    Returns:
        tuple: (n_values, theoretical_rates, empirical_rates_orig, empirical_rates_cm)
               Lists containing the results for plotting.
    """
    n_values = []
    theoretical_rates = []
    empirical_rates_orig = []
    empirical_rates_cm = [] # Countermeasure rates

    print(f"Running comparison experiment for n from {n_start} to {n_end}...")
    start_time = time.time()

    # Pre-calculate primes up to the maximum possible n^2
    max_n_squared = n_end * n_end
    print(f"Pre-calculating primes up to {max_n_squared}...")
    all_primes = primes_sieve(max_n_squared)
    print(f"Prime calculation finished. Found {len(all_primes)} primes.")

    for n in range(n_start, n_end + 1):
        n_squared = n * n
        lower_bound = n + 1 # Primes must be > n
        upper_bound = n_squared # Primes must be < n^2

        # Filter primes for the current range (n, n^2)
        primes_n_n2 = [p for p in all_primes if lower_bound <= p < upper_bound]

        if len(primes_n_n2) < 2: # Need at least 2 primes for countermeasure
            print(f"n={n}: Not enough primes ({len(primes_n_n2)}) found between {n+1} and {n_squared}. Skipping.")
            continue

        # 1. Theoretical Calculation (Original Scheme)
        K, theo_rate, num_adv_primes = calculate_adversary_k_and_theoretical_rate(n, primes_n_n2)

        # 2. Empirical Calculation (Original Scheme)
        emp_rate_orig = calculate_empirical_rate_original(K, primes_n_n2, num_trials=num_trials_empirical)

        # 3. Empirical Calculation (Countermeasure Scheme)
        emp_rate_cm = calculate_empirical_rate_two_fingerprints(K, primes_n_n2, num_trials=num_trials_empirical)

        n_values.append(n)
        theoretical_rates.append(theo_rate)
        empirical_rates_orig.append(emp_rate_orig)
        empirical_rates_cm.append(emp_rate_cm)

        if n % 20 == 0 or n == n_end: # Print progress periodically
             print(f"n={n}: Primes={len(primes_n_n2)}, Adv Primes={num_adv_primes}, Theo Rate={theo_rate:.4f}, Emp Orig={emp_rate_orig:.4f}, Emp CM={emp_rate_cm:.4f}")

    end_time = time.time()
    print(f"\nExperiment finished in {end_time - start_time:.2f} seconds.")

    return n_values, theoretical_rates, empirical_rates_orig, empirical_rates_cm

# --- Interactive Visualization (Plotly) ---

def create_interactive_plot(n_values, theoretical_rates, empirical_rates_orig, empirical_rates_cm, filename="fingerprint_comparison.html"):
    """
    Creates an interactive Plotly plot comparing the rates and saves it as HTML.

    Args:
        n_values (list): List of n values.
        theoretical_rates (list): List of theoretical rates (original scheme).
        empirical_rates_orig (list): List of empirical rates (original scheme).
        empirical_rates_cm (list): List of empirical rates (countermeasure scheme).
        filename (str): Name of the HTML file to save the plot.
    """
    if not n_values:
        print("No data to plot.")
        return

    fig = go.Figure()

    # Add Theoretical Rate (Original) trace
    fig.add_trace(go.Scatter(
        x=n_values,
        y=theoretical_rates,
        mode='lines+markers',
        name='Theoretical Rate (Original)',
        marker=dict(symbol='circle', size=5),
        line=dict(dash='solid'),
        hovertemplate='n=%{x}<br>Theoretical Rate=%{y:.6f}<extra></extra>'
    ))

    # Add Empirical Rate (Original) trace
    fig.add_trace(go.Scatter(
        x=n_values,
        y=empirical_rates_orig,
        mode='lines+markers',
        name='Empirical Rate (Original)',
        marker=dict(symbol='x', size=5),
        line=dict(dash='dash'),
        hovertemplate='n=%{x}<br>Empirical Rate (Orig)=%{y:.6f}<extra></extra>'
    ))

    # Add Empirical Rate (Countermeasure) trace
    fig.add_trace(go.Scatter(
        x=n_values,
        y=empirical_rates_cm,
        mode='lines+markers',
        name='Empirical Rate (Countermeasure)',
        marker=dict(symbol='diamond', size=5),
        line=dict(dash='dot'),
        hovertemplate='n=%{x}<br>Empirical Rate (CM)=%{y:.6f}<extra></extra>'
    ))

    # Update layout for clarity and interactivity
    fig.update_layout(
        title='Fingerprinting False Positive Rates vs. n (Adversarial Case)',
        xaxis_title='n (Number of Bits)',
        yaxis_title='False Positive Rate',
        yaxis_type="log",  # Use log scale for y-axis to see differences better
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
    # Keeping n_end relatively small (e.g., 200) for reasonable runtime.
    n_vals, theo_rates, emp_rates_o, emp_rates_c = run_comparison_experiment(
        n_start=6,
        n_end=1000,
        num_trials_empirical=10000 # Increase for more accuracy, decrease for speed
    )

    # Create and save the interactive plot
    create_interactive_plot(n_vals, theo_rates, emp_rates_o, emp_rates_c)

