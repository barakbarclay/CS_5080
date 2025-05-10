import math
import random
import numpy as np
import time
import plotly.graph_objects as go
import os

# --- Prime Number Generation (Reused) ---
def primes_sieve(limit):
    """
    Generates prime numbers up to a given limit using the Sieve of Eratosthenes.
    """
    limit = max(2, int(limit)) # Ensure limit is at least 2
    prime = [True] * limit
    if limit > 0: prime[0] = False
    if limit > 1: prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if prime[i]:
            for multiple in range(i*i, limit, i):
                prime[multiple] = False
    return [i for i, is_prime in enumerate(prime) if is_prime]

# --- Adversary K Construction ---
def calculate_k_for_adversary(n_bits, available_primes_for_k, adversary_name=""):
    """
    Calculates the adversary's number K using a specific pool of primes.
    K is the product of the smallest primes from available_primes_for_k
    such that K < 2^n_bits - 1.

    Args:
        n_bits (int): The number of bits for the numbers x and y.
        available_primes_for_k (list): Sorted list of primes the adversary can use.
        adversary_name (str): Optional name for logging.

    Returns:
        tuple: (K, num_factors_in_K)
               K (int): The adversary's number.
               num_factors_in_K (int): Number of prime factors used to construct K.
    """
    if not available_primes_for_k:
        # print(f"Adversary {adversary_name}: No primes available to build K for n={n_bits}.")
        return 1, 0

    # Calculate the upper limit for K
    # Using (1 << n_bits) - 1 is robust for large n_bits
    limit_k = (1 << n_bits) - 1
    if limit_k <= 0 : # Handles n_bits = 0 or very small n_bits where 2^n-1 is not positive
        # print(f"Adversary {adversary_name}: Limit for K is not positive for n={n_bits}. K=1.")
        return 1,0

    K = 1
    num_factors_in_K = 0
    # available_primes_for_k should be sorted (smallest first)
    for p in available_primes_for_k:
        if p <= 0: continue # Should not happen with primes
        # Check if K * p would exceed limit_k or cause overflow before multiplication
        # K > limit_k // p is a safer check against overflow if K or p is very large
        if K > limit_k // p: # If K * p would be too large
            break
        
        next_K = K * p
        # Ensure next_K is still positive (it should be if K and p are) and within limit
        if 0 < next_K < limit_k:
            K = next_K
            num_factors_in_K += 1
        else:
            # This break is important if next_K somehow becomes invalid
            # or if K was already close to limit_k and p is large.
            break
    
    # If no factors could be added (e.g., all primes in pool are > limit_k)
    if num_factors_in_K == 0:
        K = 1 # K remains 1 if no primes could be multiplied

    # print(f"Adversary {adversary_name} for n={n_bits}: K={K if K < 1e6 else '>1e6'}, Factors={num_factors_in_K}, Limit K={limit_k}")
    return K, num_factors_in_K

# --- Empirical Rate Calculations ---

def calculate_empirical_rate_single_prime(K_adversary, alice_prime_pool, num_trials=10000):
    """
    Empirical rate for schemes where Alice picks one prime.
    Args:
        K_adversary (int): The adversary's number y.
        alice_prime_pool (list): List of primes Alice chooses from.
        num_trials (int): Number of simulation trials.
    Returns:
        float: Empirical false positive rate.
    """
    if not alice_prime_pool or num_trials <= 0:
        return 0.0

    false_positives = 0
    x = 0 # Alice's number
    y = K_adversary # Bob's number

    for _ in range(num_trials):
        p_alice = random.choice(alice_prime_pool)
        h_alice = x % p_alice # Always 0 for x=0
        g_bob = y % p_alice

        if g_bob == h_alice:
            false_positives += 1
    return false_positives / num_trials

def calculate_empirical_rate_two_primes(K_adversary, alice_prime_pool, num_trials=10000):
    """
    Empirical rate for schemes where Alice picks two distinct primes.
    Args:
        K_adversary (int): The adversary's number y.
        alice_prime_pool (list): List of primes Alice chooses from.
        num_trials (int): Number of simulation trials.
    Returns:
        float: Empirical false positive rate.
    """
    if len(alice_prime_pool) < 2 or num_trials <= 0:
        return 0.0 # Not enough primes for Alice to pick two distinct ones

    false_positives = 0
    x = 0 # Alice's number
    y = K_adversary # Bob's number

    for _ in range(num_trials):
        p1_alice, p2_alice = random.sample(alice_prime_pool, 2)
        
        h1_alice = x % p1_alice # Always 0
        h2_alice = x % p2_alice # Always 0

        g1_bob = y % p1_alice
        g2_bob = y % p2_alice

        if g1_bob == h1_alice and g2_bob == h2_alice:
            false_positives += 1
    return false_positives / num_trials

# --- Main Experiment Runner ---
def run_adaptive_adversary_experiment(n_start=6, n_end=100, num_trials_empirical=10000):
    """
    Runs the experiment with an adversary that adapts K based on Alice's prime pool.
    """
    results = {
        'n_values': [],
        'theo_rate_orig_baseline': [], # Theoretical FP for original scheme, original K
        'emp_rate_orig': [],           # Original scheme (1 prime from (n, n^2)), original K
        'emp_rate_cm1': [],            # CM1: 2 primes from (n, n^2), original K
        'emp_rate_cm2': [],            # CM2: 1 prime from (n*ln(n), n^2), K adapted to this range
        'emp_rate_cm3': []             # CM3: 2 primes from (n*ln(n), n^2), K adapted to this range
    }

    print(f"Running adaptive adversary experiment for n from {n_start} to {n_end}...")
    overall_start_time = time.time()

    # Pre-calculate all primes up to the max possible n_end^2
    max_n_squared = n_end * n_end
    sieve_limit = max(max_n_squared, n_start + 2) # Ensure sieve limit is reasonable
    print(f"Pre-calculating primes up to {sieve_limit}...")
    all_available_primes = primes_sieve(sieve_limit)
    print(f"Prime calculation finished. Found {len(all_available_primes)} primes.")

    for n in range(n_start, n_end + 1):
        current_n_time = time.time()
        if n <= 1: continue # log(n) undefined or zero for n=1

        n_squared = n * n
        
        # Define prime pools for Alice
        # Pool 1: Original range (n, n^2)
        alice_pool_orig = [p for p in all_available_primes if (n < p < n_squared)]
        
        # Pool 2: Restricted range (n*ln(n), n^2)
        try:
            lower_bound_restricted = math.ceil(n * math.log(n)) # Primes > n*ln(n)
        except ValueError:
            print(f"n={n}: Skipping due to math.log error (likely n=1).")
            continue
        
        alice_pool_restricted = [p for p in all_available_primes if (lower_bound_restricted < p < n_squared)]

        # --- Check if pools are adequate ---
        if not alice_pool_orig:
            # print(f"n={n}: Alice's original prime pool is empty. Skipping.")
            continue
        
        # --- Adversary builds K for Original Scheme & CM1 ---
        # Adversary uses primes > n (i.e., Alice's original pool) to build K_orig
        K_orig, num_factors_K_orig = calculate_k_for_adversary(n, alice_pool_orig, "Original")
        
        # --- Adversary builds K for CM2 & CM3 ---
        # Adversary adapts and uses primes > n*ln(n) (Alice's restricted pool) to build K_restr
        K_restr, num_factors_K_restr = calculate_k_for_adversary(n, alice_pool_restricted, "RestrictedPool")

        # --- Calculate Rates ---
        # 1. Theoretical Rate (Original Scheme Baseline)
        #    Based on K_orig and Alice choosing from alice_pool_orig
        theo_rate_baseline = num_factors_K_orig / len(alice_pool_orig) if len(alice_pool_orig) > 0 else 0.0
        
        # 2. Empirical Rate - Original Scheme
        #    Alice picks 1 prime from alice_pool_orig, adversary uses K_orig
        emp_rate_orig = calculate_empirical_rate_single_prime(K_orig, alice_pool_orig, num_trials_empirical)
        
        # 3. Empirical Rate - CM1 (Two primes from original pool)
        #    Alice picks 2 primes from alice_pool_orig, adversary uses K_orig
        emp_rate_cm1 = calculate_empirical_rate_two_primes(K_orig, alice_pool_orig, num_trials_empirical)
        
        # 4. Empirical Rate - CM2 (One prime from restricted pool)
        #    Alice picks 1 prime from alice_pool_restricted, adversary uses K_restr (adapted K)
        emp_rate_cm2 = calculate_empirical_rate_single_prime(K_restr, alice_pool_restricted, num_trials_empirical)
        
        # 5. Empirical Rate - CM3 (Two primes from restricted pool)
        #    Alice picks 2 primes from alice_pool_restricted, adversary uses K_restr (adapted K)
        emp_rate_cm3 = calculate_empirical_rate_two_primes(K_restr, alice_pool_restricted, num_trials_empirical)

        # Store results
        results['n_values'].append(n)
        results['theo_rate_orig_baseline'].append(theo_rate_baseline)
        results['emp_rate_orig'].append(emp_rate_orig)
        results['emp_rate_cm1'].append(emp_rate_cm1)
        results['emp_rate_cm2'].append(emp_rate_cm2)
        results['emp_rate_cm3'].append(emp_rate_cm3)

        if n % 10 == 0 or n == n_end or n == n_start: # Print progress
            print(f"n={n:3} | "
                  f"P_orig={len(alice_pool_orig):4} (K_orig factors={num_factors_K_orig:2}) | "
                  f"P_restr={len(alice_pool_restricted):4} (K_restr factors={num_factors_K_restr:2}) | "
                  f"Rates: Theo={theo_rate_baseline:.4f} Orig={emp_rate_orig:.4f} CM1={emp_rate_cm1:.4f} CM2={emp_rate_cm2:.4f} CM3={emp_rate_cm3:.4f} "
                  f"({time.time() - current_n_time:.2f}s)")

    total_time = time.time() - overall_start_time
    print(f"\nExperiment finished in {total_time:.2f} seconds.")
    return results

# --- Interactive Visualization (Plotly) ---
def create_adaptive_interactive_plot(results, filename="fingerprint_adaptive_adv_plot.html"):
    """
    Creates an interactive Plotly plot for the adaptive adversary scenario.
    """
    if not results['n_values']:
        print("No data to plot.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results['n_values'], y=results['theo_rate_orig_baseline'], mode='lines', name='Theoretical Rate (Original Baseline)',
        line=dict(dash='solid', color='grey', width=1.5),
        hovertemplate='n=%{x}<br>Theo Rate (Orig Baseline)=%{y:.6f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=results['n_values'], y=results['emp_rate_orig'], mode='lines+markers', name='Emp. Orig (1 prime from (n, n²), K_orig)',
        marker=dict(symbol='x', size=6, color='blue'), line=dict(dash='dash', color='blue'),
        hovertemplate='n=%{x}<br>Emp Rate (Orig)=%{y:.6f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=results['n_values'], y=results['emp_rate_cm1'], mode='lines+markers', name='Emp. CM1 (2 primes from (n, n²), K_orig)',
        marker=dict(symbol='diamond', size=6, color='red'), line=dict(dash='dot', color='red'),
        hovertemplate='n=%{x}<br>Emp Rate (CM1)=%{y:.6f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=results['n_values'], y=results['emp_rate_cm2'], mode='lines+markers', name='Emp. CM2 (1 prime from (nln n, n²), K_restr)',
        marker=dict(symbol='star', size=6, color='green'), line=dict(dash='dashdot', color='green'),
        hovertemplate='n=%{x}<br>Emp Rate (CM2 - Adapted K)=%{y:.6f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=results['n_values'], y=results['emp_rate_cm3'], mode='lines+markers', name='Emp. CM3 (2 primes from (nln n, n²), K_restr)',
        marker=dict(symbol='triangle-up', size=6, color='purple'), line=dict(dash='longdash', color='purple'),
        hovertemplate='n=%{x}<br>Emp Rate (CM3 - Adapted K)=%{y:.6f}<extra></extra>'
    ))

    fig.update_layout(
        title='Fingerprinting: False Positive Rates with Adaptive Adversary',
        xaxis_title='n (Number of Bits)',
        yaxis_title='False Positive Rate (Log Scale)',
        yaxis_type="log",
        hovermode="x unified",
        legend_title_text='Scheme & Adversary Strategy',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    try:
        fig.write_html(filename)
        print(f"Interactive plot saved to '{os.path.abspath(filename)}'")
    except Exception as e:
        print(f"Error saving plot to HTML: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Adjust n_end and num_trials as needed for performance/accuracy
    # n_end=100 and 10000 trials can take a few minutes.
    experiment_results = run_adaptive_adversary_experiment(
        n_start=10, # Start higher to ensure n*ln(n) is distinct enough from n
        n_end=1000,
        num_trials_empirical=10000
    )
    create_adaptive_interactive_plot(experiment_results)

