import math
import random
import numpy as np
import time
import plotly.graph_objects as go
import os

# This code was written with assistance from Gemini.

# --- Prime Number Generation (Reused) ---
def primes_sieve(limit):
    """
    Generates prime numbers up to a given limit using the Sieve of Eratosthenes.
    """
    limit = max(2, int(limit))  # Ensure limit is at least 2
    prime = [True] * limit
    if limit > 0:
        prime[0] = False
    if limit > 1:
        prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if prime[i]:
            for multiple in range(i * i, limit, i):
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
        return 1, 0

    limit_k = (1 << n_bits) - 1
    if limit_k <= 0:
        return 1, 0

    K = 1
    num_factors_in_K = 0
    for p in available_primes_for_k:  # Assumes sorted smallest first
        if p <= 0:
            continue
        if K > limit_k // p:
            break

        next_K = K * p
        if 0 < next_K < limit_k:
            K = next_K
            num_factors_in_K += 1
        else:
            break

    if num_factors_in_K == 0:
        K = 1
    return K, num_factors_in_K


# --- Empirical Rate Calculations ---


def calculate_empirical_rate_single_prime(
    K_adversary, alice_prime_pool, num_trials=10000
):
    """Empirical rate for schemes where Alice picks one prime."""
    if not alice_prime_pool or num_trials <= 0:
        return 0.0
    false_positives = 0
    x = 0
    y = K_adversary
    for _ in range(num_trials):
        p_alice = random.choice(alice_prime_pool)
        if y % p_alice == x % p_alice:  # x % p_alice is always 0
            false_positives += 1
    return false_positives / num_trials


def calculate_empirical_rate_two_primes(
    K_adversary, alice_prime_pool, num_trials=10000
):
    """Empirical rate for schemes where Alice picks two distinct primes."""
    if len(alice_prime_pool) < 2 or num_trials <= 0:
        return 0.0
    false_positives = 0
    x = 0
    y = K_adversary
    for _ in range(num_trials):
        p1_alice, p2_alice = random.sample(alice_prime_pool, 2)
        if (y % p1_alice == x % p1_alice) and (y % p2_alice == x % p2_alice):
            false_positives += 1
    return false_positives / num_trials


def calculate_empirical_rate_three_primes(
    K_adversary, alice_prime_pool, num_trials=10000
):
    """Empirical rate for schemes where Alice picks three distinct primes."""
    if len(alice_prime_pool) < 3 or num_trials <= 0:
        return 0.0
    false_positives = 0
    x = 0
    y = K_adversary
    for _ in range(num_trials):
        p1_alice, p2_alice, p3_alice = random.sample(alice_prime_pool, 3)
        if (
            (y % p1_alice == x % p1_alice)
            and (y % p2_alice == x % p2_alice)
            and (y % p3_alice == x % p3_alice)
        ):
            false_positives += 1
    return false_positives / num_trials


# --- Main Experiment Runner ---
def run_adaptive_adversary_experiment(n_start=6, n_end=100, num_trials_empirical=10000):
    """Runs the experiment with an adversary that adapts K based on Alice's prime pool."""
    results = {
        "n_values": [],
        "theo_rate_orig_baseline": [],
        "emp_rate_orig": [],
        "emp_rate_cm1": [],
        "emp_rate_cm2": [],
        "emp_rate_cm3": [],
        "emp_rate_cm4": [],  # New: For three primes from restricted pool
    }

    print(f"Running adaptive adversary experiment for n from {n_start} to {n_end}...")
    overall_start_time = time.time()

    max_n_squared = n_end * n_end
    sieve_limit = max(max_n_squared, n_start + 2)
    print(f"Pre-calculating primes up to {sieve_limit}...")
    all_available_primes = primes_sieve(sieve_limit)
    print(f"Prime calculation finished. Found {len(all_available_primes)} primes.")

    for n in range(n_start, n_end + 1):
        current_n_time = time.time()
        if n <= 1:
            continue

        n_squared = n * n
        alice_pool_orig = [p for p in all_available_primes if (n < p < n_squared)]

        try:
            lower_bound_restricted = math.ceil(n * math.log(n))
        except ValueError:
            print(f"n={n}: Skipping due to math.log error.")
            continue
        alice_pool_restricted = [
            p for p in all_available_primes if (lower_bound_restricted < p < n_squared)
        ]

        if not alice_pool_orig:
            continue

        K_orig, num_factors_K_orig = calculate_k_for_adversary(
            n, alice_pool_orig, "OriginalPool"
        )
        K_restr, num_factors_K_restr = calculate_k_for_adversary(
            n, alice_pool_restricted, "RestrictedPool"
        )

        theo_rate_baseline = (
            num_factors_K_orig / len(alice_pool_orig)
            if len(alice_pool_orig) > 0
            else 0.0
        )
        emp_rate_orig = calculate_empirical_rate_single_prime(
            K_orig, alice_pool_orig, num_trials_empirical
        )
        emp_rate_cm1 = calculate_empirical_rate_two_primes(
            K_orig, alice_pool_orig, num_trials_empirical
        )
        emp_rate_cm2 = calculate_empirical_rate_single_prime(
            K_restr, alice_pool_restricted, num_trials_empirical
        )
        emp_rate_cm3 = calculate_empirical_rate_two_primes(
            K_restr, alice_pool_restricted, num_trials_empirical
        )
        # Calculate for CM4: Alice uses 3 primes from restricted pool, adversary uses K_restr
        emp_rate_cm4 = calculate_empirical_rate_three_primes(
            K_restr, alice_pool_restricted, num_trials_empirical
        )

        results["n_values"].append(n)
        results["theo_rate_orig_baseline"].append(theo_rate_baseline)
        results["emp_rate_orig"].append(emp_rate_orig)
        results["emp_rate_cm1"].append(emp_rate_cm1)
        results["emp_rate_cm2"].append(emp_rate_cm2)
        results["emp_rate_cm3"].append(emp_rate_cm3)
        results["emp_rate_cm4"].append(emp_rate_cm4)

        if n % 10 == 0 or n == n_end or n == n_start:
            print(
                f"n={n:3} | "
                f"P_orig={len(alice_pool_orig):4} (K_o factors={num_factors_K_orig:2}) | "
                f"P_restr={len(alice_pool_restricted):4} (K_r factors={num_factors_K_restr:2}) | "
                f"Rates: Theo={theo_rate_baseline:.4f} Orig={emp_rate_orig:.4f} CM1={emp_rate_cm1:.4f} "
                f"CM2={emp_rate_cm2:.4f} CM3={emp_rate_cm3:.4f} CM4={emp_rate_cm4:.4f} "
                f"({time.time() - current_n_time:.2f}s)"
            )

    total_time = time.time() - overall_start_time
    print(f"\nExperiment finished in {total_time:.2f} seconds.")
    return results


# --- Interactive Visualization (Plotly) ---
def create_adaptive_interactive_plot(
    results, filename="fingerprint_adaptive_adv_plot_v2.html"
):
    """Creates an interactive Plotly plot for the adaptive adversary scenario, including CM4."""
    if not results["n_values"]:
        print("No data to plot.")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results["n_values"],
            y=results["theo_rate_orig_baseline"],
            mode="lines",
            name="Theo. Rate (Orig Baseline)",
            line=dict(dash="solid", color="grey", width=1.5),
            hovertemplate="n=%{x}<br>Theo Rate (Orig Baseline)=%{y:.6f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results["n_values"],
            y=results["emp_rate_orig"],
            mode="lines+markers",
            name="Emp. Orig (1 prime (n,n²), K_orig)",
            marker=dict(symbol="x", size=6, color="blue"),
            line=dict(dash="dash", color="blue"),
            hovertemplate="n=%{x}<br>Emp Rate (Orig)=%{y:.6f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results["n_values"],
            y=results["emp_rate_cm1"],
            mode="lines+markers",
            name="Emp. CM1 (2 primes (n,n²), K_orig)",
            marker=dict(symbol="diamond", size=6, color="red"),
            line=dict(dash="dot", color="red"),
            hovertemplate="n=%{x}<br>Emp Rate (CM1)=%{y:.6f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results["n_values"],
            y=results["emp_rate_cm2"],
            mode="lines+markers",
            name="Emp. CM2 (1 prime (nln n,n²), K_restr)",
            marker=dict(symbol="star", size=6, color="green"),
            line=dict(dash="dashdot", color="green"),
            hovertemplate="n=%{x}<br>Emp Rate (CM2 - Adapted K)=%{y:.6f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results["n_values"],
            y=results["emp_rate_cm3"],
            mode="lines+markers",
            name="Emp. CM3 (2 primes (nln n,n²), K_restr)",
            marker=dict(symbol="triangle-up", size=6, color="purple"),
            line=dict(dash="longdash", color="purple"),
            hovertemplate="n=%{x}<br>Emp Rate (CM3 - Adapted K)=%{y:.6f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results["n_values"],
            y=results["emp_rate_cm4"],
            mode="lines+markers",
            name="Emp. CM4 (3 primes (nln n,n²), K_restr)",
            marker=dict(symbol="hexagon", size=6, color="orange"),
            line=dict(dash="solid", width=1.5, color="orange"),
            hovertemplate="n=%{x}<br>Emp Rate (CM4 - Adapted K)=%{y:.6f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Fingerprinting: False Positive Rates with Adaptive Adversary (CM4 included)",
        xaxis_title="n (Number of Bits)",
        yaxis_title="False Positive Rate (Log Scale)",
        yaxis_type="log",
        hovermode="x unified",
        legend_title_text="Scheme & Adversary Strategy",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    try:
        fig.write_html(filename)
        print(f"Interactive plot saved to '{os.path.abspath(filename)}'")
    except Exception as e:
        print(f"Error saving plot to HTML: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    experiment_results = run_adaptive_adversary_experiment(
        n_start=20,  # Start higher to ensure n*ln(n) is distinct enough and pools are not too small
        n_end=1000,  # Keep n_end moderate for runtime
        num_trials_empirical=10000,
    )
    create_adaptive_interactive_plot(experiment_results)
