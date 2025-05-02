import math
import random
import matplotlib.pyplot as plt
import numpy as np  # Using numpy for potential efficiency with large numbers/arrays if needed
import time

# This code was created with assistance from Gemini.

# --- Prime Number Generation ---


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
    prime[0] = prime[1] = False  # 0 and 1 are not prime

    for i in range(2, int(math.sqrt(limit)) + 1):
        if prime[i]:
            # Mark multiples of i as not prime
            for multiple in range(i * i, limit, i):
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


# --- Fingerprinting Analysis ---


def calculate_adversary_k_and_theoretical_rate(n, primes_between_n_n2):
    """
    Calculates the adversary's number K and the theoretical false positive rate.
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
        return 1, 0.0, 0  # No primes, K=1, rate=0

    limit = pow(2, n) - 1  # The maximum value for x or y
    if (
        limit < 0
    ):  # Handle potential overflow for very large n if not using Python's large ints
        print(
            f"Warning: 2^{n}-1 might be too large for standard types, relying on Python's arbitrary precision."
        )
        limit = (1 << n) - 1  # Bit shift alternative for power of 2

    K = 1
    num_adversary_primes = 0
    # Primes are already sorted by get_primes_in_range if using the sieve output directly
    # If not, sort them: sorted_primes = sorted(primes_between_n_n2)
    sorted_primes = primes_between_n_n2  # Already sorted from sieve generation

    for p in sorted_primes:
        # Check for potential overflow before multiplication
        # Python handles large integers, but good practice conceptually
        if K > limit // p:  # Check if K * p would exceed limit
            break
        next_K = K * p
        if next_K < limit:
            K = next_K
            num_adversary_primes += 1
        else:
            break  # Stop if the next product exceeds the limit

    total_primes = len(primes_between_n_n2)
    theoretical_rate = num_adversary_primes / total_primes if total_primes > 0 else 0.0

    return K, theoretical_rate, num_adversary_primes


def calculate_empirical_rate(K, primes_between_n_n2, num_trials=10000):
    """
    Calculates the empirical false positive rate through simulation.
    Args:
        K (int): The adversary's number.
        primes_between_n_n2 (list): List of primes p such that n < p < n^2.
        num_trials (int): The number of random trials to run.
    Returns:
        float: The empirical false positive rate.
    """
    if not primes_between_n_n2 or num_trials <= 0:
        return 0.0  # No primes or trials, rate is 0

    false_positives = 0
    x = 0  # Alice's number
    y = K  # Bob's number (adversary's choice)

    for _ in range(num_trials):
        p = random.choice(primes_between_n_n2)  # Alice chooses a random prime
        h = x % p  # Alice computes hash (always 0 for x=0)
        g = y % p  # Bob computes hash

        if g == h:  # Check for collision (false positive when x != y)
            false_positives += 1

    empirical_rate = false_positives / num_trials
    return empirical_rate


# --- Main Execution and Plotting ---


def run_experiment(n_start=6, n_end=1000, num_trials_empirical=10000):
    """
    Runs the full experiment for n from n_start to n_end and plots the results.
    Args:
        n_start (int): The starting value of n.
        n_end (int): The ending value of n (inclusive).
        num_trials_empirical (int): Number of trials for empirical calculation.
    """
    n_values = []
    theoretical_rates = []
    empirical_rates = []

    print(f"Running experiment for n from {n_start} to {n_end}...")
    start_time = time.time()

    # Pre-calculate primes up to the maximum possible n^2 to optimize
    max_n_squared = n_end * n_end
    print(f"Pre-calculating primes up to {max_n_squared}...")
    all_primes = primes_sieve(max_n_squared)
    print(f"Prime calculation finished. Found {len(all_primes)} primes.")

    prime_dict = {p: True for p in all_primes}  # Use a dictionary for faster lookups

    for n in range(n_start, n_end + 1):
        n_squared = n * n
        lower_bound = n + 1  # Primes must be > n
        upper_bound = n_squared  # Primes must be < n^2

        # Efficiently filter pre-calculated primes for the current range (n, n^2)
        primes_n_n2 = [p for p in all_primes if lower_bound <= p < upper_bound]

        if not primes_n_n2:
            print(f"n={n}: No primes found between {n+1} and {n_squared}. Skipping.")
            continue

        # 1. Theoretical Calculation
        K, theo_rate, num_adv_primes = calculate_adversary_k_and_theoretical_rate(
            n, primes_n_n2
        )

        # 2. Empirical Calculation
        emp_rate = calculate_empirical_rate(
            K, primes_n_n2, num_trials=num_trials_empirical
        )

        n_values.append(n)
        theoretical_rates.append(theo_rate)
        empirical_rates.append(emp_rate)

        if (
            n == 6 or n == 10 or n % 50 == 0 or n == n_end
        ):  # Print progress periodically
            print(
                f"n={n}: Primes={len(primes_n_n2)}, Adv Primes={num_adv_primes}, K={K if K < 1e10 else '>1e10'}, Theo Rate={theo_rate:.4f}, Emp Rate={emp_rate:.4f}"
            )

    end_time = time.time()
    print(f"\nExperiment finished in {end_time - start_time:.2f} seconds.")

    # 3. Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(
        n_values,
        theoretical_rates,
        label="Theoretical False Positive Rate",
        marker=".",
        linestyle="-",
        markersize=3,
    )
    plt.plot(
        n_values,
        empirical_rates,
        label=f"Empirical False Positive Rate ({num_trials_empirical} trials)",
        marker="x",
        linestyle="--",
        markersize=3,
        alpha=0.7,
    )

    plt.xlabel("n (Number of Bits)")
    plt.ylabel("False Positive Rate")
    plt.title("Randomized Fingerprinting: Adversarial False Positive Rate vs. n")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.ylim(bottom=0)  # Rate cannot be negative
    # Optional: Use a log scale for y-axis if rates vary widely
    # plt.yscale('log')
    plt.tight_layout()
    plt.show()


# --- Run the experiment ---
if __name__ == "__main__":
    # Adjust n_end and num_trials_empirical as needed.
    # Higher n_end takes significantly longer.
    # Higher num_trials_empirical gives more accurate empirical results but takes longer.
    run_experiment(n_start=6, n_end=1000, num_trials_empirical=10000)
