import math
import numpy as np
import matplotlib.pyplot as plt

def calculate_upper_bound_rate(n):
    """
    Calculates an approximate upper bound on the false positive rate
    using theoretical estimates based on the Prime Number Theorem.

    Args:
        n (float or int): The number of bits. Must be greater than 2.

    Returns:
        float: An approximate upper bound for the false positive rate.
               Returns float('inf') if calculation is not possible (e.g., n <= 2).
    """
    # Ensure n is large enough for logarithms and denominators
    if n <= 2:
        # log(n) is undefined or zero for n<=1, n^2-2n is zero for n=2
        return float('inf')

    try:
        # Use natural logarithm (ln)
        ln_n = math.log(n)
        ln_2 = math.log(2)

        # 1. Overestimate N: Max number of primes in K
        #    Find the largest integer m such that n^m < 2^n
        #    m * ln(n) < n * ln(2) => m < n * ln(2) / ln(n)
        #    We use floor to get the largest integer m satisfying this.
        #    This approximates the number of primes (all assumed >= n)
        #    whose product could be less than 2^n.
        N_upper_estimate = math.floor(n * ln_2 / ln_n)

        # Handle cases where the estimate is non-positive (can happen for small n)
        if N_upper_estimate <= 0:
             # If N is estimated as 0 or less, the rate is effectively 0
             # (or very small, returning 0 is a reasonable upper bound here)
             # This might happen if n*ln(2)/ln(n) < 1. e.g., n=3: 3*ln(2)/ln(3) ~ 1.89 -> floor=1
             # e.g., n=2: 2*ln(2)/ln(2) = 2 -> floor=2.
             # Let's return a very small number instead of 0 if N_upper is 0,
             # to avoid division by zero issues later if D is also small,
             # and to show decrease on log plot.
             # However, if N_upper is truly 0, the bound *is* 0.
             # Let's stick to the formula: if N_upper is 0, rate is 0.
             if N_upper_estimate == 0:
                 return 0.0


        # 2. Underestimate D: Number of primes between n and n^2
        #    Using Prime Number Theorem: pi(x) ~ x / ln(x)
        #    D = pi(n^2) - pi(n)
        #    D ~ (n^2 / ln(n^2)) - (n / ln(n))
        #    D ~ (n^2 / (2 * ln(n))) - (n / ln(n))
        #    D ~ (n^2 - 2n) / (2 * ln(n))
        #    This formula tends to underestimate D for smaller n, which is
        #    what we need for an upper bound on N/D.
        denominator_D = (n * n - 2 * n)
        if denominator_D <= 0:
             # Should not happen for n > 2
             return float('inf')

        D_lower_estimate = denominator_D / (2 * ln_n)

        # Ensure D estimate is positive
        if D_lower_estimate <= 0:
            # This might happen if the PNT approximation breaks down for small n,
            # though unlikely for n >= 10.
            return float('inf') # Rate is undefined or extremely large

        # 3. Calculate Upper Bound on Rate = N_upper / D_lower
        rate_upper_bound = N_upper_estimate / D_lower_estimate

        return rate_upper_bound

    except (ValueError, OverflowError) as e:
        # Catch potential math errors (e.g., log of non-positive)
        # or overflow with very large n, although Python handles large ints
        print(f"Warning: Calculation error for n={n}: {e}")
        return float('nan') # Not a Number indicates an issue

# --- Plotting ---

def plot_upper_bound(n_start=10, n_end_power=100, num_points=200):
    """
    Generates a log-log plot of the theoretical upper bound on the
    false positive rate for n ranging from n_start to 10^n_end_power.

    Args:
        n_start (int): The starting value of n.
        n_end_power (int): The power of 10 for the ending value of n.
        num_points (int): The number of points to plot (logarithmically spaced).
    """
    print(f"Calculating upper bound for n from {n_start} to 10^{n_end_power}...")

    # Generate n values logarithmically spaced
    # Use float128 for n if available and needed, but standard float64 often sufficient
    # np.logspace handles large exponents directly
    n_values = np.logspace(np.log10(n_start), n_end_power, num=num_points)

    # Calculate the upper bound rate for each n
    # Use list comprehension, handle potential NaN results
    rates = [calculate_upper_bound_rate(n) for n in n_values]

    # Filter out any NaN results if they occurred
    valid_indices = [i for i, r in enumerate(rates) if not np.isnan(r) and r != float('inf')]
    n_values_valid = n_values[valid_indices]
    rates_valid = [rates[i] for i in valid_indices]

    if not list(n_values_valid):
         print("Error: No valid rates calculated. Cannot plot.")
         return

    print("Calculation complete. Generating plot...")

    # Create the log-log plot
    plt.figure(figsize=(10, 6))
    plt.loglog(n_values_valid, rates_valid, marker='.', linestyle='-', markersize=3)

    # Add asymptotic line for comparison: Rate ~ (2*ln2)/n
    asymptotic_rates = (2 * math.log(2)) / n_values_valid
    plt.loglog(n_values_valid, asymptotic_rates, label=r'Asymptotic Behavior ($ \approx \frac{2 \ln 2}{n}$)', linestyle='--', color='red', alpha=0.7)


    plt.xlabel('n (Number of Bits)')
    plt.ylabel('Upper Bound on False Positive Rate')
    plt.title('Theoretical Upper Bound on Fingerprinting False Positive Rate (Log-Log Scale)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# --- Run the plotting function ---
if __name__ == "__main__":
    # Plot for n from 10 up to 1 googol (10^100)
    plot_upper_bound(n_start=10, n_end_power=100, num_points=500)
