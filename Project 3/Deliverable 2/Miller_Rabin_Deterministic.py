import random
import time

# Core Miller-Rabin test for a single base 'a'
# This is the _test function from a typical Miller-Rabin implementation
def _is_strong_probable_prime(n, a, d, s):
    """
    Checks if n is a strong probable prime to base a.
    n - 1 = 2^s * d where d is odd.
    """
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True # n is s-p-p to base a

    for _ in range(s - 1):
        x = pow(x, 2, n)
        if x == n - 1:
            return True # n is s-p-p to base a
        if x == 1:
            return False # n is composite (non-trivial sqrt of 1 found)
    return False # n is composite if we fall through

def miller_rabin_single_base_check(n, a):
    """
    Performs the Miller-Rabin test for number n with a single base a.
    Returns True if n is a strong probable prime to base a, False if n is definitely composite.
    """
    if n == a: # If n is one of the bases, it's prime (bases are small primes)
        return True # Or handle this by ensuring n > largest base if bases are fixed.
                    # More robustly, the main M-R handles n=2,3.
                    # If n equals a base, typically a small prime, it would have been caught.
                    # This check is more for `a` chosen randomly relative to `n`.
                    # For fixed small prime bases, if n is that base, it's prime.
        pass # This specific scenario is usually pre-filtered.


    if n % a == 0: # If 'a' divides 'n', and 'a' < 'n', then 'n' is composite.
                   # If 'a' == 'n', then 'n' is prime (and 'a' is prime).
                   # This is handled by initial checks in a full primality test.
        return n == a # True if n is the prime base 'a', False if 'a' is a factor of composite n

    # Write n - 1 as 2^s * d
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    return _is_strong_probable_prime(n, a, d, s)

def miller_rabin_deterministic_jaeschke(n):
    """
    Deterministic Miller-Rabin test for n < 4,759,123,141
    using bases {2, 7, 61}.
    Returns "prime" or "composite".
    """
    if n <= 1: return "composite"
    if n == 2 or n == 3 or n == 5 or n == 7 or n == 61: # The bases themselves are prime
        # Check if n is one of the bases and handle other small primes
        if n in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}: # Small primes
            return "prime"
    if n % 2 == 0 or n % 3 == 0 or n % 5 == 0: # Basic trial division for very small factors
        return "composite" # if n is not 2, 3, or 5 itself

    # Specific limit for these bases
    if n >= 4759123141:
        # For numbers outside this range, this deterministic set is not guaranteed.
        # Fall back to probabilistic or use a different deterministic set.
        # For this example, we'll indicate it's out of range for this specific function.
        # A more general function would select bases based on n.
        return "out of range for this deterministic set, use probabilistic"

    bases = [2, 7, 61]

    # Initial checks for small numbers (already handled by above if statements, but good practice)
    if n < 2: return "composite" # 0, 1
    # Primes 2, 3 are handled by the n % 2/3 check if not caught by equality to bases
    # if n in {2,3}: return "prime" # Covered

    for a in bases:
        if n == a: # If n is one of the bases, it's prime.
            return "prime" # This can be simplified by ensuring bases are smaller than n
                           # or by handling small primes before the loop.
                           # Given the pre-checks, if n matches a base, it must be that prime.
        if not miller_rabin_single_base_check(n, a):
            return "composite" # Found a witness, n is definitely composite
    return "prime" # Passed all deterministic checks for this range

# Probabilistic Miller-Rabin from previous task (for comparison)
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
        a = random.randrange(2, n - 1) if n > 4 else 2 # Handle n=4 for randrange
        if not _is_strong_probable_prime(n, a, d, s):
            return "composite"
    return "probably prime"

# Benchmarking comparison

def run_benchmark():
    print("\n--- Efficiency Comparison ---")
    # Test numbers up to a certain limit, e.g., 1,000,000
    # This is well within the 4.7 billion limit for Jaeschke set.
    test_limit = 100000 # Reduced for quicker benchmark demonstration
    numbers_to_test = list(range(test_limit)) # Test 0 to test_limit-1

    # Deterministic Jaeschke
    start_time_det = time.time()
    for num in numbers_to_test:
        miller_rabin_deterministic_jaeschke(num)
    end_time_det = time.time()
    time_taken_det = end_time_det - start_time_det
    print(f"Deterministic (Jaeschke, bases {{2,7,61}}) for N < {test_limit}: {time_taken_det:.4f} seconds")

    # Probabilistic Miller-Rabin with k=3 (to match number of bases)
    k_prob = 3
    start_time_prob_k3 = time.time()
    for num in numbers_to_test:
        miller_rabin_probabilistic(num, k_prob)
    end_time_prob_k3 = time.time()
    time_taken_prob_k3 = end_time_prob_k3 - start_time_prob_k3
    print(f"Probabilistic (k={k_prob}) for N < {test_limit}: {time_taken_prob_k3:.4f} seconds")

    # Probabilistic Miller-Rabin with k=5 (a common small k)
    k_prob_5 = 5
    start_time_prob_k5 = time.time()
    for num in numbers_to_test:
        miller_rabin_probabilistic(num, k_prob_5)
    end_time_prob_k5 = time.time()
    time_taken_prob_k5 = end_time_prob_k5 - start_time_prob_k5
    print(f"Probabilistic (k={k_prob_5}) for N < {test_limit}: {time_taken_prob_k5:.4f} seconds")

    # Probabilistic Miller-Rabin with k=10 (higher confidence)
    k_prob_10 = 10
    start_time_prob_k10 = time.time()
    for num in numbers_to_test:
        miller_rabin_probabilistic(num, k_prob_10)
    end_time_prob_k10 = time.time()
    time_taken_prob_k10 = end_time_prob_k10 - start_time_prob_k10
    print(f"Probabilistic (k={k_prob_10}) for N < {test_limit}: {time_taken_prob_k10:.4f} seconds")

    print("\nNote: The deterministic Jaeschke test provides certainty for n < 4,759,123,141.")
    print("Probabilistic tests offer a trade-off between speed (iterations) and certainty.")
    print("The benchmark includes overhead of looping and function calls in Python.")
    print("The `miller_rabin_deterministic_jaeschke` has some extra initial checks compared to a bare 3-iteration probabilistic.")

# Example usage of the deterministic test:
if __name__ == "__main__":
    print("--- Deterministic Miller-Rabin (Jaeschke for n < 4,759,123,141) ---")
    test_numbers = [1, 2, 3, 4, 17, 25, 61, 97, 2047, 1373653, 4759123140, 4759123141-2, 4759123141] # Last one is out of range for this specific func
    
    # Small primes and composites
    print(f"Is 1 prime? {miller_rabin_deterministic_jaeschke(1)}")
    print(f"Is 2 prime? {miller_rabin_deterministic_jaeschke(2)}")
    print(f"Is 17 prime? {miller_rabin_deterministic_jaeschke(17)}")
    print(f"Is 22 (composite) prime? {miller_rabin_deterministic_jaeschke(22)}")
    print(f"Is 61 prime? {miller_rabin_deterministic_jaeschke(61)}") # A base
    print(f"Is 97 prime? {miller_rabin_deterministic_jaeschke(97)}")

    # Smallest strong pseudoprime for base 2 (composite)
    print(f"Is 2047 prime? {miller_rabin_deterministic_jaeschke(2047)}") # Should be composite

    # Number close to the limit (prime)
    # A large prime less than the limit: 4,759,123,123 (is prime)
    large_prime_in_range = 4759123123
    print(f"Is {large_prime_in_range} prime? {miller_rabin_deterministic_jaeschke(large_prime_in_range)}")

    # Number just at the limit (the function will state it's out of range for this specific version)
    # The Jaeschke set is for n < 4,759,123,141.
    # So, 4,759,123,140 is the largest number testable by this specific implementation.
    print(f"Is 4759123140 prime? {miller_rabin_deterministic_jaeschke(4759123140)}")
    print(f"Is 4759123141 prime? {miller_rabin_deterministic_jaeschke(4759123141)}")


    run_benchmark()