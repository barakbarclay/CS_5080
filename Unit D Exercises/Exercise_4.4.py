import numpy as np
import matplotlib.pyplot as plt

# Code written with assistance from Gemini. Proofread and edited by me.


def linear_search(arr, key):
    """Performs linear search on an array."""
    for i, val in enumerate(arr):
        if val == key:
            return i
    return -1


def empirical_verification(M, alphabet, key, n_values, num_arrays):
    """Verifies linear search probabilities empirically."""

    for n in n_values:
        results = {i: 0 for i in range(-1, n)}
        for _ in range(num_arrays):
            arr = np.random.choice(alphabet, n)
            result = linear_search(arr, key)
            results[result] += 1

        frequencies = {i: count / num_arrays for i, count in results.items()}

        # Theoretical probabilities (Simplified model)
        theoretical_probs = {}
        if key in alphabet:
            for i in range(n):
                theoretical_probs[i] = (1 / M) * ((1 - 1 / M) ** i)
            theoretical_probs[-1] = 1 - sum(theoretical_probs.values())
        else:
            for i in range(n):
                theoretical_probs[i] = 0
            theoretical_probs[-1] = 1

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(
            frequencies.keys(), frequencies.values(), label="Empirical Frequencies"
        )
        plt.scatter(
            theoretical_probs.keys(),
            theoretical_probs.values(),
            label="Theoretical Probabilities",
        )
        plt.yscale("log")
        plt.title(f"Linear Search: n = {n}")
        plt.xlabel("Index")
        plt.ylabel("Probability (Log Scale)")
        plt.legend()
        plt.xticks(list(frequencies.keys()))
        plt.grid(True)
        plt.show()


# Parameters
M = 5
alphabet = ["A", "B", "C", "D", "E"]
key = "A"
n_values = [5, 10, 20, 50]
num_arrays = 10000  # Adjust as needed

# Run verification
empirical_verification(M, alphabet, key, n_values, num_arrays)
