import numpy as np
import math
import parts_1_and_2_secretary_simulation as p12_sim

# --- Constants ---
N_CANDIDATES = 100  # Default number of candidates (n)
N_TRIALS = 10000  # Number of simulation runs for empirical results
# Use the theoretically optimal threshold fraction for the classical strategy
K_FRACTION = 1 / np.e
K_THRESHOLD = max(
    1, min(N_CANDIDATES - 1, round(N_CANDIDATES * K_FRACTION))
)  # Rejection threshold k (Ensure 0 < k < n)

# --- Part 1/2 Adapters/Re-used Components ---


# Wrapper function to call imported generator with correct signature
def generate_candidates_uniform_wrapper(n):
    """Generates n candidate scores from Uniform(0, 1) using imported function."""
    return p12_sim.generate_candidates(n, distribution="uniform")


# --- Part 3: Core Implementation ---


def find_true_best(candidates):
    """Finds the score and index of the best candidate."""
    # Ensure input is handled correctly (e.g., numpy array)
    candidates = np.asarray(candidates)
    if (
        not candidates.any()
    ):  # Check if array is not empty or all zeros if that's possible
        # Check size instead for empty array
        if candidates.size == 0:
            return -1, -1
    true_best_score = np.max(candidates)
    true_best_index = np.argmax(candidates)
    return true_best_score, true_best_index


# Basic Estimator (from Part 1, Stretch Goal 1.3)
def basic_estimator(max_seen_so_far, candidates_seen_count, total_candidates):
    """
    Estimates the best value based *only* on the maximum value seen so far.
    Args:
        max_seen_so_far: The maximum score observed among the first 'candidates_seen_count'.
        candidates_seen_count: The number of candidates observed so far (t).
        total_candidates: The total number of candidates (n).
    Returns:
        The estimated value (which is just max_seen_so_far).
    """
    # This basic estimator ignores t and n, just returns the current best observed.
    return max_seen_so_far


# Task 1: Introducing Variable Distributions
def generate_candidates_variable_uniform(n):
    """
    Generates n candidate scores:
    - First floor(n/2) from Uniform(0, 0.5)
    - Remaining from Uniform(0.5, 1)
    """
    n_half1 = math.floor(n / 2)
    n_half2 = n - n_half1

    scores_half1 = np.random.uniform(0, 0.5, n_half1)
    scores_half2 = np.random.uniform(0.5, 1, n_half2)

    # Combine and shuffle to maintain random arrival order overall
    all_scores = np.concatenate((scores_half1, scores_half2))
    np.random.shuffle(
        all_scores
    )  # Shuffle to simulate random arrival from the mixed pool
    # If arrival order matters (first half candidates strictly from dist 1, etc.)
    # then don't shuffle. The prompt implies the *pool* changes halfway.
    # Let's assume the candidates arrive sequentially from these distributions without shuffling.
    # Revert shuffle for sequential arrival:
    # all_scores = np.concatenate((scores_half1, scores_half2))

    return np.concatenate((scores_half1, scores_half2))


# Classical Strategy Implementation (Adapted to include estimator call)
def run_classical_strategy(candidates, k, estimator_func):
    """
    Runs the classical secretary strategy and applies an estimator.

    Args:
        candidates: A list or numpy array of candidate scores.
        k: The rejection threshold (number of candidates to reject initially).
        estimator_func: A function to call for estimating the best value.
                        Signature: estimator_func(max_seen_so_far, t, n) -> estimate

    Returns:
        A dictionary containing:
        - 'selected_score': Score of the selected candidate (None if none selected).
        - 'selected_index': Index of the selected candidate (None if none selected).
        - 'was_best_selected': Boolean indicating if the true best was selected.
        - 'estimate_at_selection': The estimator's output just before selection,
                                   or the final estimate if no selection.
        - 'true_best_score': The actual highest score among all candidates.
        - 'estimates': List of estimates made after the rejection phase.
    """
    n = len(candidates)
    candidates = np.asarray(candidates)  # Ensure numpy array for np ops

    if n == 0:
        return {
            "selected_score": None,
            "selected_index": None,
            "was_best_selected": False,
            "estimate_at_selection": None,
            "true_best_score": -1,
            "estimates": [],
        }

    true_best_score, true_best_index = find_true_best(candidates)

    # --- Rejection Phase ---
    if k >= n or k < 0:  # Handle edge cases for k
        # If k >= n, we reject everyone. If k < 0, invalid input, treat as k=0?
        # Let's follow the rule: reject k, then select. If k>=n, no selection possible after rejection.
        max_seen_in_rejection = -1
        if k > 0 and n > 0:
            # We still need to see the first k to potentially estimate later
            max_seen_in_rejection = (
                np.max(candidates[: min(k, n)]) if min(k, n) > 0 else -1
            )

        # Need to calculate final estimate if no selection possible
        final_estimate = None
        if n > 0:
            max_seen_overall = np.max(candidates) if n > 0 else -1
            final_estimate = estimator_func(max_seen_overall, n, n)

        return {
            "selected_score": None,
            "selected_index": None,
            "was_best_selected": False,
            "estimate_at_selection": final_estimate,
            "true_best_score": true_best_score,
            "estimates": [final_estimate] if final_estimate is not None else [],
        }

    max_seen_in_rejection = -1  # Default for k=0
    if k > 0:
        max_seen_in_rejection = np.max(candidates[:k])

    # --- Selection Phase ---
    selected_score = None
    selected_index = None
    estimate_at_selection = None
    estimates = []
    max_seen_so_far = (
        max_seen_in_rejection  # Needed for estimator after rejection phase
    )

    for i in range(k, n):
        current_candidate_score = candidates[i]
        # Update max seen *before* estimating for step t=i+1
        if i > 0:  # Update max_seen_so_far based on candidates up to i-1
            max_seen_so_far = np.max(candidates[:i])  # Max of candidates 0 to i-1
        else:  # if i=0 (only happens if k=0)
            max_seen_so_far = -1  # No prior candidates seen

        # Calculate estimate at step t = i+1 (after seeing candidate i)
        # The estimator should logically be applied *before* deciding on candidate i,
        # using data up to i-1, OR applied after seeing candidate i, using data up to i.
        # Let's assume the estimate is formed *after* seeing candidate i, using data up to i.
        current_max_seen = np.max(
            candidates[: i + 1]
        )  # Max including the current candidate
        current_estimate = estimator_func(current_max_seen, i + 1, n)
        estimates.append(current_estimate)

        # Selection condition for classical strategy:
        # Better than *all* previously seen candidates (including rejection phase)
        # Note: max_seen_so_far here includes up to i-1. We need max up to i-1 for comparison.
        best_seen_before_current = max_seen_in_rejection
        if i > k:  # If we are beyond the first element after rejection phase
            best_seen_before_current = np.max(candidates[:i])  # Max of 0..i-1

        if current_candidate_score > best_seen_before_current:
            selected_score = current_candidate_score
            selected_index = i
            estimate_at_selection = (
                current_estimate  # Use the estimate calculated at this step
            )
            break  # Stop after first selection

    # If loop finishes without selection
    if selected_index is None and estimates:
        estimate_at_selection = estimates[-1]  # Use the last calculated estimate
    elif (
        not estimates and n > 0
    ):  # e.g., k=n case handled above, but maybe k=n-1, loop runs once?
        # If loop didn't run (k=n) or only ran once and didn't select, get final estimate
        max_seen_overall = np.max(candidates) if n > 0 else -1
        estimate_at_selection = estimator_func(max_seen_overall, n, n)
        # This case might be redundant due to the k>=n check earlier

    was_best_selected = selected_index == true_best_index

    return {
        "selected_score": selected_score,
        "selected_index": selected_index,
        "was_best_selected": was_best_selected,
        "estimate_at_selection": estimate_at_selection,
        "true_best_score": true_best_score,
        "estimates": estimates,
    }


# Simulation Runner (Adapted from Part 1/2 concepts)
def run_simulation(n, num_trials, k, candidate_generator, estimator_func):
    """Runs multiple trials of the secretary problem."""
    success_count = 0
    estimator_diffs = []  # Store difference: estimate_at_selection - true_best_score
    selections_made = 0

    for _ in range(num_trials):
        candidates = candidate_generator(n)
        result = run_classical_strategy(candidates, k, estimator_func)

        if result["was_best_selected"]:
            success_count += 1

        if result["selected_index"] is not None:
            selections_made += 1
            # Calculate estimator performance only when a selection is made
            if result["estimate_at_selection"] is not None:
                diff = result["estimate_at_selection"] - result["true_best_score"]
                estimator_diffs.append(diff)
            # else: # Handle case where selection happened but estimate is None (shouldn't normally happen)
            # print("Warning: Selection made but estimate is None.")

    success_rate = success_count / num_trials if num_trials > 0 else 0
    # Calculate average difference only over trials where a selection was made
    avg_estimator_diff = np.mean(estimator_diffs) if estimator_diffs else None
    selection_rate = selections_made / num_trials if num_trials > 0 else 0

    return {
        "success_rate": success_rate,
        "avg_estimator_diff": avg_estimator_diff,
        "selection_rate": selection_rate,
        "num_selections_made": selections_made,
    }


# Task 3: Advanced Estimator Implementation
def advanced_estimator(max_seen_so_far, t, n):
    """
    Estimates the expected maximum value of the *entire* sequence of n candidates,
    given the maximum value seen ('max_seen_so_far') after observing 't' candidates,
    assuming the underlying distribution is Uniform(0, 1).

    Uses the formula derived from order statistics for U(0,1):
    E[Max(X_1..X_n) | Max(X_1..X_t) = m] = m + (1-m) * Sum_{i=1}^{n-t} (1 / (t + i))

    Args:
        max_seen_so_far: The maximum score observed among the first 't' candidates.
        t: The number of candidates observed so far.
        n: The total number of candidates.

    Returns:
        The estimated expected maximum value for the whole sequence.
    """
    if t >= n:
        # If we've seen all candidates, the max seen is the true max.
        return max_seen_so_far
    if t <= 0:
        # Cannot estimate based on 0 observations. Return a default (e.g., expected max of n U(0,1) vars = n/(n+1))
        # Or handle as an error/None. Let's return n/(n+1) as a prior estimate.
        return n / (n + 1)
    if max_seen_so_far < 0:  # Should not happen with U(0,1) unless t=0
        return n / (n + 1)  # Default prior

    # Calculate the sum term: Sum_{i=1}^{n-t} (1 / (t + i))
    # This is related to Harmonic numbers: H_{n} - H_{t}
    # H_k = sum_{j=1}^k 1/j
    # sum_{i=1}^{n-t} (1 / (t + i)) = (1/(t+1) + 1/(t+2) + ... + 1/n)
    # = (1 + 1/2 + ... + 1/n) - (1 + 1/2 + ... + 1/t) = H_n - H_t
    # Using math.gamma for Digamma function: psi(x) = d/dx ln(gamma(x))
    # H_n approx log(n) + gamma (Euler-Mascheroni const)
    # H_n = psi(n + 1) + gamma
    # H_n - H_t = psi(n + 1) - psi(t + 1)

    # Let's compute the sum directly for simplicity and accuracy for moderate n
    sum_term = 0
    for i in range(1, n - t + 1):
        # Add check for potential division by zero though t+i should be > 0
        if t + i == 0:
            continue  # Should not happen given t>=1 check
        sum_term += 1.0 / (t + i)

    # Formula assumes max_seen_so_far is indeed the max of first t draws from U(0,1)
    # It estimates the expected *final* maximum given this information.
    estimated_max = max_seen_so_far + (1 - max_seen_so_far) * sum_term

    # Clamp estimate between 0 and 1 as scores are in U(0,1)
    return max(0, min(1, estimated_max))


# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"--- Secretary Problem Simulation ---")
    print(
        f"Parameters: n = {N_CANDIDATES}, k = {K_THRESHOLD} (reject first {K_THRESHOLD}), Trials = {N_TRIALS}"
    )
    print("-" * 30)

    # Baseline: Single Uniform Distribution with Basic Estimator
    print("Running Scenario 1: Single Uniform Distribution U(0, 1)")
    print("Estimator: Basic (max seen so far)")
    results_uniform_basic = run_simulation(
        N_CANDIDATES,
        N_TRIALS,
        K_THRESHOLD,
        generate_candidates_uniform_wrapper,  # USE WRAPPER
        basic_estimator,
    )
    print(f"  Success Rate (finding best): {results_uniform_basic['success_rate']:.4f}")
    print(
        f"  Selection Rate (selecting any): {results_uniform_basic['selection_rate']:.4f}"
    )
    if results_uniform_basic["avg_estimator_diff"] is not None:
        print(
            f"  Avg. Estimator Difference (Estimate - True Best, when selected): {results_uniform_basic['avg_estimator_diff']:.4f}"
        )
    else:
        print(
            f"  Avg. Estimator Difference: Not applicable (no selections made or estimator failed)."
        )
    print("-" * 30)

    # Task 1 & 2: Variable Distribution Scenario with Basic Estimator
    print(f"Running Scenario 2: Variable Distribution (U(0, 0.5) then U(0.5, 1))")
    print("Estimator: Basic (max seen so far)")
    results_variable_basic = run_simulation(
        N_CANDIDATES,
        N_TRIALS,
        K_THRESHOLD,
        generate_candidates_variable_uniform,  # USE LOCAL
        basic_estimator,
    )
    print(
        f"  Success Rate (finding best): {results_variable_basic['success_rate']:.4f}"
    )
    print(
        f"  Selection Rate (selecting any): {results_variable_basic['selection_rate']:.4f}"
    )
    if results_variable_basic["avg_estimator_diff"] is not None:
        print(
            f"  Avg. Estimator Difference (Estimate - True Best, when selected): {results_variable_basic['avg_estimator_diff']:.4f}"
        )
    else:
        print(f"  Avg. Estimator Difference: Not applicable (no selections made).")
    print("-" * 30)

    # Task 2: Comparison Discussion Placeholder
    print("Comparison (Scenario 1 vs Scenario 2):")
    print(
        f"  Change in Success Rate: {results_variable_basic['success_rate'] - results_uniform_basic['success_rate']:.4f}"
    )
    print(
        f"  Change in Avg. Estimator Diff: {(results_variable_basic['avg_estimator_diff'] or 0) - (results_uniform_basic['avg_estimator_diff'] or 0):.4f}"
    )
    print("  Discussion:")
    print("  The classical strategy's success rate often relies on the assumption")
    print("  that the relative ranks are uniformly distributed and the distribution")
    print("  doesn't change drastically. When the distribution shifts significantly")
    print("  (like here, from low values to high values), the optimal threshold k")
    print("  calculated for U(0,1) might be suboptimal.")
    print("  - If k is too small (e.g., k < n/2), the rejection phase primarily sees")
    print("    lower-value candidates. The threshold set by max_seen_in_rejection")
    print("    will be low, causing the strategy to potentially select the very first")
    print("    candidate from the higher U(0.5, 1) distribution, which is unlikely")
    print("    to be the absolute best overall.")
    print(
        "  - If k is large (e.g., k > n/2), the rejection phase sees both distributions,"
    )
    print("    potentially setting a high threshold, but might reject good candidates")
    print("    from the second, better distribution.")
    print("  The observed change in success rate reflects this sensitivity.")
    print(
        "  The basic estimator's performance (simply using max seen) might also change;"
    )
    print("  if selection happens early in the U(0.5,1) phase, the estimate might be")
    print("  closer to the true best (as scores are higher), but if selection happens")
    print("  late or not at all, the comparison is complex.")
    print("-" * 30)

    # Task 3: Run with Advanced Estimator (on single uniform distribution for comparison)
    print(f"Running Scenario 3: Single Uniform Distribution U(0, 1)")
    print("Estimator: Advanced (Order Statistics based for U(0,1))")
    results_uniform_advanced = run_simulation(
        N_CANDIDATES,
        N_TRIALS,
        K_THRESHOLD,
        generate_candidates_uniform_wrapper,  # USE WRAPPER
        advanced_estimator,
    )
    print(
        f"  Success Rate (finding best): {results_uniform_advanced['success_rate']:.4f}"
    )  # Should be same as basic est.
    print(
        f"  Selection Rate (selecting any): {results_uniform_advanced['selection_rate']:.4f}"
    )  # Should be same as basic est.
    if results_uniform_advanced["avg_estimator_diff"] is not None:
        print(
            f"  Avg. Estimator Difference (Estimate - True Best, when selected): {results_uniform_advanced['avg_estimator_diff']:.4f}"
        )
    else:
        print(f"  Avg. Estimator Difference: Not applicable (no selections made).")
    print("-" * 30)

    # Compare Estimator Performance on Uniform distribution
    print("Comparison (Basic vs Advanced Estimator on Uniform Distribution):")
    print(
        f"  Basic Estimator Avg. Difference:   {results_uniform_basic['avg_estimator_diff']:.4f}"
    )
    print(
        f"  Advanced Estimator Avg. Difference: {results_uniform_advanced['avg_estimator_diff']:.4f}"
    )
    print("  Discussion:")
    print("  The Advanced Estimator attempts to predict the final maximum based on")
    print("  order statistics, incorporating the number of candidates seen (t) and")
    print("  remaining (n-t). It generally provides a more informed guess than simply")
    print(
        "  using the current maximum (Basic Estimator), especially earlier in the process."
    )
    print(
        "  We expect the Advanced Estimator's average difference to be closer to zero,"
    )
    print("  indicating a potentially more accurate estimation of the true best value,")
    print("  *assuming the U(0,1) distribution holds*.")
    print("  Note: This advanced estimator is specifically tuned for U(0,1). Its")
    print("  performance might degrade significantly if the underlying distribution")
    print("  is different (like in the variable distribution scenario).")
    print("-" * 30)

    # Optional: Run Advanced Estimator on Variable Distribution
    # Note: The advanced estimator implemented assumes U(0,1), so its performance
    # here might be poor or misleading. A truly robust advanced estimator would
    # need to adapt to or be aware of the changing distribution.
    # print(f"Running Scenario 4: Variable Distribution")
    # print("Estimator: Advanced (Order Statistics based for U(0,1))")
    # results_variable_advanced = run_simulation(
    #     N_CANDIDATES, N_TRIALS, K_THRESHOLD,
    #     generate_candidates_variable_uniform, advanced_estimator
    # )
    # print(f"  Success Rate: {results_variable_advanced['success_rate']:.4f}")
    # print(f"  Selection Rate: {results_variable_advanced['selection_rate']:.4f}")
    # if results_variable_advanced['avg_estimator_diff'] is not None:
    #    print(f"  Avg. Estimator Difference: {results_variable_advanced['avg_estimator_diff']:.4f}")
    # else:
    #    print(f"  Avg. Estimator Difference: N/A")
    # print("  (Note: Advanced estimator assumes U(0,1), may perform poorly here)")
    # print("-" * 30)
