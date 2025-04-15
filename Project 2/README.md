# Project Title: Quicksort and Secretary Problem Analysis

## Overview
This project provides a framework for analyzing the performance of Quicksort algorithms with various pivot strategies and exploring solutions to the Secretary Problem using different strategies and estimators. It includes functionality to measure runtime, recursion depth, and balance metrics for Quicksort, as well as success rates and estimation errors for the Secretary Problem.

## Features

### Quicksort Analysis
- Evaluate Quicksort with pivot strategies: First, Last, Middle, Median-of-Three.
- Measure runtime, comparisons, recursion depth, and pivot balance.
- Analyze performance on random, sorted, and nearly sorted arrays.
- Visualize results with detailed plots.
- Save results to CSV files for further analysis.

### Secretary Problem Analysis
- Simulate the Secretary Problem with classical and advanced estimators.
- Analyze success rates and estimation errors.
- Explore performance across different candidate distributions.
- Generate plots for success rates and estimation errors.

## Installation
To set up the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   ```
   *(Replace `<repository-url>` with the actual URL of your repository)*

2. **Navigate to the project directory:**
   ```bash
   cd <project-directory-name>
   ```
   *(Replace `<project-directory-name>` with the name of the folder created by the clone, likely `Project 2` or similar)*

3. **Create and activate a virtual environment** (optional but recommended):
   - **On Windows:**
      ```bash
      python -m venv myenv
      myenv\Scripts\activate
      ```
   - **On macOS/Linux:**
      ```bash
      python3 -m venv myenv
      source myenv/bin/activate
      ```

4. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have a `requirements.txt` file listing the necessary packages like NumPy, Matplotlib, Pandas.)*

## Usage
To run the main scripts and analyze the algorithms, execute the following commands from the project's root directory:

### Quicksort Analysis
```bash
python Deliverable\1\part_1_quicksort_pivots.py
python Deliverable\1\part_2_distribution_analysis.py
python Deliverable\1\part_2_nearly_sorted_analysis.py
python Deliverable\1\part_3_random_arrays_and_plotting.py
```

### Secretary Problem Analysis
```bash
python Deliverable\2\parts_1_and_2_secretary_simulation.py
python Deliverable\2\part3_secretary_analysis.py
```

## Explanation of Key Files

### Deliverable 1: Quicksort Analysis
- **`part_1_quicksort_pivots.py`**: Implements and evaluates Quicksort with different pivot strategies, collecting metrics like runtime, comparisons, and recursion depth.
- **`part_2_distribution_analysis.py`**: Analyzes Quicksort performance on arrays generated from various distributions (e.g., uniform, normal, exponential, sorted).
- **`part_2_nearly_sorted_analysis.py`**: Evaluates Quicksort performance on nearly sorted arrays with varying noise levels.
- **`part_3_random_arrays_and_plotting.py`**: Runs experiments on random arrays and generates plots for runtime, comparisons, and recursion depth.

### Deliverable 2: Secretary Problem Analysis
- **`parts_1_and_2_secretary_simulation.py`**: Simulates the Secretary Problem using classical and estimator-based strategies, analyzing success rates and estimation errors.
- **`part3_secretary_analysis.py`**: Explores advanced estimators and compares their performance on different candidate distributions.

## Results
- Results are saved as CSV files in the respective directories for further analysis.
- Plots are generated to visualize performance metrics and saved as PNG files.

## Dependencies
- Python 3.x
- NumPy
- Matplotlib
- Pandas

## Notes
- Ensure sufficient recursion depth for large arrays in Quicksort experiments by adjusting `sys.setrecursionlimit` if necessary.
- For Secretary Problem simulations, modify parameters like the number of candidates (n) and trials as needed.

## License
This project is for educational purposes as part of CS 5080 Advanced Algorithm Analysis.
