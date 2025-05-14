# Project 3

## Overview
This project consists of two main deliverables focused on advanced algorithm analysis, specifically related to fingerprinting and primality testing. The deliverables include Python scripts for empirical rate calculations, interactive visualizations, and benchmarking of deterministic algorithms.

## Deliverable 1
- **Deliverable_1_Part_3.py**: Contains Python code for calculating empirical rates and generating interactive plots related to fingerprinting and adversarial scenarios. It includes functions for prime number generation, empirical rate calculations, and creating interactive plots using Plotly.
- **HTML Files**: 
  - `fingerprint_adaptive_adv_plot_v2.html`: An interactive Plotly plot for the adaptive adversary scenario.
  - `fingerprint_adaptive_adv_plot.html`: Another version of the interactive plot.
  - `fingerprint_comparison_multi_cm.html`: A comparison of fingerprinting methods across multiple configurations.
  - `fingerprint_comparison.html`: A comparison of fingerprinting methods.

## Deliverable 2
- **Miller_Rabin_Deterministic.py**: Implements the Miller-Rabin primality test and benchmarks its performance across different variants.
- **CSV and PNG Files**:
  - `deterministic_benchmark_comparison.csv`: Benchmark comparison data for deterministic algorithms.
  - `deterministic_benchmark_graph.png`: A visual representation of the benchmarking data.
  - `deterministic_primality_tests_results.csv`: Results from deterministic primality tests.
  - `deterministic_variant_examples.csv`: Examples of different variants used in the deterministic algorithms.

## Setup Instructions
1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Install the required dependencies listed in `requirements.txt` using pip:
   ```
   pip install -r requirements.txt
   ```

## Usage Examples
- To run the empirical rate calculations and generate plots, execute the `Deliverable_1_Part_3.py` script:
  ```
  python Deliverable\ 1/Deliverable_1_Part_3.py
  ```
- To benchmark the Miller-Rabin test, run the `Miller_Rabin_Deterministic.py` script:
  ```
  python Deliverable\ 2/Miller_Rabin_Deterministic.py
  ```