# Project Title: CH and TNR Metrics Comparison

## Overview
This project provides a framework for measuring and comparing various metrics related to query performance. It includes functionality to track memory usage and display results in a tabular format, making it easier to analyze the efficiency of different query ordering methods.

## Features
- Measure preprocessing time and memory usage.
- Measure query time and memory usage.
- Display results in a clear, tabular format.
- Save results to a CSV file for further analysis.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```
   cd project-1
   ```

3. It is recommended to create a virtual environment:
   ```
   python -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
   ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the main script and measure query metrics, execute the following command:
```
python Project\ 1/ch_metrics.py
python Project\ 1/tnr_metrics.py
python Project\ 1/tnr_metrics_unmodified.py
python Project\ 1/ch_hw_graph.py
python Project\ 1/tnr_falcon.py
python Project\ 1/tnr_hw_graph.py
```

### Explanation of Runnable Files

- `ch_metrics.py`: Measures and compares various metrics related to Contraction Hierarchies (CH) using different ordering methods.
- `tnr_metrics.py`: Measures and compares various metrics related to Transit Node Routing (TNR) using different ordering methods.
- `tnr_metrics_unmodified.py`: Similar to `tnr_metrics.py` but uses an Andy's CH version of the TNR algorithm for comparison.
- `ch_hw_graph.py`: Runs CH metrics on the homework graph.
- `tnr_falcon.py`: Runs TNR metrics on the road network of Falcon, Colorado.
- `tnr_hw_graph.py`: Runs TNR metrics on the homework graph.