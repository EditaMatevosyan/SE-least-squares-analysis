# SE Least Squares Analysis

This repository contains the implementation for **Project 44 – Model Calibration and Trend Estimation Using Least Squares Approximation** for the Software Engineering course.

The system demonstrates:
- linear least-squares modeling of traffic sensor data,
- residual and error analysis,
- incremental / streaming model updates,
- solver robustness testing under noise, and
- validation through orthogonality checks.

---

## Features

- Solve least squares using normal equations
- Solve using SVD pseudoinverse (numerical stability)
- Compute residual vectors and Euclidean norms
- Validate orthogonality of residuals
- Perform incremental / streaming updates
- Test solver robustness under noise
- Load and analyze real-world CSV data
- Generate figures for the project report

---

## Running the Code

Run all experiments (produces figures):
python3 main.py

Synthetic mode (CLI):
python3 main_cli.py --mode synthetic --solver normal

Real dataset mode:
python3 main_cli.py --mode real --csv-path my_traffic.csv --solver svd

---

## Output Files

Generated in the figures/ directory:
- param_error_vs_noise.png
- residual_norm_vs_noise.png
- streaming_convergence.png

---

## Real Data Format

CSV must contain:
timestamp,traffic_flow,temperature

Example:
2025-01-01 08:00:00,120,6.2

Timestamp is converted to:
- hour of day (numeric feature)
- weekend indicator (0/1)

Temperature may be omitted.

---

## Mathematical Background

The project solves:
minimize_x ||Ax - b||²

Where:
- A is the measurement matrix
- b is the observation vector
- x̂ is the estimated parameter vector

Solvers:
- Normal equations: (AᵀA)x = Aᵀb
- SVD pseudoinverse: x̂ = pinv(A)b

Residual orthogonality condition:
Aᵀ(b – A x̂) ≈ 0

Incremental solver maintains:
AᵀA and Aᵀb
across batches.

---

## Validation Summary

- Recovers true parameters on synthetic data
- Incremental estimates converge toward batch solution
- Noise increases error in a predictable way
- Normal and SVD solvers give matching results when A is well conditioned
- Real data performance depends on dataset size and variation

---

## Requirements

Python 3.8+

Install dependencies:
pip install numpy matplotlib

---

## Author

Developed by **Edita Matevosyan (2026)**
Software Engineering Project 44
