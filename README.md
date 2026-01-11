SE Least Squares Analysis

This repository contains the implementation and experiments for Project 44 – Model Calibration and Trend Estimation Using Least Squares Approximation in the Software Engineering course.

The goal is to build a modular software system that:

models smart-city traffic sensor data with a linear least squares model,

computes residuals and error metrics,

supports incremental / streaming updates,

validates residual orthogonality and model stability, and

evaluates solver robustness under measurement noise.

Repository Structure
SE-least-squares-analysis/
├─ figures/               # Saved plots (created automatically by main.py)
├─ least_squares.py       # Normal-equation solver, SVD solver, incremental LS
├─ validation.py          # Orthogonality + robustness checks
├─ make_data.py           # Synthetic dataset generator + CSV loader
├─ main.py                # Full experiment runner (generates all plots)
├─ main_cli.py            # Command-line interface
├─ my_traffic.csv         # Optional real dataset (user-provided)
└─ README.md


Note: Import paths are flat — modules are imported directly (no src/ folder).

Features

Solve least squares using normal equations

Solve with SVD pseudoinverse for numerical stability

Compute residuals and Euclidean residual norms

Validate orthogonality of residuals to column space

Support streaming updates (incremental batches)

Stress test solver with increasing noise

Load real sensor data from CSV

Generate presentation-quality graphs

Running the Code
Full experiment suite (with figures)
python3 main.py

Command-line interface (synthetic)
python3 main_cli.py --mode synthetic --solver normal

Command-line interface (real CSV)
python3 main_cli.py --mode real --csv-path my_traffic.csv --solver svd

Output Files (created automatically)

Generated inside the figures/ directory:

param_error_vs_noise.png

residual_norm_vs_noise.png

streaming_convergence.png

These figures correspond to Section 7 (Results & Discussion) of the written report.

Real Data Format

Your CSV must contain:

timestamp,traffic_flow,temperature


Example:

2025-01-01 08:00:00,120,6.2
2025-01-01 09:00:00,150,7.0
2025-01-01 10:00:00,175,9.1


If temperature is missing, a fallback constant value is used.

Timestamp is processed to compute:

hour-of-day numeric feature, and

weekend indicator (0/1).

Mathematical Formulation

We solve:

minimize_x ||Ax - b||²


Where:

A = feature matrix (observations × variables)

b = measured sensor output

x̂ = estimated parameter vector

Two solvers are provided:

Normal equations: (AᵀA)x = Aᵀb

SVD pseudoinverse: x̂ = pinv(A)b

Residual orthogonality check verifies:

Aᵀ(b – A x̂) ≈ 0


Incremental estimation maintains:

AᵀA and Aᵀb


across batches, without storing all past data.

Validation Summary

Synthetic data recovers true parameters accurately

Streaming updates converge toward batch solution

Noise sensitivity increases error and residuals smoothly

Normal and SVD solvers perform similarly when data is well-conditioned

Real data tests produce meaningful least-squares estimates (interpretation limited by dataset size)

Requirements

Python 3.8+
Dependencies:

pip install numpy matplotlib

Academic Context

This repository supports the following report sections:

Mathematical Background

System Design

Solver Implementation

Streaming Extension

Testing & Validation

Results & Discussion with Figures

Author & Usage

Developed by Edita Matevosyan (2026)
for Software Engineering Project 44

Reuse permitted with attribution.
