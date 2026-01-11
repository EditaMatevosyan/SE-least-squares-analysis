from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Callable

import numpy as np


@dataclass
class OrthogonalityResult:
    max_dot: float
    mean_dot: float
    passed: bool


@dataclass
class NoiseStats:
    mean_param_error: float
    mean_residual_norm: float


def check_orthogonality(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    tol: float = 1e-10,
) -> OrthogonalityResult:
    """
    Check orthogonality of residual r = b - Ax to the column space of A.

    Returns
    -------
    OrthogonalityResult
        max_dot, mean_dot, passed flag.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)

    r = b - A @ x  # residual
    dots = A.T @ r  # dot products with each column

    abs_dots = np.abs(dots)
    max_dot = float(abs_dots.max())
    mean_dot = float(abs_dots.mean())
    passed = max_dot <= tol

    return OrthogonalityResult(max_dot=max_dot, mean_dot=mean_dot, passed=passed)


def run_noise_sensitivity(
    sigmas: np.ndarray,
    n_trials: int,
    make_data_fn: Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]],
    solver_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Dict[float, NoiseStats]:
    """
    Run repeated experiments for different noise levels and summarize results.

    Parameters
    ----------
    sigmas : np.ndarray
        Array of noise standard deviations.
    n_trials : int
        Number of trials per sigma.
    make_data_fn : callable
        Function taking sigma -> (A, b, x_true).
    solver_fn : callable
        Function taking (A, b) -> x_hat.

    Returns
    -------
    Dict[float, NoiseStats]
        Mapping noise_sigma -> stats.
    """
    results: Dict[float, NoiseStats] = {}

    for sigma in sigmas:
        param_errors: list[float] = []
        residual_norms: list[float] = []

        for _ in range(n_trials):
            A, b, x_true = make_data_fn(float(sigma))
            x_hat = solver_fn(A, b)
            param_errors.append(float(np.linalg.norm(x_hat - x_true)))
            residual_norms.append(float(np.linalg.norm(b - A @ x_hat)))

        results[float(sigma)] = NoiseStats(
            mean_param_error=float(np.mean(param_errors)),
            mean_residual_norm=float(np.mean(residual_norms)),
        )

    return results
