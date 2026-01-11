from __future__ import annotations

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import numpy as np

from least_squares import (
    least_squares_normal,
    least_squares_svd,
    compute_residual,
    IncrementalLeastSquares,
)
from make_data import SyntheticConfig, make_synthetic_data
from validation import check_orthogonality, run_noise_sensitivity


def ensure_figures_dir(dir_name: str = "figures") -> str:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    return dir_name


def plot_noise_results(results_normal, results_svd, fig_dir: str) -> None:
    import matplotlib.pyplot as plt

    sigmas = sorted(results_normal.keys())
    sig_arr = np.array(sigmas, dtype=float)

    param_err_normal = np.array(
        [results_normal[s].mean_param_error for s in sigmas], dtype=float
    )
    res_norm_normal = np.array(
        [results_normal[s].mean_residual_norm for s in sigmas], dtype=float
    )

    param_err_svd = np.array(
        [results_svd[s].mean_param_error for s in sigmas], dtype=float
    )
    res_norm_svd = np.array(
        [results_svd[s].mean_residual_norm for s in sigmas], dtype=float
    )

    # Parameter error vs noise
    plt.figure()
    plt.plot(sig_arr, param_err_normal, marker="o", label="Normal equations")
    plt.plot(sig_arr, param_err_svd, marker="x", label="SVD")
    plt.xlabel("Noise standard deviation σ")
    plt.ylabel("Mean parameter error ||x̂ - x_true||₂")
    plt.title("Parameter error vs noise level")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "param_error_vs_noise.png"), dpi=300)

    # Residual norm vs noise
    plt.figure()
    plt.plot(sig_arr, res_norm_normal, marker="o", label="Normal equations")
    plt.plot(sig_arr, res_norm_svd, marker="x", label="SVD")
    plt.xlabel("Noise standard deviation σ")
    plt.ylabel("Mean residual norm ||b - Ax̂||₂")
    plt.title("Residual norm vs noise level")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "residual_norm_vs_noise.png"), dpi=300)


def plot_streaming_convergence(x_true, x_batch1, x_batch2, fig_dir: str) -> None:
    import matplotlib.pyplot as plt

    indices = np.arange(len(x_true))

    width = 0.25

    plt.figure()
    plt.bar(indices - width, x_true, width=width, label="True x")
    plt.bar(indices, x_batch1, width=width, label="After batch 1")
    plt.bar(indices + width, x_batch2, width=width, label="After batch 2")
    plt.xticks(indices, [f"x{i}" for i in range(len(x_true))])
    plt.ylabel("Parameter value")
    plt.title("Streaming convergence of parameter estimates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "streaming_convergence.png"), dpi=300)


def main() -> None:
    fig_dir = ensure_figures_dir()

    # 1) Basic synthetic experiment
    config = SyntheticConfig(n_samples=200, noise_sigma=0.1, random_state=42)
    A, b, x_true = make_synthetic_data(config)

    x_hat_normal = least_squares_normal(A, b)
    r_normal, r_norm_normal = compute_residual(A, x_hat_normal, b)
    ortho_normal = check_orthogonality(A, b, x_hat_normal)

    x_hat_svd = least_squares_svd(A, b)
    r_svd, r_norm_svd = compute_residual(A, x_hat_svd, b)
    ortho_svd = check_orthogonality(A, b, x_hat_svd)

    print("=== Basic Synthetic Experiment ===")
    print("True x:        ", x_true)
    print("Normal x̂:     ", x_hat_normal)
    print("SVD x̂:        ", x_hat_svd)
    print("Normal residual norm:", r_norm_normal)
    print("SVD residual norm:   ", r_norm_svd)
    print(
        "Orthogonality (normal): "
        f"max_dot={ortho_normal.max_dot:.2e}, "
        f"mean_dot={ortho_normal.mean_dot:.2e}, "
        f"passed={ortho_normal.passed}"
    )
    print(
        "Orthogonality (SVD):    "
        f"max_dot={ortho_svd.max_dot:.2e}, "
        f"mean_dot={ortho_svd.mean_dot:.2e}, "
        f"passed={ortho_svd.passed}"
    )

    # 2) Streaming / incremental experiment with two batches
    config1 = SyntheticConfig(n_samples=100, noise_sigma=0.1, random_state=1)
    config2 = SyntheticConfig(n_samples=100, noise_sigma=0.1, random_state=2)

    A1, b1, x_true_1 = make_synthetic_data(config1)
    A2, b2, x_true_2 = make_synthetic_data(config2)

    # we use the same x_true in both configs by design, so just reuse x_true_1
    ils = IncrementalLeastSquares(n_features=A1.shape[1])
    ils.add_batch(A1, b1)
    x_batch1 = ils.solve()

    ils.add_batch(A2, b2)
    x_batch2 = ils.solve()

    print("\n=== Streaming / Incremental Experiment ===")
    print("True x:           ", x_true_1)
    print("After batch 1 x̂: ", x_batch1)
    print("After batch 2 x̂: ", x_batch2)

    plot_streaming_convergence(x_true_1, x_batch1, x_batch2, fig_dir)

    # 3) Noise sensitivity experiment for normal vs SVD
    print("\n=== Noise Sensitivity Experiments ===")

    sigmas = np.array([0.0, 0.05, 0.1, 0.2, 0.5])

    def make_data_for_sigma(sigma: float):
        cfg = SyntheticConfig(n_samples=200, noise_sigma=sigma, random_state=123)
        return make_synthetic_data(cfg)

    results_normal = run_noise_sensitivity(
        sigmas=sigmas,
        n_trials=30,
        make_data_fn=make_data_for_sigma,
        solver_fn=least_squares_normal,
    )
    results_svd = run_noise_sensitivity(
        sigmas=sigmas,
        n_trials=30,
        make_data_fn=make_data_for_sigma,
        solver_fn=least_squares_svd,
    )

    for sigma in sigmas:
        stats_n = results_normal[float(sigma)]
        stats_s = results_svd[float(sigma)]
        print(
            f"σ={sigma:.2f} | "
            f"Normal: param_err={stats_n.mean_param_error:.4f}, "
            f"res_norm={stats_n.mean_residual_norm:.4f} | "
            f"SVD: param_err={stats_s.mean_param_error:.4f}, "
            f"res_norm={stats_s.mean_residual_norm:.4f}"
        )

    plot_noise_results(results_normal, results_svd, fig_dir)
    print(f"\nPlots saved in directory: {fig_dir}/")


if __name__ == "__main__":
    main()
