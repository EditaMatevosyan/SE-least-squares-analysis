import numpy as np
from least_squares import (
    least_squares_normal,
    least_squares_svd,
    orthogonality_check,
)

def noise_sensitivity_experiment(
    noise_levels=(0.1, 0.5, 1.0, 2.0),
    n_trials=30,
    random_seed=0,
):
    """
    Compare normal-equation and SVD solvers under increasing noise.

    For each noise level σ:
      - generate synthetic data y = 2 + 3 t + N(0, σ^2)
      - estimate parameters with both solvers
      - record parameter error and residual norms
      - average over n_trials
    """
    rng = np.random.default_rng(random_seed)

    # True parameters
    beta_true = np.array([2.0, 3.0])

    # Fixed design (t values)
    t = np.linspace(0, 10, 50)
    A = np.column_stack([np.ones_like(t), t])

    results = []

    for sigma in noise_levels:
        err_norm_normal = []
        resid_norm_normal = []

        err_norm_svd = []
        resid_norm_svd = []

        for _ in range(n_trials):
            # Generate noisy observations
            noise = rng.normal(0.0, sigma, size=t.shape)
            y = beta_true[0] + beta_true[1] * t + noise

            # Normal equations
            x_n = least_squares_normal(A, y)
            r_n, rnorm_n, _ = orthogonality_check(A, y, x_n)
            err_norm_normal.append(np.linalg.norm(x_n - beta_true))
            resid_norm_normal.append(rnorm_n)

            # SVD
            x_s = least_squares_svd(A, y)
            r_s, rnorm_s, _ = orthogonality_check(A, y, x_s)
            err_norm_svd.append(np.linalg.norm(x_s - beta_true))
            resid_norm_svd.append(rnorm_s)

        results.append({
            "sigma": sigma,
            "param_err_normal": np.mean(err_norm_normal),
            "resid_norm_normal": np.mean(resid_norm_normal),
            "param_err_svd": np.mean(err_norm_svd),
            "resid_norm_svd": np.mean(resid_norm_svd),
        })

    # Print summary table
    print("Noise sensitivity experiment (averaged over", n_trials, "trials):\n")
    print(f"{'σ':>6} | {'||x_n - x*||':>14} | {'||r_n||':>10} | {'||x_s - x*||':>14} | {'||r_s||':>10}")
    print("-" * 65)
    for res in results:
        print(f"{res['sigma']:6.2f} | "
              f"{res['param_err_normal']:14.6f} | "
              f"{res['resid_norm_normal']:10.6f} | "
              f"{res['param_err_svd']:14.6f} | "
              f"{res['resid_norm_svd']:10.6f}")

if __name__ == "__main__":
    noise_sensitivity_experiment()
