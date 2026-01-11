from __future__ import annotations

import argparse
import os
import sys

# Ensure this script's directory is on sys.path so we can import sibling modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import numpy as np

from least_squares import least_squares_normal, least_squares_svd
from make_data import SyntheticConfig, make_synthetic_data, load_real_csv
from validation import check_orthogonality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Least-squares solver demo (synthetic or real data)."
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Use synthetic data or real CSV file.",
    )
    parser.add_argument(
        "--solver",
        choices=["normal", "svd"],
        default="normal",
        help="Least-squares solver to use.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="Noise sigma for synthetic data.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of synthetic samples.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="",
        help="Path to CSV file for real data mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "synthetic":
        cfg = SyntheticConfig(
            n_samples=args.samples,
            noise_sigma=args.noise,
            random_state=42,
        )
        A, b, x_true = make_synthetic_data(cfg)
        print("Using synthetic data.")
        print("True x:", x_true)
    else:
        if not args.csv_path:
            raise SystemExit("Error: --csv-path is required for mode=real.")
        A, b = load_real_csv(args.csv_path)
        x_true = None
        print(f"Loaded real data from {args.csv_path}")
        print(f"A shape: {A.shape}, b length: {b.shape[0]}")

    solver_fn = least_squares_normal if args.solver == "normal" else least_squares_svd

    x_hat = solver_fn(A, b)
    print("\nEstimated x̂:", x_hat)

    ortho = check_orthogonality(A, b, x_hat)
    print(
        f"Orthogonality: max_dot={ortho.max_dot:.2e}, "
        f"mean_dot={ortho.mean_dot:.2e}, passed={ortho.passed}"
    )

    if x_true is not None:
        err = float(np.linalg.norm(x_hat - x_true))
        print(f"Parameter error ||x̂ - x_true||₂ = {err:.4f}")


if __name__ == "__main__":
    main()
