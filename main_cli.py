import argparse
import numpy as np
from least_squares import (
    least_squares_normal,
    least_squares_svd,
    residual_analysis,
    IncrementalLeastSquares,
)

def load_csv(path):
    """
    Load CSV with no headers.
    Assumes columns are: feature1, feature2, ..., feature_n, target
    """
    data = np.loadtxt(path, delimiter=",")
    A = data[:, :-1]       # all columns except last
    b = data[:, -1]        # last column is the target
    return A, b

def main():
    parser = argparse.ArgumentParser(
        description="Least Squares Solver Interface (Task 6)"
    )
    parser.add_argument(
        "--method",
        choices=["normal", "svd", "incremental"],
        default="normal",
        help="Choose solution method."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV file formatted as features...,target."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for incremental mode."
    )
    args = parser.parse_args()

    # Load data
    if args.data:
        A, b = load_csv(args.data)
    else:
        # Fallback synthetic example
        t = np.linspace(0, 10, 100)
        b = 2 + 3 * t + np.random.randn(len(t)) * 0.5
        A = np.column_stack([np.ones_like(t), t])

    if args.method == "normal":
        x = least_squares_normal(A, b)
    elif args.method == "svd":
        x = least_squares_svd(A, b)
    elif args.method == "incremental":
        n_features = A.shape[1]
        inc = IncrementalLeastSquares(n_features)
        # Split into batches
        for i in range(0, A.shape[0], args.batch_size):
            inc.add_batch(A[i:i+args.batch_size], b[i:i+args.batch_size])
        x = inc.solve()
    else:
        raise ValueError(f"Unknown method {args.method}")

    residual, residual_norm = residual_analysis(A, b, x)

    print("\n=== Least Squares Result ===")
    print("Estimated x:", x)
    print("Residual norm:", residual_norm)
    print("=============================\n")

if __name__ == "__main__":
    main()
