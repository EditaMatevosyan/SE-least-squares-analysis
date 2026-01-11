import numpy as np

# Task 2: Implement normal-equation-based solver
class LeastSquaresError(Exception):
    """Custom exception for least-squares related errors."""
    pass


def least_squares_normal(A, b):
    """
    Compute the least-squares solution x that minimizes ||Ax - b||_2^2
    using the normal equations: (A^T A) x = A^T b.

    Parameters
    ----------
    A : array-like, shape (m, n)
        Measurement matrix (rows = observations, columns = features).
    b : array-like, shape (m,)
        Observation vector.

    Returns
    -------
    x : np.ndarray, shape (n,)
        Estimated parameter vector.

    Raises
    ------
    LeastSquaresError
        If the input shapes are invalid or the system is not overdetermined.
    numpy.linalg.LinAlgError
        If (A^T A) is singular or nearly singular and cannot be solved.
    """
    # Convert inputs to NumPy arrays of type float
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # Basic shape checks
    if A.ndim != 2:
        raise LeastSquaresError("A must be a 2D matrix.")
    if b.ndim == 2 and b.shape[1] == 1:
        b = b.reshape(-1)
    elif b.ndim != 1:
        raise LeastSquaresError("b must be a 1D vector or a column vector.")

    m, n = A.shape
    if b.shape[0] != m:
        raise LeastSquaresError(
            f"Dimension mismatch: A has {m} rows but b has length {b.shape[0]}."
        )

    if m < n:
        # Under-determined system – not the usual least-squares scenario
        raise LeastSquaresError(
            f"System is not overdetermined: got m = {m}, n = {n} (need m >= n)."
        )

    # Step 1: Compute normal matrix N = A^T A
    ATA = A.T @ A     # shape (n, n)

    # Step 2: Compute right-hand side c = A^T b
    ATb = A.T @ b     # shape (n,)

    # Step 3: Solve (A^T A) x = A^T b
    x = np.linalg.solve(ATA, ATb)

    return x

# Optional: Least squares using SVD (part of Task 2/4 enhancement)
def least_squares_svd(A, b, tol=1e-10):
    """
    Least-squares solution using SVD (A = U Σ V^T).

    Parameters
    ----------
    A : array-like, shape (m, n)
        Measurement matrix.
    b : array-like, shape (m,)
        Observation vector.
    tol : float
        Tolerance for treating singular values as zero.

    Returns
    -------
    x : np.ndarray, shape (n,)
        Estimated parameter vector.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    # Compute thin SVD: A = U Σ Vt
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # Pseudoinverse step: invert non-zero singular values
    s_inv = np.array([1/s_i if s_i > tol else 0 for s_i in s])

    # LS solution: x = V Σ⁺ U^T b
    x = Vt.T @ (s_inv * (U.T @ b))
    return x


# Task 3: Error and approximation analysis module
def residual_analysis(A, b, x):
    """
    Compute residual vector and residual norm for Ax ≈ b.

    Parameters
    ----------
    A : array-like, shape (m, n)
    b : array-like, shape (m,)
    x : array-like, shape (n,)

    Returns
    -------
    residual : np.ndarray, shape (m,)
        Residual vector r = Ax - b.
    residual_norm : float
        Euclidean norm ||r||_2.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)

    residual = A @ x - b
    residual_norm = float(np.linalg.norm(residual, ord=2))

    return residual, residual_norm


# Task 4: Incremental / streaming least squares estimator
class IncrementalLeastSquares:
    """
    Incremental / streaming least squares estimator based on normal equations.

    Maintains N = A^T A and c = A^T b and updates them when new data batches arrive.
    """

    def __init__(self, n_features):
        """
        Parameters
        ----------
        n_features : int
            Number of features (columns of A), including intercept if used.
        """
        self.n_features = n_features
        # Initialize N = A^T A and c = A^T b to zeros
        self.N = np.zeros((n_features, n_features), dtype=float)
        self.c = np.zeros(n_features, dtype=float)
        self.m_total = 0  # total number of observations seen

    def add_batch(self, A_batch, b_batch):
        """
        Add a new batch of data and update N and c.

        Parameters
        ----------
        A_batch : array-like, shape (k, n_features)
            New rows of the measurement matrix.
        b_batch : array-like, shape (k,)
            Corresponding observations.
        """
        A_batch = np.asarray(A_batch, dtype=float)
        b_batch = np.asarray(b_batch, dtype=float).reshape(-1)

        k, n = A_batch.shape
        if n != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {n}."
            )
        if b_batch.shape[0] != k:
            raise ValueError(
                f"Batch size mismatch: A_batch has {k} rows, b_batch has {b_batch.shape[0]} entries."
            )

        # Update aggregated normal matrix and right-hand side
        self.N += A_batch.T @ A_batch
        self.c += A_batch.T @ b_batch
        self.m_total += k

    def solve(self):
        """
        Solve for the current least squares estimate x.

        Returns
        -------
        x : np.ndarray, shape (n_features,)
            Current parameter estimate based on all data seen so far.
        """
        if self.m_total == 0:
            raise RuntimeError("No data has been added yet.")

        # Solve N x = c
        x = np.linalg.solve(self.N, self.c)
        return x
    

#Task 5: Orthogonality check for least squares
def orthogonality_check(A, b, x):
    """
    Check the orthogonality condition A^T r ≈ 0 for least squares residuals.

    Parameters
    ----------
    A : array-like, shape (m, n)
    b : array-like, shape (m,)
    x : array-like, shape (n,)

    Returns
    -------
    residual : np.ndarray, shape (m,)
        Residual vector r = Ax - b.
    residual_norm : float
        Norm ||r||_2.
    At_r_norm : float
        Norm ||A^T r||_2, should be close to 0 for a correct LS solution.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)

    # Residual
    residual = A @ x - b
    residual_norm = float(np.linalg.norm(residual, ord=2))

    # Orthogonality measure
    At_r = A.T @ residual
    At_r_norm = float(np.linalg.norm(At_r, ord=2))

    return residual, residual_norm, At_r_norm

