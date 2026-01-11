from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


class LeastSquaresError(Exception):
    """Custom exception for least-squares related errors."""
    pass


def least_squares_normal(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve min_x ||Ax - b||_2^2 via the normal equations (A^T A)x = A^T b.

    Parameters
    ----------
    A : np.ndarray of shape (m, n)
        Measurement matrix (rows = observations, columns = features).
    b : np.ndarray of shape (m,)
        Observation vector.

    Returns
    -------
    x : np.ndarray of shape (n,)
        Least-squares estimate.

    Raises
    ------
    LeastSquaresError
        If shapes are invalid or the system is not overdetermined (m <= n).
    numpy.linalg.LinAlgError
        If A^T A is singular or nearly singular.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2:
        raise LeastSquaresError("A must be a 2D matrix.")
    if b.ndim != 1:
        raise LeastSquaresError("b must be a 1D vector.")
    m, n = A.shape
    if b.shape[0] != m:
        raise LeastSquaresError(
            f"Shape mismatch: A has {m} rows but b has length {b.shape[0]}."
        )
    if m <= n:
        raise LeastSquaresError(
            f"System is not overdetermined: m={m}, n={n}. Need m > n."
        )

    ATA = A.T @ A
    ATb = A.T @ b

    x = np.linalg.solve(ATA, ATb)
    return x


def least_squares_svd(A: np.ndarray, b: np.ndarray, rcond: float | None = None) -> np.ndarray:
    """
    Solve min_x ||Ax - b||_2^2 via SVD-based pseudoinverse.

    Parameters
    ----------
    A : np.ndarray of shape (m, n)
        Measurement matrix.
    b : np.ndarray of shape (m,)
        Observation vector.
    rcond : float or None
        Cutoff for small singular values (passed to numpy.linalg.pinv).

    Returns
    -------
    x : np.ndarray of shape (n,)
        Least-squares estimate.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2:
        raise LeastSquaresError("A must be a 2D matrix.")
    if b.ndim != 1:
        raise LeastSquaresError("b must be a 1D vector.")
    m, n = A.shape
    if b.shape[0] != m:
        raise LeastSquaresError(
            f"Shape mismatch: A has {m} rows but b has length {b.shape[0]}."
        )

    A_pinv = np.linalg.pinv(A, rcond=rcond)
    x = A_pinv @ b
    return x


def compute_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute residual vector r = b - Ax and its Euclidean norm.

    Parameters
    ----------
    A : np.ndarray of shape (m, n)
    x : np.ndarray of shape (n,)
    b : np.ndarray of shape (m,)

    Returns
    -------
    r : np.ndarray of shape (m,)
        Residual vector.
    r_norm : float
        Euclidean norm of the residual vector.
    """
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    b = np.asarray(b, dtype=float)

    r = b - A @ x
    r_norm = float(np.linalg.norm(r, ord=2))
    return r, r_norm


@dataclass
class IncrementalState:
    ATA: np.ndarray
    ATb: np.ndarray
    n_obs: int


class IncrementalLeastSquares:
    """
    Incremental least squares using aggregated normal equations.

    Maintains:
        ATA = sum_i a_i^T a_i
        ATb = sum_i a_i^T b_i
    where a_i is a row of A, b_i the corresponding observation.
    """

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self.ATA = np.zeros((n_features, n_features), dtype=float)
        self.ATb = np.zeros(n_features, dtype=float)
        self.n_obs = 0

    def add_batch(self, A_batch: np.ndarray, b_batch: np.ndarray) -> None:
        """
        Incorporate a new batch of observations.

        Parameters
        ----------
        A_batch : np.ndarray of shape (m_batch, n_features)
        b_batch : np.ndarray of shape (m_batch,)
        """
        A_batch = np.asarray(A_batch, dtype=float)
        b_batch = np.asarray(b_batch, dtype=float)

        if A_batch.ndim != 2:
            raise LeastSquaresError("A_batch must be a 2D matrix.")
        if b_batch.ndim != 1:
            raise LeastSquaresError("b_batch must be a 1D vector.")
        m_batch, n = A_batch.shape
        if n != self.n_features:
            raise LeastSquaresError(
                f"Feature mismatch: expected {self.n_features}, got {n}."
            )
        if b_batch.shape[0] != m_batch:
            raise LeastSquaresError(
                f"Shape mismatch: A_batch has {m_batch} rows but "
                f"b_batch has length {b_batch.shape[0]}."
            )

        self.ATA += A_batch.T @ A_batch
        self.ATb += A_batch.T @ b_batch
        self.n_obs += m_batch

    def solve(self) -> np.ndarray:
        """
        Compute the current least-squares estimate x using aggregated ATA, ATb.

        Returns
        -------
        x : np.ndarray of shape (n_features,)
        """
        if self.n_obs == 0:
            raise LeastSquaresError("No observations have been added yet.")
        x = np.linalg.solve(self.ATA, self.ATb)
        return x

    def get_state(self) -> IncrementalState:
        """Return a snapshot of the internal state (for debugging or logging)."""
        return IncrementalState(ATA=self.ATA.copy(), ATb=self.ATb.copy(), n_obs=self.n_obs)
