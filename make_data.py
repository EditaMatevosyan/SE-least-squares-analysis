from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Iterable

import numpy as np


@dataclass
class SyntheticConfig:
    n_samples: int = 200
    noise_sigma: float = 0.1
    random_state: int | None = None


def make_synthetic_data(config: SyntheticConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic traffic-flow data with features:
        [1, time_of_day_norm, is_weekend, temperature_norm]

    Parameters
    ----------
    config : SyntheticConfig
        n_samples, noise_sigma, random_state.

    Returns
    -------
    A : np.ndarray of shape (m, 4)
        Measurement matrix.
    b : np.ndarray of shape (m,)
        Observations.
    x_true : np.ndarray of shape (4,)
        Ground-truth parameter vector.
    """
    rng = np.random.default_rng(config.random_state)

    n = config.n_samples

    # time_of_day in hours [0, 23], normalized to [0, 1]
    time_of_day = rng.integers(0, 24, size=n)
    time_of_day_norm = time_of_day / 23.0

    # is_weekend ∈ {0, 1}
    is_weekend = rng.integers(0, 2, size=n)

    # temperature in °C, say between -5 and 35, normalized
    temperature = rng.uniform(-5.0, 35.0, size=n)
    temperature_norm = (temperature - (-5.0)) / (35.0 - (-5.0))

    # true parameters: [bias, time_of_day_coef, weekend_coef, temperature_coef]
    x_true = np.array([20.0, 15.0, 10.0, 5.0], dtype=float)

    # Build design matrix A
    A = np.column_stack(
        [
            np.ones(n, dtype=float),
            time_of_day_norm,
            is_weekend.astype(float),
            temperature_norm,
        ]
    )

    # Clean outputs
    b_clean = A @ x_true

    # Add Gaussian noise
    noise = rng.normal(loc=0.0, scale=config.noise_sigma, size=n)
    b = b_clean + noise

    return A, b, x_true


def parse_timestamp_to_features(timestamp_str: str) -> Tuple[float, int]:
    """
    Convert timestamp string to (time_of_day_norm, is_weekend).

    Assumes ISO-like strings: 'YYYY-MM-DD HH:MM:SS' or close.
    """
    # You can adjust the format to match your real data
    dt = datetime.fromisoformat(timestamp_str.strip().replace("T", " "))
    hour = dt.hour
    weekday = dt.weekday()  # 0=Mon, ..., 5=Sat, 6=Sun
    time_of_day_norm = hour / 23.0
    is_weekend = 1 if weekday >= 5 else 0
    return time_of_day_norm, is_weekend


def load_real_csv(
    path: str,
    has_header: bool = True,
    delimiter: str = ",",
    timestamp_col: int = 0,
    traffic_col: int = 1,
    temp_col: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real traffic data from a CSV file and build (A, b).

    Expected columns (you can adapt indices):
        timestamp, traffic_flow, [temperature]

    Parameters
    ----------
    path : str
        Path to CSV file.
    has_header : bool
        Whether the first line is a header.
    delimiter : str
        Column separator.
    timestamp_col : int
        Index of timestamp column.
    traffic_col : int
        Index of traffic flow column.
    temp_col : int or None
        Index of temperature column (if None, use synthetic temperature=20°C).

    Returns
    -------
    A : np.ndarray of shape (m, 4)
    b : np.ndarray of shape (m,)
    """
    import csv

    rows: list[list[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader, None)
        for row in reader:
            if not row:
                continue
            rows.append(row)

    time_norms: list[float] = []
    weekends: list[int] = []
    temps: list[float] = []
    traffic: list[float] = []

    for row in rows:
        ts = row[timestamp_col]
        t_norm, is_weekend = parse_timestamp_to_features(ts)
        time_norms.append(t_norm)
        weekends.append(is_weekend)

        if temp_col is not None:
            temp_val = float(row[temp_col])
        else:
            temp_val = 20.0  # fallback constant
        temps.append(temp_val)

        traffic_val = float(row[traffic_col])
        traffic.append(traffic_val)

    time_norms_arr = np.asarray(time_norms, dtype=float)
    weekends_arr = np.asarray(weekends, dtype=float)
    temps_arr = np.asarray(temps, dtype=float)
    traffic_arr = np.asarray(traffic, dtype=float)

    # normalize temp like before
    t_min, t_max = temps_arr.min(), temps_arr.max()
    if t_max > t_min:
        temp_norm = (temps_arr - t_min) / (t_max - t_min)
    else:
        temp_norm = np.zeros_like(temps_arr)

    n = len(traffic_arr)
    A = np.column_stack(
        [
            np.ones(n, dtype=float),
            time_norms_arr,
            weekends_arr,
            temp_norm,
        ]
    )
    b = traffic_arr

    return A, b
