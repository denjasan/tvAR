#!/usr/bin/env python3
"""
Baseline TVAR (time-varying AR) for PhysioNet AFDB (MIT-BIH Atrial Fibrillation Database).
- Builds an RR-interval time series (tachogram) from WFDB annotation files (.qrs / .qrsc / .atr).
- Regularizes it to an equidistant time grid (default: 1.0 second).
- Fits a TVAR(p) model in a walk-forward (one-step-ahead) manner:
    1) Sliding-window (uniform weights)
    2) One-sided Gaussian kernel (causal, exponentially downweights older samples)

The dataset must be placed in ./dataset/ next to this script.
Example files for a record "04043":
    ./dataset/04043.hea
    ./dataset/04043.dat
    ./dataset/04043.qrs  (or .qrsc / .atr)

This script is intentionally verbose: prints intermediate info and produces plots.

Notes:
- For the baseline forecasting model we do NOT need to read the ECG signal (.dat).
  We only need:
    - sampling frequency (from .hea)
    - beat timestamps (from .qrs/.qrsc/.atr)
- WFDB annotation format parsing is implemented minimally but robustly enough
  to extract beat sample indices and (optionally) AUX strings.

Run (single record):
    python tvar_afdb_class.py --dataset ./dataset --record 00735 --model sliding --verbose --make_plots
    python tvar_afdb_class.py --dataset ./dataset --record 00735 --model kernel --verbose --make_plots
    python base_tvAR.py --dataset ./dataset --record 00735 --model gam --gam_n_splines 25 --gam_smooth 10 --verbose --make_plots

Run (all records in dataset):
    python tvar_afdb_class.py --dataset ./dataset --all_records --model sliding --make_plots --verbose
    python tvar_afdb_class.py --dataset ./dataset --all_records --model kernel --make_plots --verbose

Outputs:
    ./outputs/<record>_*.png

================================================================================
ABBREVIATIONS (printed in console output and used in plots)
================================================================================
AFDB  : MIT-BIH Atrial Fibrillation Database (PhysioNet dataset of long ECG records with AF episodes).
ECG   : Electrocardiogram (electrical signal of the heart). Stored in .dat, but not used directly here.
WFDB  : WaveForm DataBase format (PhysioNet file format family: .hea header, .dat signal, .qrs/.atr annotations).
QRS   : QRS complex in ECG (ventricular depolarization); its timestamp corresponds to a heartbeat.
RR    : Interval between two consecutive R-peaks (heartbeat-to-heartbeat interval), measured in seconds.
TVAR  : Time-Varying AutoRegressive model. AR coefficients a_i are allowed to change over time.
AR(p) : AutoRegressive model of order p:
          x[t] = c + sum_{i=1..p} a_i * x[t-i] + eps
p     : AR order (number of lags). Example: p=8 uses x[t-1]..x[t-8].
fs    : Sampling frequency of the ECG (Hz = samples per second), taken from .hea.
dt    : Time step of the *regular* RR time grid (seconds). Default dt=1.0 means 1 sample per second for RR(t).
n_sig : Number of signal channels in the record (from .hea). Some headers may show 0 depending on packaging.
n_samples : Number of ECG samples in the record (from .hea). May be 0/unknown in some headers.
ann   : Annotation (event) in WFDB annotation file (.qrs/.qrsc/.atr).
n_ann : Number of parsed annotations.
beats / n_beats : Extracted beat events (heartbeat timestamps in samples) and their count.
MSE   : Mean Squared Error = mean((x - x_hat)^2). Units: seconds^2 for RR.
RMSE  : Root Mean Squared Error = sqrt(MSE). Units: seconds for RR.
MAE   : Mean Absolute Error = mean(|x - x_hat|). Units: seconds for RR.
MAPE  : Mean Absolute Percentage Error = mean(|x - x_hat|/|x|)*100%. Units: percent.
one-step-ahead forecast: Predict x[t+1] using data up to time t (walk-forward evaluation).
sliding window: Use the last W points equally (uniform weights) to fit AR coefficients at each time step.
one-sided kernel (causal): Use only the past with decaying weights for older points (Gaussian decay).

================================================================================
WHAT IS ACTUALLY MODELED / FORECASTED?
================================================================================
We do NOT forecast the raw ECG waveform.
We forecast the RR-interval series (tachogram), i.e., the heart rate variability proxy:
    - detect beat times from WFDB annotations (.qrs/.qrsc/.atr)
    - convert beat times to RR intervals in seconds
    - interpolate RR to a regular time grid (dt seconds)
Then TVAR(p) forecasts RR(t+dt) from the recent RR history.

================================================================================
PLOTS EXPLAINED (files in ./outputs)
================================================================================
1) <record>_rr_overview.png
   "RR(t) overview (regularized)"
   - X axis: time in hours; Y axis: RR interval (s)
   - Purpose: global view of nonstationarity (slow trends, regime changes).

2) <record>_rr_first30min.png
   "RR(t) first 30 minutes"
   - X axis: time in minutes; Y axis: RR interval (s)
   - Purpose: zoomed local view to sanity-check interpolation.

3) <record>_<model>_forecast_segment.png
   "One-step forecast segment"
   - Actual RR(t) and one-step forecast x_hat(t) on a short interval.
   - Purpose: qualitative check that the model follows dynamics.

4) <record>_<model>_abs_error.png
   "Absolute forecast error |x - x_hat|"
   - X axis: time in hours; Y axis: |error| (s)
   - Purpose: anomaly-detection proxy (spikes may indicate regime change).

5) <record>_<model>_coeffs.png
   "TVAR coefficients a_i(t)"
   - X axis: time in hours; Y axis: a1(t), a2(t), ...
   - Purpose: interpret time-varying dependencies.

6) <record>_<model>_split_observed_forecast.png
   "Observed (left) vs Forecast (right)"
   - Full ground truth faintly + observed part (left) + forecast part (right) + vertical split line.
   - Purpose: visual separation between input history and predicted region.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclasses.dataclass(frozen=True)
class WfdbHeader:
    record_name: str
    n_sig: int
    fs: float
    n_samples: Optional[int]
    signal_lines: List[str]


@dataclasses.dataclass
class WfdbAnnotation:
    sample: int
    ann_type: int
    aux: Optional[str] = None


@dataclasses.dataclass
class RecordResult:
    record: str
    model: str
    p: int
    window: int
    dt: float
    kernel_bandwidth: float
    ridge: float
    n_grid: int
    duration_hours: float
    metrics: Dict[str, float]
    outputs: List[Path]


# =============================================================================
# MAIN CLASS
# =============================================================================

class AfdbTvarRunner:
    """
    A reusable runner class to process one or many AFDB records.

    Typical usage from Python:
        runner = AfdbTvarRunner(dataset_dir="./dataset", outputs_dir="./outputs")
        res = runner.run_record("00735", model="sliding")
        results = runner.run_all_records(model="kernel")

    It keeps no global state except configuration (p, window, dt, ...).
    """

    # WFDB parsing constants
    WFDB_ANN_SKIP = 59
    WFDB_ANN_AUX = 63
    BEAT_TYPE_MIN = 1
    BEAT_TYPE_MAX = 49

    def __init__(
        self,
        dataset_dir: str | Path = "./dataset",
        outputs_dir: str | Path = "./outputs",
        verbose: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.outputs_dir = Path(outputs_dir).resolve()
        self.verbose = verbose

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Utility printing
    # -------------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # -------------------------------------------------------------------------
    # Record discovery
    # -------------------------------------------------------------------------

    def list_records(self) -> List[str]:
        hea_files = sorted(self.dataset_dir.glob("*.hea"))
        return [p.stem for p in hea_files]

    # -------------------------------------------------------------------------
    # WFDB header parsing
    # -------------------------------------------------------------------------

    def read_header(self, record: str) -> WfdbHeader:
        hea_path = self.dataset_dir / f"{record}.hea"
        if not hea_path.exists():
            raise FileNotFoundError(f"Header not found: {hea_path}")

        lines: List[str] = []
        with hea_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                lines.append(line)

        if not lines:
            raise ValueError(f"Header is empty or only comments: {hea_path}")

        rec_tokens = lines[0].split()
        if len(rec_tokens) < 2:
            raise ValueError(f"Invalid record line: {lines[0]}")

        record_name = rec_tokens[0].split("/")[0]
        n_sig = int(rec_tokens[1])

        fs = 250.0
        n_samples: Optional[int] = None

        if len(rec_tokens) >= 3:
            try:
                fs = float(rec_tokens[2])
            except ValueError:
                fs = 250.0

        if len(rec_tokens) >= 4:
            try:
                n_samples = int(rec_tokens[3])
            except ValueError:
                n_samples = None

        signal_lines = lines[1:1 + n_sig]
        return WfdbHeader(
            record_name=record_name,
            n_sig=n_sig,
            fs=fs,
            n_samples=n_samples,
            signal_lines=signal_lines,
        )

    # -------------------------------------------------------------------------
    # WFDB annotation parsing (minimal but robust)
    # -------------------------------------------------------------------------

    @staticmethod
    def _read_int32_le(buf: np.ndarray, idx: int) -> int:
        b0 = int(buf[idx + 0])
        b1 = int(buf[idx + 1])
        b2 = int(buf[idx + 2])
        b3 = int(buf[idx + 3])
        val = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        if val >= 2**31:
            val -= 2**32
        return val

    def read_annotations(self, record: str) -> Tuple[str, List[WfdbAnnotation]]:
        candidates = ["qrsc", "qrs", "atr"]
        ann_path: Optional[Path] = None
        used_ext = ""

        for ext in candidates:
            p = self.dataset_dir / f"{record}.{ext}"
            if p.exists():
                ann_path = p
                used_ext = ext
                break

        if ann_path is None:
            raise FileNotFoundError(
                f"No annotation file found for record={record}. "
                f"Tried: {', '.join(candidates)} in {self.dataset_dir}"
            )

        raw = np.fromfile(str(ann_path), dtype=np.uint8)
        if raw.size < 2:
            raise ValueError(f"Annotation file seems too small: {ann_path}")

        annotations: List[WfdbAnnotation] = []
        i = 0
        sample = 0

        while i + 1 < raw.size:
            a = int(raw[i])
            b = int(raw[i + 1])
            i += 2

            if a == 0 and b == 0:
                break

            ann_type = b >> 2
            time_inc = a + ((b & 0x03) << 8)

            if ann_type == self.WFDB_ANN_SKIP:
                if i + 4 > raw.size:
                    break
                skip = self._read_int32_le(raw, i)
                i += 4
                sample += skip
                continue

            sample += time_inc

            aux_str: Optional[str] = None
            if ann_type == self.WFDB_ANN_AUX:
                aux_len = time_inc
                if aux_len > 0 and i + aux_len <= raw.size:
                    aux_bytes = raw[i:i + aux_len].tobytes()
                    i += aux_len
                    aux_str = aux_bytes.decode("ascii", errors="ignore").strip("\x00").strip()

            annotations.append(WfdbAnnotation(sample=sample, ann_type=ann_type, aux=aux_str))

        return used_ext, annotations

    def extract_beat_samples(self, annotations: Sequence[WfdbAnnotation]) -> np.ndarray:
        beats: List[int] = []
        for a in annotations:
            if self.BEAT_TYPE_MIN <= a.ann_type <= self.BEAT_TYPE_MAX:
                beats.append(a.sample)

        beats_arr = np.array(beats, dtype=np.int64)
        beats_arr = np.unique(beats_arr)
        beats_arr.sort()
        return beats_arr

    # -------------------------------------------------------------------------
    # RR series + regularization
    # -------------------------------------------------------------------------

    @staticmethod
    def build_rr_series(
        beat_samples: np.ndarray,
        fs: float,
        rr_min_s: float = 0.3,
        rr_max_s: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if beat_samples.size < 3:
            raise ValueError("Not enough beats to build RR series.")

        beat_times = beat_samples.astype(np.float64) / float(fs)
        rr = np.diff(beat_times)
        times_rr = beat_times[1:]

        rr_clean = rr.copy()
        mask_ok = (rr_clean >= rr_min_s) & (rr_clean <= rr_max_s)
        rr_clean[~mask_ok] = np.nan
        return times_rr, rr_clean

    @staticmethod
    def interpolate_nans_1d(x: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError("interpolate_nans_1d expects a 1D array.")

        y = x.astype(np.float64).copy()
        idx = np.arange(y.size)

        valid = np.isfinite(y)
        if valid.sum() < 2:
            return np.nan_to_num(y, nan=0.0)

        first = idx[valid][0]
        last = idx[valid][-1]
        y[:first] = y[first]
        y[last + 1:] = y[last]

        valid = np.isfinite(y)
        y[~valid] = np.interp(idx[~valid], idx[valid], y[valid])
        return y

    @classmethod
    def regularize_to_grid(
        cls,
        t_irregular: np.ndarray,
        y_irregular: np.ndarray,
        dt: float,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if t_irregular.size != y_irregular.size:
            raise ValueError("t_irregular and y_irregular must have same length.")

        y_filled = cls.interpolate_nans_1d(y_irregular)

        if t_start is None:
            t_start = float(t_irregular[0])
        if t_end is None:
            t_end = float(t_irregular[-1])

        n_grid = int(math.floor((t_end - t_start) / dt)) + 1
        t_grid = t_start + dt * np.arange(n_grid, dtype=np.float64)
        y_grid = np.interp(t_grid, t_irregular, y_filled)
        return t_grid, y_grid

    # -------------------------------------------------------------------------
    # TVAR (sliding / kernel) estimation
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_ar_design(x: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
        L = x.size
        if L <= p:
            raise ValueError("Need L > p to build design.")

        y = x[p:].copy()
        X = np.ones((L - p, p + 1), dtype=np.float64)
        for i in range(1, p + 1):
            X[:, i] = x[p - i:L - i]
        return X, y

    @staticmethod
    def _ridge_solve(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
        XtX = X.T @ X
        Xty = X.T @ y
        reg = ridge * np.eye(XtX.shape[0], dtype=np.float64)
        return np.linalg.solve(XtX + reg, Xty)

    @classmethod
    def _weighted_ridge_solve(cls, X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float) -> np.ndarray:
        w = np.asarray(w, dtype=np.float64)
        w = np.clip(w, 0.0, np.inf)
        sw = np.sqrt(w + 1e-12)
        Xw = X * sw[:, None]
        yw = y * sw
        return cls._ridge_solve(Xw, yw, ridge=ridge)

    @classmethod
    def tvar_walk_forward(
        cls,
        x: np.ndarray,
        p: int,
        window: int,
        model: str,
        kernel_bandwidth: float,
        ridge: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = x.size
        preds = np.full(N, np.nan, dtype=np.float64)
        coefs = np.full((N, p), np.nan, dtype=np.float64)

        for k in range(N - 1):
            end = k + 1
            start = max(0, end - window)
            x_win = x[start:end]

            if x_win.size <= p:
                continue

            X, y = cls._build_ar_design(x_win, p=p)

            if model == "sliding":
                beta = cls._ridge_solve(X, y, ridge=ridge)
            elif model == "kernel":
                row_idx = np.arange(y.size, dtype=np.float64)
                y_global = start + (p + row_idx)
                latest = float(end - 1)
                dist = latest - y_global
                h = max(1e-6, float(kernel_bandwidth))
                w = np.exp(-0.5 * (dist / h) ** 2)
                beta = cls._weighted_ridge_solve(X, y, w=w, ridge=ridge)
            else:
                raise ValueError(f"Unknown model={model}. Use 'sliding' or 'kernel'.")

            if k - p + 1 < 0:
                continue

            lags = np.array([x[k - i] for i in range(0, p)], dtype=np.float64)
            preds[k + 1] = beta[0] + float(np.dot(beta[1:], lags))
            coefs[k + 1, :] = beta[1:]

        return preds, coefs

    @classmethod
    def tvar_walk_forward_with_betas(
            cls,
            x: np.ndarray,
            p: int,
            window: int,
            model: str,
            kernel_bandwidth: float,
            ridge: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Same as tvar_walk_forward, but also returns beta_full[t] = [c, a1..ap]
        aligned so that beta_full[t] predicts x[t] from x[t-1..t-p].

        Returns:
            preds: length N, preds[t] predicts x[t] (NaN where undefined)
            coefs: (N, p), coefs[t] = [a1(t)..ap(t)] (NaN where undefined)
            beta_full: (N, p+1), beta_full[t] = [c(t), a1(t)..ap(t)]
        """
        N = x.size
        preds = np.full(N, np.nan, dtype=np.float64)
        coefs = np.full((N, p), np.nan, dtype=np.float64)
        beta_full = np.full((N, p + 1), np.nan, dtype=np.float64)

        for k in range(N - 1):
            end = k + 1
            start = max(0, end - window)
            x_win = x[start:end]

            if x_win.size <= p:
                continue

            X, y = cls._build_ar_design(x_win, p=p)

            if model == "sliding":
                beta = cls._ridge_solve(X, y, ridge=ridge)
            elif model == "kernel":
                row_idx = np.arange(y.size, dtype=np.float64)
                y_global = start + (p + row_idx)
                latest = float(end - 1)
                dist = latest - y_global
                h = max(1e-6, float(kernel_bandwidth))
                w = np.exp(-0.5 * (dist / h) ** 2)
                beta = cls._weighted_ridge_solve(X, y, w=w, ridge=ridge)
            else:
                raise ValueError(f"Unknown model={model}. Use 'sliding' or 'kernel'.")

            if k - p + 1 < 0:
                continue

            lags = np.array([x[k - i] for i in range(0, p)], dtype=np.float64)  # x[k], x[k-1], ...
            preds[k + 1] = beta[0] + float(np.dot(beta[1:], lags))
            coefs[k + 1, :] = beta[1:]
            beta_full[k + 1, :] = beta  # beta_full[t] predicts x[t]

        return preds, coefs, beta_full

    @staticmethod
    def _open_uniform_knots(n_basis: int, degree: int) -> np.ndarray:
        """
        Create an open uniform knot vector on [0, 1] for B-splines.
        Knot count = n_basis + degree + 1.
        """
        if n_basis <= degree + 1:
            raise ValueError("n_basis must be > degree + 1.")

        n_knots = n_basis + degree + 1
        n_internal = n_knots - 2 * (degree + 1)

        if n_internal > 0:
            internal = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
            knots = np.concatenate(
                [np.zeros(degree + 1), internal, np.ones(degree + 1)]
            )
        else:
            knots = np.concatenate([np.zeros(degree + 1), np.ones(degree + 1)])

        return knots.astype(np.float64)

    @classmethod
    def bspline_basis_matrix(cls, t: np.ndarray, n_basis: int, degree: int) -> np.ndarray:
        """
        Evaluate B-spline basis functions at points t in [0, 1].

        Returns:
            B: shape (len(t), n_basis)
        """
        t = np.asarray(t, dtype=np.float64)
        t = np.clip(t, 0.0, 1.0)

        knots = cls._open_uniform_knots(n_basis=n_basis, degree=degree)
        n = t.size

        # N_{i,0}(t)
        N0 = np.zeros((n_basis, n), dtype=np.float64)
        for i in range(n_basis):
            left = knots[i]
            right = knots[i + 1]
            N0[i, :] = ((t >= left) & (t < right)).astype(np.float64)

        # Special case: t == 1 goes to the last basis function
        N0[-1, t == 1.0] = 1.0

        N = N0
        # Cox–de Boor recursion
        for k in range(1, degree + 1):
            Nk = np.zeros_like(N)
            for i in range(n_basis):
                denom1 = knots[i + k] - knots[i]
                denom2 = knots[i + k + 1] - knots[i + 1]

                term1 = 0.0
                if denom1 > 0:
                    term1 = ((t - knots[i]) / denom1) * N[i, :]

                term2 = 0.0
                if (i + 1) < n_basis and denom2 > 0:
                    term2 = ((knots[i + k + 1] - t) / denom2) * N[i + 1, :]

                Nk[i, :] = term1 + term2

            N = Nk

        return N.T  # (n, n_basis)

    @staticmethod
    def _second_difference_penalty(n_basis: int) -> np.ndarray:
        """
        Build P = D2^T D2 where D2 encodes 2nd differences over spline coefficients.
        This is a simple smoothness penalty (analog of GAM spline penalty).
        """
        if n_basis < 3:
            return np.zeros((n_basis, n_basis), dtype=np.float64)

        D2 = np.zeros((n_basis - 2, n_basis), dtype=np.float64)
        for r in range(n_basis - 2):
            D2[r, r] = 1.0
            D2[r, r + 1] = -2.0
            D2[r, r + 2] = 1.0

        return D2.T @ D2  # (n_basis, n_basis)

    @classmethod
    def tvar_gam_spline_fit_predict(
            cls,
            x: np.ndarray,
            p: int,
            n_splines: int,
            degree: int,
            smooth_lambda: float,
            ridge: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GAM-like spline TVAR:

            x[t] = c + sum_{i=1..p} a_i(t) * x[t-i] + eps
            a_i(t) = sum_{k=1..K} theta_{i,k} * B_k(t)

        Fit is done globally (offline) using all rows t=p..N-1 with
        ridge + smoothness penalty on theta (2nd differences across k).

        Returns:
            preds: length N, preds[t] predicts x[t] using x[t-1..t-p] (NaN for t<p)
            coefs: shape (N, p), coefs[t, i-1] = a_i(t) (NaN for t<p)
        """
        x = np.asarray(x, dtype=np.float64)
        N = x.size
        if N <= p + 5:
            raise ValueError("Series too short for GAM TVAR.")

        # Time normalized to [0, 1]
        t_norm = np.linspace(0.0, 1.0, N, dtype=np.float64)
        B = cls.bspline_basis_matrix(t=t_norm, n_basis=n_splines, degree=degree)  # (N, K)

        # Build design matrix for t = p..N-1
        M = N - p
        y = x[p:]  # target x[t]
        B_used = B[p:, :]  # basis at target time t

        K = n_splines
        X = np.ones((M, 1 + p * K), dtype=np.float64)  # intercept + p blocks

        for i in range(1, p + 1):
            lag = x[p - i:N - i]  # x[t-i], aligned with y
            X[:, 1 + (i - 1) * K: 1 + i * K] = B_used * lag[:, None]

        # Penalty matrix: intercept unpenalized, spline params penalized per lag-block
        P_block = cls._second_difference_penalty(K)
        P = np.zeros((1 + p * K, 1 + p * K), dtype=np.float64)
        for i in range(p):
            s = 1 + i * K
            P[s:s + K, s:s + K] = P_block

        XtX = X.T @ X
        Xty = X.T @ y

        A = XtX + ridge * np.eye(XtX.shape[0], dtype=np.float64) + smooth_lambda * P
        beta = np.linalg.solve(A, Xty)

        c = float(beta[0])
        theta = beta[1:].reshape(p, K)  # (p, K)

        # Compute a_i(t) for all times
        a_tp = B @ theta.T  # (N, p)
        coefs = a_tp.copy()
        coefs[:p, :] = np.nan

        # One-step prediction: preds[t] predicts x[t]
        preds = np.full(N, np.nan, dtype=np.float64)
        for t in range(p, N):
            lags = np.array([x[t - i] for i in range(1, p + 1)], dtype=np.float64)  # x[t-1]..x[t-p]
            preds[t] = c + float(np.dot(coefs[t, :], lags))

        return preds, coefs

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_metrics(x: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        mask = np.isfinite(pred)
        if mask.sum() == 0:
            return {
                "n_eval": 0,
                "mse": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "mape": float("nan"),
                "mae_pct_mean": float("nan"),
                "rmse_pct_mean": float("nan"),
                "abs_q50": float("nan"),
                "abs_q90": float("nan"),
                "abs_q95": float("nan"),
                "abs_q99": float("nan"),
                "abs_max": float("nan"),
            }

        x_eval = x[mask]
        pred_eval = pred[mask]
        err = x_eval - pred_eval
        abs_err = np.abs(err)

        mse = float(np.mean(err ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(abs_err))

        denom = np.maximum(np.abs(x_eval), 1e-12)
        mape = float(np.mean(abs_err / denom) * 100.0)

        mean_x = float(np.mean(x_eval))
        mae_pct_mean = float(mae / max(mean_x, 1e-12) * 100.0)
        rmse_pct_mean = float(rmse / max(mean_x, 1e-12) * 100.0)

        return {
            "n_eval": int(mask.sum()),
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "mae_pct_mean": mae_pct_mean,
            "rmse_pct_mean": rmse_pct_mean,
            "abs_q50": float(np.quantile(abs_err, 0.50)),
            "abs_q90": float(np.quantile(abs_err, 0.90)),
            "abs_q95": float(np.quantile(abs_err, 0.95)),
            "abs_q99": float(np.quantile(abs_err, 0.99)),
            "abs_max": float(np.max(abs_err)),
        }

    @classmethod
    def multistep_free_run_metrics(
            cls,
            x: np.ndarray,
            beta_full: np.ndarray,
            p: int,
            horizons: List[int],
            stride: int = 60,
    ) -> Dict[int, Dict[str, float]]:
        """
        Free-run multi-step evaluation:
        At anchor k, take beta = beta_full[k+1] (estimated using past up to k),
        then simulate predictions forward for max(horizons) steps, feeding predictions
        back into the lag state (no teacher forcing).

        To keep it fast, anchors are evaluated every `stride` samples.

        Returns:
            dict[h] -> metrics dict (same keys as compute_metrics)
        """
        horizons = sorted(set(int(h) for h in horizons if int(h) > 0))
        if not horizons:
            return {}

        N = x.size
        max_h = max(horizons)

        preds_by_h = {h: np.full(N, np.nan, dtype=np.float64) for h in horizons}

        # anchor k means last observed index is k, we predict k+1..k+max_h
        for k in range(p, N - max_h - 1, stride):
            beta = beta_full[k + 1, :]
            if not np.all(np.isfinite(beta)):
                continue

            c = float(beta[0])
            a = beta[1:]  # (p,)

            # state = [x[k], x[k-1], ..., x[k-p+1]]
            state = [float(x[k - i]) for i in range(0, p)]

            for step in range(1, max_h + 1):
                x_next = c + float(np.dot(a, np.asarray(state, dtype=np.float64)))

                if step in preds_by_h:
                    preds_by_h[step][k + step] = x_next

                # shift state: new becomes most recent
                state = [x_next] + state[:-1]

        return {h: cls.compute_metrics(x, preds_by_h[h]) for h in horizons}

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def _save_plot(self, path: Path, title: str) -> None:
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        self._log(f"[Saved] {path}")

    def plot_series(
        self,
        t: np.ndarray,
        x: np.ndarray,
        out_path: Path,
        title: str,
        max_minutes: Optional[float] = None,
    ) -> None:
        plt.figure(figsize=(12, 4))
        if max_minutes is not None:
            t0 = float(t[0])
            tmax = t0 + 60.0 * max_minutes
            mask = t <= tmax
            plt.plot(t[mask] / 60.0, x[mask])
            plt.xlabel("Time (minutes)")
        else:
            plt.plot(t / 3600.0, x)
            plt.xlabel("Time (hours)")
        plt.ylabel("RR interval (s)")
        plt.grid(True, alpha=0.3)
        self._save_plot(out_path, title)
        plt.close()

    def plot_forecast_segment(
        self,
        t: np.ndarray,
        x: np.ndarray,
        pred: np.ndarray,
        out_path: Path,
        title: str,
        start_min: float,
        duration_min: float,
    ) -> None:
        plt.figure(figsize=(12, 4))
        t0 = float(t[0])
        a = t0 + 60.0 * start_min
        b = a + 60.0 * duration_min
        mask = (t >= a) & (t <= b)
        plt.plot(t[mask] / 60.0, x[mask], label="Actual")
        plt.plot(t[mask] / 60.0, pred[mask], label="One-step forecast", alpha=0.8)
        plt.xlabel("Time (minutes)")
        plt.ylabel("RR interval (s)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        self._save_plot(out_path, title)
        plt.close()

    def plot_error(
        self,
        t: np.ndarray,
        x: np.ndarray,
        pred: np.ndarray,
        out_path: Path,
        title: str,
    ) -> None:
        err = x - pred
        plt.figure(figsize=(12, 4))
        plt.plot(t / 3600.0, np.abs(err))
        plt.xlabel("Time (hours)")
        plt.ylabel("|Forecast error| (s)")
        plt.grid(True, alpha=0.3)
        self._save_plot(out_path, title)
        plt.close()

    def plot_coeffs(
        self,
        t: np.ndarray,
        coefs: np.ndarray,
        out_path: Path,
        title: str,
        max_lags_to_show: int = 5,
    ) -> None:
        plt.figure(figsize=(12, 5))
        p = coefs.shape[1]
        m = min(p, max_lags_to_show)
        for i in range(m):
            plt.plot(t / 3600.0, coefs[:, i], label=f"a{i+1}(t)")
        plt.xlabel("Time (hours)")
        plt.ylabel("AR coefficients")
        plt.grid(True, alpha=0.3)
        plt.legend()
        self._save_plot(out_path, title)
        plt.close()

    def plot_observed_vs_forecast_split(
        self,
        t: np.ndarray,
        x: np.ndarray,
        pred: np.ndarray,
        split_idx: int,
        out_path: Path,
        title: str,
    ) -> None:
        split_idx = int(np.clip(split_idx, 1, len(t) - 2))

        plt.figure(figsize=(12, 4))
        plt.plot(t / 3600.0, x, alpha=0.25, label="Ground truth (full)")
        plt.plot(t[:split_idx] / 3600.0, x[:split_idx], label="Observed (input)")
        plt.plot(t[split_idx:] / 3600.0, pred[split_idx:], label="Forecast (model)", alpha=0.9)

        split_x = t[split_idx] / 3600.0
        plt.axvline(split_x, linestyle="--", linewidth=1.5, alpha=0.9)
        plt.text(
            split_x,
            float(np.nanmin(x)),
            "  split",
            rotation=90,
            va="bottom",
            ha="left",
        )

        plt.xlabel("Time (hours)")
        plt.ylabel("RR interval (s)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        self._save_plot(out_path, title)
        plt.close()

    # -------------------------------------------------------------------------
    # High-level execution: single record / all records
    # -------------------------------------------------------------------------

    def run_record(
        self,
        record: str,
        model: str = "sliding",
        p: int = 8,
        window: int = 600,
        dt: float = 1.0,
        kernel_bandwidth: float = 120.0,
        ridge: float = 1e-3,
        plot_start_min: float = 60.0,
        plot_duration_min: float = 10.0,
        split_frac: float = 0.7,
        make_plots: bool = True,
        gam_n_splines: int = 25,
        gam_degree: int = 3,
        gam_smooth: float = 10.0,
    ) -> RecordResult:
        if model not in ("sliding", "kernel", "gam"):
            raise ValueError("model must be 'sliding' or 'kernel'.")

        self._log("=" * 80)
        self._log("AFDB TVAR run_record()")
        self._log(f"Dataset dir : {self.dataset_dir}")
        self._log(f"Record      : {record}")
        self._log(f"Model       : {model}")
        self._log(f"AR order p  : {p}")
        self._log(f"Window      : {window} samples")
        self._log(f"dt          : {dt} seconds (regular RR grid)")
        if model == "kernel":
            self._log(f"Kernel bw   : {kernel_bandwidth} samples")
        if model == "gam":
            self._log(f"GAM splines  : K={gam_n_splines}, degree={gam_degree}, lambda={gam_smooth}")
        self._log(f"Ridge       : {ridge}")
        self._log("=" * 80)

        header = self.read_header(record)
        self._log("[Header parsed]")
        self._log(f"  record_name : {header.record_name}")
        self._log(f"  n_sig       : {header.n_sig}")
        self._log(f"  fs          : {header.fs} Hz")
        self._log(f"  n_samples   : {header.n_samples}")

        used_ext, ann = self.read_annotations(record)
        self._log("[Annotations parsed]")
        self._log(f"  file used   : {record}.{used_ext}")
        self._log(f"  n_ann       : {len(ann)}")

        beat_samples = self.extract_beat_samples(ann)
        if beat_samples.size == 0:
            raise RuntimeError("No beat samples extracted. Check annotation parsing.")

        self._log("[Beats extracted]")
        self._log(f"  n_beats     : {beat_samples.size}")
        self._log(f"  first beat  : sample={beat_samples[0]}  time={beat_samples[0]/header.fs:.3f}s")
        self._log(f"  last beat   : sample={beat_samples[-1]} time={beat_samples[-1]/header.fs:.3f}s")
        duration_h = (beat_samples[-1] / header.fs) / 3600.0
        self._log(f"  approx duration: {duration_h:.2f} hours")

        times_rr, rr_irregular = self.build_rr_series(beat_samples, fs=header.fs)
        self._log("[RR series built (irregular)]")
        self._log(f"  n_rr        : {rr_irregular.size}")
        self._log(f"  rr nan %    : {100.0 * np.mean(~np.isfinite(rr_irregular)):.2f}%")
        self._log(f"  rr mean (raw, nan-safe): {np.nanmean(rr_irregular):.4f} s")

        t_grid, rr_grid = self.regularize_to_grid(times_rr, rr_irregular, dt=dt)
        self._log("[RR series regularized]")
        self._log(f"  n_grid      : {rr_grid.size}")
        self._log(f"  total time  : {(t_grid[-1] - t_grid[0]) / 3600.0:.2f} hours")
        self._log(f"  rr mean     : {float(np.mean(rr_grid)):.4f} s")
        self._log(f"  rr std      : {float(np.std(rr_grid)):.4f} s")

        if model in ("sliding", "kernel"):
            preds, coefs, beta_full = self.tvar_walk_forward_with_betas(
                x=rr_grid,
                p=p,
                window=window,
                model=model,
                kernel_bandwidth=kernel_bandwidth,
                ridge=ridge,
            )
        elif model == "gam":
            preds, coefs = self.tvar_gam_spline_fit_predict(
                x=rr_grid,
                p=p,
                n_splines=gam_n_splines,
                degree=gam_degree,
                smooth_lambda=gam_smooth,
                ridge=ridge,
            )
            # beta_full[t] predicts x[t] using x[t-1..t-p] (intercept is constant c inside GAM fit)
            # Here we approximate intercept by solving: pred[t] = c + dot(a(t), lags) -> c = pred[t] - dot(...)
            # Safer: treat intercept as 0 if you don't want leakage. But better: just set c=0 and keep a(t).
            # Minimal/robust for evaluation: set c=0.
            beta_full = np.full((rr_grid.size, p + 1), np.nan, dtype=np.float64)
            beta_full[:, 0] = 0.0
            beta_full[:, 1:] = coefs
        else:
            raise ValueError("Unknown model.")

        metrics = self.compute_metrics(rr_grid, preds)
        self._log("[Forecast metrics: one-step-ahead]")
        self._log(f"  n_eval      : {metrics['n_eval']}")
        self._log(f"  MSE         : {metrics['mse']:.6e} (s^2)")
        self._log(f"  RMSE        : {metrics['rmse']:.6e} (s)")
        self._log(f"  MAE         : {metrics['mae']:.6e} (s)")
        self._log(f"  MAE/mean    : {metrics['mae_pct_mean']:.4f}%")
        self._log(f"  RMSE/mean   : {metrics['rmse_pct_mean']:.4f}%")
        self._log(f"  MAPE        : {metrics['mape']:.4f}%")
        self._log("[Tail metrics for |error|]")
        self._log(f"  |err| q95   : {metrics['abs_q95']:.6e} (s)")
        self._log(f"  |err| q99   : {metrics['abs_q99']:.6e} (s)")
        self._log(f"  |err| max   : {metrics['abs_max']:.6e} (s)")

        # Free-run multi-step (these metrics usually differentiate models much better)
        horizons = [1, 10, 30]  # in samples; with dt=1s => 1s, 10s, 30s
        ms = self.multistep_free_run_metrics(
            x=rr_grid,
            beta_full=beta_full,
            p=p,
            horizons=horizons,
            stride=60,  # evaluate every 60 seconds to keep it fast
        )
        self._log("[Free-run multi-step metrics (evaluated every 60 samples)]")
        for h in horizons:
            m = ms.get(h, {})
            if not m:
                continue
            self._log(
                f"  H={h:2d} | MAE={m['mae']:.6e}s ({m['mae_pct_mean']:.4f}%) | "
                f"RMSE={m['rmse']:.6e}s ({m['rmse_pct_mean']:.4f}%) | "
                f"|err|q95={m['abs_q95']:.6e}s | max={m['abs_max']:.6e}s"
            )

        outputs: List[Path] = []

        if make_plots:
            # 1) RR overview
            p1 = self.outputs_dir / f"{record}_rr_overview.png"
            self.plot_series(
                t=t_grid,
                x=rr_grid,
                out_path=p1,
                title=f"RR(t) overview (regularized), record={record}, dt={dt}s",
                max_minutes=None,
            )
            outputs.append(p1)

            # 2) First 30 minutes
            p2 = self.outputs_dir / f"{record}_rr_first30min.png"
            self.plot_series(
                t=t_grid,
                x=rr_grid,
                out_path=p2,
                title=f"RR(t) first 30 minutes, record={record}, dt={dt}s",
                max_minutes=30.0,
            )
            outputs.append(p2)

            # 3) Forecast segment
            p3 = self.outputs_dir / f"{record}_{model}_forecast_segment.png"
            self.plot_forecast_segment(
                t=t_grid,
                x=rr_grid,
                pred=preds,
                out_path=p3,
                title=f"One-step forecast segment, record={record}, model={model}, p={p}, window={window}",
                start_min=plot_start_min,
                duration_min=plot_duration_min,
            )
            outputs.append(p3)

            # 4) Absolute error
            p4 = self.outputs_dir / f"{record}_{model}_abs_error.png"
            self.plot_error(
                t=t_grid,
                x=rr_grid,
                pred=preds,
                out_path=p4,
                title=f"Absolute forecast error |x - x_hat|, record={record}, model={model}",
            )
            outputs.append(p4)

            # 5) Coefficients
            p5 = self.outputs_dir / f"{record}_{model}_coeffs.png"
            self.plot_coeffs(
                t=t_grid,
                coefs=coefs,
                out_path=p5,
                title=f"TVAR coefficients a_i(t), record={record}, model={model}, p={p}",
                max_lags_to_show=min(5, p),
            )
            outputs.append(p5)

            # 6) Split plot
            split_idx = int(np.clip(split_frac, 0.0, 1.0) * len(t_grid))
            p6 = self.outputs_dir / f"{record}_{model}_split_observed_forecast.png"
            self.plot_observed_vs_forecast_split(
                t=t_grid,
                x=rr_grid,
                pred=preds,
                split_idx=split_idx,
                out_path=p6,
                title=f"Observed (left) vs Forecast (right), record={record}, model={model}, split={split_frac:.2f}",
            )
            outputs.append(p6)

        return RecordResult(
            record=record,
            model=model,
            p=p,
            window=window,
            dt=dt,
            kernel_bandwidth=kernel_bandwidth,
            ridge=ridge,
            n_grid=int(rr_grid.size),
            duration_hours=float((t_grid[-1] - t_grid[0]) / 3600.0),
            metrics=metrics,
            outputs=outputs,
        )

    def run_all_records(
        self,
        model: str = "sliding",
        p: int = 8,
        window: int = 600,
        dt: float = 1.0,
        kernel_bandwidth: float = 120.0,
        ridge: float = 1e-3,
        plot_start_min: float = 60.0,
        plot_duration_min: float = 10.0,
        split_frac: float = 0.7,
        make_plots: bool = False,
        stop_on_error: bool = False,
    ) -> List[RecordResult]:
        """
        Run through all records in dataset_dir.

        Defaults:
          make_plots=False (to avoid generating hundreds of images).
          You can enable it if you want per-record visuals.

        Returns list of RecordResult.
        """
        records = self.list_records()
        if not records:
            raise FileNotFoundError(f"No .hea files found in: {self.dataset_dir}")

        results: List[RecordResult] = []
        self._log(f"[run_all_records] Found {len(records)} records.")

        for idx, rec in enumerate(records, start=1):
            self._log("-" * 80)
            self._log(f"[{idx}/{len(records)}] Processing record={rec}")
            try:
                res = self.run_record(
                    record=rec,
                    model=model,
                    p=p,
                    window=window,
                    dt=dt,
                    kernel_bandwidth=kernel_bandwidth,
                    ridge=ridge,
                    plot_start_min=plot_start_min,
                    plot_duration_min=plot_duration_min,
                    split_frac=split_frac,
                    make_plots=make_plots,
                )
                results.append(res)
            except Exception as e:
                self._log(f"[ERROR] record={rec}: {e}")
                if stop_on_error:
                    raise

        return results


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AFDB TVAR runner class: process a single record or all records."
    )
    parser.add_argument("--dataset", type=str, default="./dataset", help="Path to AFDB folder.")
    parser.add_argument("--outputs", type=str, default="./outputs", help="Where to save plots.")
    parser.add_argument("--verbose", action="store_true", help="Verbose console output.")
    parser.add_argument(
        "--model",
        type=str,
        default="sliding",
        choices=["sliding", "kernel", "gam"],
        help="TVAR variant: sliding window, one-sided kernel, or GAM-like spline TVAR.",
    )
    parser.add_argument("--p", type=int, default=8, help="AR order p.")
    parser.add_argument("--window", type=int, default=600, help="Window length in samples (on regular grid).")
    parser.add_argument("--dt", type=float, default=1.0, help="Regular grid step in seconds for RR(t).")
    parser.add_argument(
        "--kernel_bandwidth",
        type=float,
        default=120.0,
        help="Kernel bandwidth (in samples, not seconds). Used only for model=kernel.",
    )
    parser.add_argument("--ridge", type=float, default=1e-3, help="Ridge regularization strength.")
    parser.add_argument("--plot_start_min", type=float, default=60.0, help="Start minute for forecast segment plot.")
    parser.add_argument("--plot_duration_min", type=float, default=10.0, help="Duration minutes for segment plot.")
    parser.add_argument("--split_frac", type=float, default=0.7, help="Split location (0..1) for split plot.")
    parser.add_argument("--make_plots", action="store_true", help="Generate plots (single or all records).")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--record", type=str, default="", help="Record name (e.g., 04043).")
    group.add_argument("--all_records", action="store_true", help="Process all records in dataset.")

    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="If set, stop immediately when a record fails in --all_records mode.",
    )

    parser.add_argument(
        "--gam_n_splines",
        type=int,
        default=25,
        help="GAM mode only: number of B-spline basis functions K.",
    )
    parser.add_argument(
        "--gam_degree",
        type=int,
        default=3,
        help="GAM mode only: B-spline degree (3=cubic).",
    )
    parser.add_argument(
        "--gam_smooth",
        type=float,
        default=10.0,
        help="GAM mode only: smoothness penalty lambda (higher -> smoother a_i(t)).",
    )

    args = parser.parse_args()

    runner = AfdbTvarRunner(
        dataset_dir=args.dataset,
        outputs_dir=args.outputs,
        verbose=bool(args.verbose),
    )

    if args.all_records:
        results = runner.run_all_records(
            model=args.model,
            p=args.p,
            window=args.window,
            dt=args.dt,
            kernel_bandwidth=args.kernel_bandwidth,
            ridge=args.ridge,
            plot_start_min=args.plot_start_min,
            plot_duration_min=args.plot_duration_min,
            split_frac=args.split_frac,
            make_plots=bool(args.make_plots),
            stop_on_error=bool(args.stop_on_error)
        )

        # Print a compact per-record summary
        print("=" * 80)
        print("SUMMARY (per record):")
        for r in results:
            m = r.metrics
            print(
                f"{r.record} | n={r.n_grid:6d} | "
                f"MAE={m['mae']:.4f}s ({m['mae_pct_mean']:.2f}%) | "
                f"RMSE={m['rmse']:.4f}s ({m['rmse_pct_mean']:.2f}%) | "
                f"MAPE={m['mape']:.2f}%"
            )
        print("=" * 80)

    else:
        records = runner.list_records()
        if not records:
            raise FileNotFoundError(f"No .hea files found in: {runner.dataset_dir}")

        record = args.record.strip() if args.record.strip() else records[0]
        runner.run_record(
            record=record,
            model=args.model,
            p=args.p,
            window=args.window,
            dt=args.dt,
            kernel_bandwidth=args.kernel_bandwidth,
            ridge=args.ridge,
            plot_start_min=args.plot_start_min,
            plot_duration_min=args.plot_duration_min,
            split_frac=args.split_frac,
            make_plots=True if args.make_plots else False,
            gam_n_splines=args.gam_n_splines,
            gam_degree=args.gam_degree,
            gam_smooth=args.gam_smooth,
        )


if __name__ == "__main__":
    main()