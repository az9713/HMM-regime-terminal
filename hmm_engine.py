"""
hmm_engine.py — HMM fitting, BIC model selection, regime analysis.

Core math module: BIC selection over 2-8 states with random restarts,
Viterbi decoding, forward-backward posteriors, Shannon entropy,
regime labeling and statistics, transition matrix analysis.
"""

import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp


class RegimeDetector:
    """Fit an optimally-selected HMM and extract regime information."""

    def __init__(self, config: dict):
        hmm_cfg = config.get("hmm", {})
        self.min_states = hmm_cfg.get("min_states", 2)
        self.max_states = hmm_cfg.get("max_states", 8)
        self.n_restarts = hmm_cfg.get("n_restarts", 20)
        self.n_iter = hmm_cfg.get("n_iter", 200)
        self.tol = hmm_cfg.get("tol", 1e-4)
        self.model_type = hmm_cfg.get("model_type", "gaussian")
        self.covariance_type = hmm_cfg.get("covariance_type", "full")
        self.gmm_n_mix = hmm_cfg.get("gmm_n_mix", 2)

        self.model = None
        self.n_states = None
        self.bic_scores = {}
        self.aic_scores = {}
        self.labels = {}
        self._regime_stats = None

    def _count_params(self, n: int, d: int) -> int:
        """
        Count free parameters for GaussianHMM:
          k = (n-1)             # initial state probs
            + n*(n-1)           # transition matrix rows (each sums to 1)
            + n*d               # means
            + n*d*(d+1)/2       # covariance (full)
        """
        k = (n - 1) + n * (n - 1) + n * d
        if self.covariance_type == "full":
            k += n * d * (d + 1) // 2
        elif self.covariance_type == "diag":
            k += n * d
        elif self.covariance_type == "spherical":
            k += n
        elif self.covariance_type == "tied":
            k += d * (d + 1) // 2
        return k

    def fit_and_select(self, X_train: np.ndarray) -> dict:
        """
        Test n_states from min to max, with n_restarts random seeds each.
        Select model with lowest BIC.
        Returns dict of {n_states: bic_score}.
        """
        T, d = X_train.shape
        best_bic = np.inf
        best_model = None

        for n in range(self.min_states, self.max_states + 1):
            best_ll_n = -np.inf
            best_model_n = None

            for seed in range(self.n_restarts):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = GaussianHMM(
                            n_components=n,
                            covariance_type=self.covariance_type,
                            n_iter=self.n_iter,
                            tol=self.tol,
                            random_state=seed,
                        )
                        model.fit(X_train)
                        ll = model.score(X_train)

                    if ll > best_ll_n:
                        best_ll_n = ll
                        best_model_n = model
                except Exception:
                    continue

            if best_model_n is None:
                continue

            k = self._count_params(n, d)
            bic = -2 * best_ll_n + k * np.log(T)
            aic = -2 * best_ll_n + 2 * k
            self.bic_scores[n] = bic
            self.aic_scores[n] = aic

            if bic < best_bic:
                best_bic = bic
                best_model = best_model_n

        if best_model is None:
            raise RuntimeError("All HMM fits failed")

        self.model = best_model
        self.n_states = best_model.n_components
        return self.bic_scores

    def decode(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Viterbi decoding and forward-backward algorithm.
        Returns (states, posteriors) where posteriors is (T, n_states).
        """
        states = self.model.predict(X)
        posteriors = self.model.predict_proba(X)
        return states, posteriors

    def label_regimes(self, X: np.ndarray) -> dict[int, str]:
        """
        Sort states by mean log_return (column 0) and assign labels.
        Labels range from 'crash' (lowest return) to 'bull_run' (highest).
        """
        n = self.n_states
        # Compute mean of first feature (log_return) per state
        means = self.model.means_[:, 0]
        sorted_idx = np.argsort(means)

        if n == 2:
            label_names = ["bear", "bull"]
        elif n == 3:
            label_names = ["bear", "neutral", "bull"]
        elif n == 4:
            label_names = ["crash", "bear", "bull", "bull_run"]
        elif n == 5:
            label_names = ["crash", "bear", "neutral", "bull", "bull_run"]
        else:
            label_names = [f"regime_{i}" for i in range(n)]
            if n >= 2:
                label_names[0] = "crash"
                label_names[-1] = "bull_run"

        self.labels = {}
        for rank, state_id in enumerate(sorted_idx):
            self.labels[int(state_id)] = label_names[rank]

        return self.labels

    def regime_statistics(self) -> pd.DataFrame:
        """
        Compute per-regime statistics:
          - mean log_return, volatility (from emission means/covars)
          - expected duration E[d_i] = 1 / (1 - a_ii)
          - stationary distribution weights
        """
        n = self.n_states
        transmat = self.model.transmat_
        means = self.model.means_
        covars = self.model.covars_

        # Expected duration
        durations = 1.0 / (1.0 - np.diag(transmat))

        # Stationary distribution: left eigenvector of transmat
        eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()

        rows = []
        for i in range(n):
            label = self.labels.get(i, f"state_{i}")
            if self.covariance_type == "full":
                vol = np.sqrt(covars[i][0, 0])
            elif self.covariance_type == "diag":
                vol = np.sqrt(covars[i][0])
            elif self.covariance_type == "spherical":
                vol = np.sqrt(covars[i])
            else:  # tied
                vol = np.sqrt(covars[0, 0])

            rows.append({
                "state": i,
                "label": label,
                "mean_return": means[i, 0],
                "volatility": vol,
                "expected_duration": durations[i],
                "stationary_weight": stationary[i],
            })

        self._regime_stats = pd.DataFrame(rows)
        return self._regime_stats

    def shannon_entropy(self, posteriors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Shannon entropy per bar: H_t = -sum(p_i * log2(p_i))
        Normalized confidence = 1 - H / log2(n_states)
        Returns (entropy, confidence).
        """
        n = self.n_states
        eps = 1e-12
        p = np.clip(posteriors, eps, 1.0)
        entropy = -np.sum(p * np.log2(p), axis=1)
        max_entropy = np.log2(n)
        confidence = 1.0 - entropy / max_entropy if max_entropy > 0 else np.ones_like(entropy)
        return entropy, confidence

    def transition_matrix(self) -> np.ndarray:
        """Return the fitted transition matrix."""
        return self.model.transmat_.copy()

    def log_likelihood_series(self, X: np.ndarray, window: int = 50) -> np.ndarray:
        """
        Compute rolling log-likelihood over a sliding window.
        Useful for detecting model degradation.
        Returns array of length T with NaN for first (window-1) bars.
        """
        T = len(X)
        ll_series = np.full(T, np.nan)
        for t in range(window, T + 1):
            chunk = X[t - window : t]
            try:
                ll_series[t - 1] = self.model.score(chunk) / window
            except Exception:
                ll_series[t - 1] = np.nan
        return ll_series
