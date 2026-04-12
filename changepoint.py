"""
changepoint.py — Bayesian Online Changepoint Detection (BOCD).

Implements Adams & MacKay (2007) for real-time structural break detection
in financial time series. Complements the HMM by providing an independent,
online regime-change signal that requires no batch refit and often detects
transitions earlier than Viterbi decoding.

Key math:
  - Maintains posterior over run length r_t (bars since last changepoint)
  - Uses Normal-Inverse-Gamma conjugate prior for O(1) Bayesian updates
  - Predictive distribution is Student-t with 2*alpha degrees of freedom
  - Constant hazard H = 1/lambda controls expected run length
  - P(changepoint at t) = P(r_t = 0 | x_{1:t})

Reference:
  Adams, R.P. & MacKay, D.J.C. (2007)
  "Bayesian Online Changepoint Detection"
  arXiv:0710.3742
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln
from dataclasses import dataclass, field


@dataclass
class ChangepointResult:
    """Container for BOCD outputs."""

    changepoint_prob: np.ndarray       # P(changepoint at t) for each bar
    run_length_dist: np.ndarray        # (T, R_max) run-length posterior matrix
    max_run_length: np.ndarray         # MAP run length per bar
    detected_changepoints: list[int]   # bar indices where P(cp) > threshold
    hazard_rate: float

    # Comparison with HMM transitions (populated by compare_with_hmm_transitions)
    early_detection_bars: list[int] = field(default_factory=list)
    avg_early_detection: float = 0.0


class BayesianChangepointDetector:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

    Maintains a posterior over run lengths using Normal-Inverse-Gamma
    conjugate priors for efficient O(1)-per-step Bayesian updates.

    The run-length r_t is the number of bars since the last changepoint.
    At each step the algorithm computes:
      - Growth: P(r_t = r_{t-1}+1) — current regime continues
      - Changepoint: P(r_t = 0) — new regime starts

    A spike in P(r_t = 0) signals a structural break.
    """

    def __init__(
        self,
        hazard_rate: float = 1 / 100,
        mu0: float = 0.0,
        kappa0: float = 0.1,
        alpha0: float = 1.0,
        beta0: float = 0.001,
        max_run_length: int = 500,
        detection_threshold: float = 0.3,
        cp_window: int = 10,
    ):
        """
        Parameters
        ----------
        hazard_rate : float
            Prior probability of a changepoint at any bar.
            1/100 means one changepoint per ~100 bars on average.
        mu0 : float
            Prior mean of the Normal-Inverse-Gamma (NIG).
        kappa0 : float
            Prior precision scaling. Lower = let data speak faster.
        alpha0 : float
            Prior shape of the Inverse-Gamma on variance.
        beta0 : float
            Prior rate of the Inverse-Gamma on variance.
            For hourly log returns (~0.005 std), 0.001 is reasonable.
        max_run_length : int
            Truncate the run-length distribution at this value.
            Keeps memory and compute bounded at O(T * R_max).
        detection_threshold : float
            Minimum P(young segment) to flag a detected changepoint.
        cp_window : int
            Changepoint score = P(run_length < cp_window).
            This measures the probability that we are within cp_window
            bars of a structural break. In stable data this is low
            (~H*cp_window); at changepoints it spikes toward 1.
        """
        self.hazard = hazard_rate
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.max_run_length = max_run_length
        self.detection_threshold = detection_threshold
        self.cp_window = cp_window

    def run(self, data: np.ndarray) -> ChangepointResult:
        """
        Run BOCD on a 1-D time series.

        Parameters
        ----------
        data : array of shape (T,)
            The time series (typically log returns).

        Returns
        -------
        ChangepointResult with changepoint probabilities,
        run-length distribution, and detected changepoints.
        """
        T = len(data)
        R_max = min(T + 1, self.max_run_length)

        # Run-length probability vector (current timestep)
        R = np.zeros(R_max)
        R[0] = 1.0

        # Normal-Inverse-Gamma sufficient statistics per run length
        mu = np.full(R_max, self.mu0)
        kappa = np.full(R_max, self.kappa0)
        alpha = np.full(R_max, self.alpha0)
        beta = np.full(R_max, self.beta0)

        # Output arrays
        cp_prob = np.zeros(T)
        map_run = np.zeros(T, dtype=int)
        rl_dist = np.zeros((T, R_max))

        for t in range(T):
            x = data[t]
            active = min(t + 1, R_max)

            # --- Step 1: Predictive probabilities (Student-t) ---
            # P(x_t | run_length = r) for each active run length
            nu = 2 * alpha[:active]
            var = beta[:active] * (kappa[:active] + 1) / (
                alpha[:active] * kappa[:active]
            )
            var = np.maximum(var, 1e-12)
            z = x - mu[:active]

            # Log Student-t PDF for numerical stability
            log_pred = (
                gammaln((nu + 1) / 2)
                - gammaln(nu / 2)
                - 0.5 * np.log(nu * np.pi * var)
                - (nu + 1) / 2 * np.log1p(z ** 2 / (nu * var))
            )
            pred = np.exp(log_pred)

            # --- Step 2: Growth and changepoint probabilities ---
            joint = R[:active] * pred
            cp = joint.sum() * self.hazard
            growth = joint * (1 - self.hazard)

            # --- Step 3: Update run-length distribution ---
            new_R = np.zeros(R_max)
            new_R[0] = cp

            grow_end = min(active, R_max - 1)
            new_R[1 : grow_end + 1] = growth[:grow_end]

            # Accumulate at boundary (run lengths that exceed R_max)
            if active >= R_max:
                new_R[R_max - 1] += growth[R_max - 1]

            # Normalize to get posterior
            evidence = new_R.sum()
            if evidence > 0:
                new_R /= evidence

            R = new_R

            # Record outputs
            rl_active = min(t + 2, R_max)
            map_run[t] = np.argmax(R[:rl_active])
            rl_dist[t, :rl_active] = R[:rl_active]

            # Changepoint score: P(run_length < cp_window | data)
            # In stable data this is low; at structural breaks it spikes.
            # Note: P(r=0) is always ≈ hazard_rate by construction,
            # so we use the cumulative P(r < window) instead.
            cp_end = min(self.cp_window, rl_active)
            cp_prob[t] = R[:cp_end].sum()

            # --- Step 4: Update NIG sufficient statistics ---
            new_mu = np.full(R_max, self.mu0)
            new_kappa = np.full(R_max, self.kappa0)
            new_alpha = np.full(R_max, self.alpha0)
            new_beta = np.full(R_max, self.beta0)

            # Shift: stats at run length r+1 = stats at r updated with x
            up_end = min(active, R_max - 1)
            k_prev = kappa[:up_end]
            m_prev = mu[:up_end]
            a_prev = alpha[:up_end]
            b_prev = beta[:up_end]

            new_kappa[1 : up_end + 1] = k_prev + 1
            new_mu[1 : up_end + 1] = (k_prev * m_prev + x) / new_kappa[
                1 : up_end + 1
            ]
            new_alpha[1 : up_end + 1] = a_prev + 0.5
            new_beta[1 : up_end + 1] = b_prev + k_prev * (x - m_prev) ** 2 / (
                2 * new_kappa[1 : up_end + 1]
            )

            # Boundary: in-place update for self-loop at R_max-1
            if active >= R_max:
                r = R_max - 1
                new_kappa[r] = kappa[r] + 1
                new_mu[r] = (kappa[r] * mu[r] + x) / new_kappa[r]
                new_alpha[r] = alpha[r] + 0.5
                new_beta[r] = (
                    beta[r]
                    + kappa[r] * (x - mu[r]) ** 2 / (2 * new_kappa[r])
                )

            mu = new_mu
            kappa = new_kappa
            alpha = new_alpha
            beta = new_beta

        detected = self._detect_changepoints(cp_prob)

        return ChangepointResult(
            changepoint_prob=cp_prob,
            run_length_dist=rl_dist,
            max_run_length=map_run,
            detected_changepoints=detected,
            hazard_rate=self.hazard,
        )

    def _detect_changepoints(
        self, cp_prob: np.ndarray, min_distance: int = 5
    ) -> list[int]:
        """
        Identify changepoint bars where P(cp) exceeds the threshold.

        Uses a cooldown of `min_distance` bars to avoid double-detecting
        the same structural break.
        """
        detected = []
        last_cp = -min_distance
        for t in range(len(cp_prob)):
            if (
                cp_prob[t] > self.detection_threshold
                and (t - last_cp) >= min_distance
            ):
                detected.append(t)
                last_cp = t
        return detected

    def compare_with_hmm_transitions(
        self,
        result: ChangepointResult,
        hmm_transition_bars: list[int],
        max_lookback: int = 30,
    ) -> ChangepointResult:
        """
        Compare BOCD changepoints with HMM regime transitions to
        quantify the early-detection advantage.

        For each HMM transition, finds the nearest prior BOCD changepoint
        within `max_lookback` bars and records how many bars earlier
        BOCD detected the shift.

        Parameters
        ----------
        result : ChangepointResult
            Output from self.run().
        hmm_transition_bars : list[int]
            Bar indices where HMM states[t] != states[t-1].
        max_lookback : int
            Maximum number of bars to look back from each HMM transition.

        Returns
        -------
        Updated ChangepointResult with early_detection_bars and
        avg_early_detection populated.
        """
        early_bars = []

        for hmm_t in hmm_transition_bars:
            best_lead = None
            for cp_t in result.detected_changepoints:
                if hmm_t - max_lookback <= cp_t <= hmm_t:
                    lead = hmm_t - cp_t
                    if best_lead is None or lead > best_lead:
                        best_lead = lead
            if best_lead is not None and best_lead > 0:
                early_bars.append(best_lead)

        result.early_detection_bars = early_bars
        result.avg_early_detection = (
            float(np.mean(early_bars)) if early_bars else 0.0
        )

        return result

    def combined_instability_score(
        self,
        cp_prob: np.ndarray,
        entropy_gradient: np.ndarray,
        weight_bocd: float = 0.6,
        weight_entropy: float = 0.4,
    ) -> np.ndarray:
        """
        Blend BOCD changepoint probability with HMM entropy gradient
        into a single regime-instability score in [0, 1].

        When both BOCD and HMM entropy agree that a transition is
        occurring, the combined score is high — this is much stronger
        evidence than either signal alone.

        Parameters
        ----------
        cp_prob : array of shape (T,)
            BOCD changepoint probability from self.run().
        entropy_gradient : array of shape (T,)
            Change in Shannon entropy over a short window.
            Positive = uncertainty rising = transition brewing.
        weight_bocd : float
            Weight for BOCD signal.
        weight_entropy : float
            Weight for entropy gradient signal.

        Returns
        -------
        Array of shape (T,) in [0, 1].
        """
        T = min(len(cp_prob), len(entropy_gradient))

        # Normalize entropy gradient to [0, 1]
        eg = entropy_gradient[:T].copy()
        eg = np.clip(eg, 0, None)  # only positive gradients matter
        eg_max = eg.max()
        if eg_max > 0:
            eg_norm = eg / eg_max
        else:
            eg_norm = np.zeros(T)

        cp = cp_prob[:T]

        combined = weight_bocd * cp + weight_entropy * eg_norm
        return np.clip(combined, 0.0, 1.0)
