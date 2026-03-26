"""
changepoint.py — Bayesian Online Changepoint Detection (BOCPD).

Implements the Adams & MacKay (2007) algorithm for real-time detection
of regime changes. Complements HMM by detecting *when* regimes shift
(often faster than Viterbi) while HMM identifies *which* regime.

The algorithm maintains a run-length distribution at each timestep:
  P(r_t | x_{1:t})
where r_t is the number of steps since the last changepoint.
A spike in changepoint probability (r_t = 0) signals a regime shift.

Uses a conjugate Normal-Inverse-Gamma prior for Gaussian observations,
enabling exact Bayesian inference without MCMC.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BOCPDResult:
    """Container for BOCPD analysis results."""
    changepoint_prob: np.ndarray      # P(changepoint) at each bar
    run_length_map: np.ndarray        # (T, T) run-length posterior matrix
    expected_run_length: np.ndarray   # E[r_t] at each bar
    map_run_length: np.ndarray        # argmax run length at each bar
    detected_changepoints: list[int]  # bar indices of detected changepoints
    segments: list[dict]              # segment boundaries and stats


class BayesianChangepointDetector:
    """
    Online Bayesian Changepoint Detection (Adams & MacKay 2007).

    Maintains a posterior over run lengths at each step. When the
    posterior mass on r_t=0 spikes, a changepoint is detected.

    Parameters
    ----------
    hazard_rate : float
        Prior probability of a changepoint at any given bar.
        Lower = fewer expected changepoints (longer regimes).
        Typical: 1/100 to 1/250 for financial data.
    mu_prior : float
        Prior mean for the Normal-Inverse-Gamma conjugate.
    kappa_prior : float
        Prior precision scale (higher = more confident in mu_prior).
    alpha_prior : float
        Prior shape for inverse-gamma variance. Must be > 0.
    beta_prior : float
        Prior scale for inverse-gamma variance. Must be > 0.
    threshold : float
        Changepoint probability threshold for detection (0-1).
    """

    def __init__(self, config: dict):
        cp_cfg = config.get("changepoint", {})
        self.hazard_rate = cp_cfg.get("hazard_rate", 1 / 150)
        self.mu_prior = cp_cfg.get("mu_prior", 0.0)
        self.kappa_prior = cp_cfg.get("kappa_prior", 1.0)
        self.alpha_prior = cp_cfg.get("alpha_prior", 1.0)
        self.beta_prior = cp_cfg.get("beta_prior", 1.0)
        self.threshold = cp_cfg.get("threshold", 0.3)

    def _hazard_function(self, r: np.ndarray) -> np.ndarray:
        """Constant hazard: P(changepoint) = hazard_rate at every step."""
        return np.full_like(r, self.hazard_rate, dtype=float)

    def _student_t_pdf(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """
        Predictive distribution under Normal-Inverse-Gamma conjugate:
        a Student-t with:
          df = 2 * alpha
          loc = mu
          scale = sqrt(beta * (kappa + 1) / (alpha * kappa))
        """
        df = 2.0 * alpha
        scale_sq = beta * (kappa + 1.0) / (alpha * kappa)
        scale = np.sqrt(scale_sq)

        # Student-t log-pdf for numerical stability
        from scipy.special import gammaln

        log_pdf = (
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * np.log(df * np.pi * scale_sq)
            - ((df + 1.0) / 2.0) * np.log(1.0 + ((x - mu) ** 2) / (df * scale_sq))
        )

        return np.exp(log_pdf)

    def detect(self, data: np.ndarray) -> BOCPDResult:
        """
        Run BOCPD on a 1-D time series.

        Parameters
        ----------
        data : np.ndarray
            1-D array of observations (e.g., log returns).

        Returns
        -------
        BOCPDResult with changepoint probabilities, run-length posterior,
        detected changepoints, and segment statistics.
        """
        T = len(data)

        # Run-length posterior: R[t, r] = P(r_t = r | x_{1:t})
        # Only need to store up to t+1 run lengths at time t
        R = np.zeros((T + 1, T + 1))
        R[0, 0] = 1.0  # Prior: run length 0 with probability 1

        # Sufficient statistics for each run length (Normal-Inverse-Gamma)
        mu = np.full(T + 1, self.mu_prior)
        kappa = np.full(T + 1, self.kappa_prior)
        alpha = np.full(T + 1, self.alpha_prior)
        beta = np.full(T + 1, self.beta_prior)

        changepoint_prob = np.zeros(T)
        expected_run_length = np.zeros(T)
        map_run_length = np.zeros(T, dtype=int)

        for t in range(T):
            x = data[t]

            # 1. Evaluate predictive probability under each run length
            pred_probs = self._student_t_pdf(x, mu[: t + 1], kappa[: t + 1],
                                             alpha[: t + 1], beta[: t + 1])

            # 2. Compute growth probabilities (run continues)
            hazard = self._hazard_function(np.arange(t + 1))
            growth = R[t, : t + 1] * pred_probs * (1.0 - hazard)

            # 3. Compute changepoint probability (new run starts)
            cp = np.sum(R[t, : t + 1] * pred_probs * hazard)

            # 4. Update run-length posterior
            R[t + 1, 0] = cp
            R[t + 1, 1 : t + 2] = growth

            # Normalize
            evidence = R[t + 1, : t + 2].sum()
            if evidence > 0:
                R[t + 1, : t + 2] /= evidence

            # 5. Update sufficient statistics (NIG conjugate update)
            new_mu = (kappa[: t + 1] * mu[: t + 1] + x) / (kappa[: t + 1] + 1.0)
            new_kappa = kappa[: t + 1] + 1.0
            new_alpha = alpha[: t + 1] + 0.5
            new_beta = (
                beta[: t + 1]
                + 0.5 * kappa[: t + 1] * (x - mu[: t + 1]) ** 2 / (kappa[: t + 1] + 1.0)
            )

            # Shift: run length i at time t becomes run length i+1 at time t+1
            mu_new = np.full(T + 1, self.mu_prior)
            kappa_new = np.full(T + 1, self.kappa_prior)
            alpha_new = np.full(T + 1, self.alpha_prior)
            beta_new = np.full(T + 1, self.beta_prior)

            mu_new[1 : t + 2] = new_mu
            kappa_new[1 : t + 2] = new_kappa
            alpha_new[1 : t + 2] = new_alpha
            beta_new[1 : t + 2] = new_beta

            mu = mu_new
            kappa = kappa_new
            alpha = alpha_new
            beta = beta_new

            # Record results
            run_lengths = np.arange(t + 2)
            posterior = R[t + 1, : t + 2]
            expected_run_length[t] = np.sum(run_lengths * posterior)
            map_run_length[t] = np.argmax(posterior)

            # Changepoint score: mass on short run lengths (r < short_window)
            # This captures the posterior belief that a changepoint happened recently.
            # With constant hazard, P(r=0) = hazard always, so we instead
            # measure how much mass is on short run lengths vs the prior expectation.
            short_window = min(5, t + 2)
            short_mass = np.sum(posterior[:short_window])
            # Baseline: under geometric(hazard) prior, mass on r<5 ≈ 1-(1-h)^5
            baseline_short = 1.0 - (1.0 - self.hazard_rate) ** short_window
            changepoint_prob[t] = max(0.0, min(1.0,
                (short_mass - baseline_short) / max(1.0 - baseline_short, 1e-10)
            ))

        # Detect changepoints using run-length-based score
        detected = self._detect_changepoints(changepoint_prob)

        # Build segments
        segments = self._build_segments(data, detected)

        # Trim run-length matrix to (T, T) for storage
        run_length_map = R[1 : T + 1, :T]

        return BOCPDResult(
            changepoint_prob=changepoint_prob,
            run_length_map=run_length_map,
            expected_run_length=expected_run_length,
            map_run_length=map_run_length,
            detected_changepoints=detected,
            segments=segments,
        )

    def _detect_changepoints(self, cp_prob: np.ndarray) -> list[int]:
        """Find peaks in changepoint probability above threshold.

        Skips the first `warmup` bars where the prior dominates.
        """
        detected = []
        T = len(cp_prob)
        warmup = min(10, T // 5)  # ignore initial settling period

        for t in range(warmup, T):
            if cp_prob[t] < self.threshold:
                continue
            # Local peak: higher than both neighbors (or at boundary)
            is_peak = True
            if t > warmup and cp_prob[t] < cp_prob[t - 1]:
                is_peak = False
            if t < T - 1 and cp_prob[t] < cp_prob[t + 1]:
                is_peak = False
            if is_peak:
                detected.append(t)

        return sorted(detected)

    def _build_segments(self, data: np.ndarray, changepoints: list[int]) -> list[dict]:
        """Build segments between consecutive changepoints with statistics."""
        T = len(data)
        boundaries = [0] + changepoints + [T]
        boundaries = sorted(set(boundaries))

        segments = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment_data = data[start:end]
            if len(segment_data) == 0:
                continue
            segments.append({
                "start": start,
                "end": end,
                "length": end - start,
                "mean": float(np.mean(segment_data)),
                "std": float(np.std(segment_data)) if len(segment_data) > 1 else 0.0,
                "cumulative": float(np.sum(segment_data)),
            })
        return segments


def ensemble_hmm_bocpd(
    hmm_states: np.ndarray,
    hmm_posteriors: np.ndarray,
    hmm_entropy: np.ndarray,
    hmm_confidence: np.ndarray,
    hmm_labels: dict[int, str],
    bocpd_result: BOCPDResult,
    hmm_transition_bars: list[int] | None = None,
    agreement_window: int = 5,
) -> pd.DataFrame:
    """
    Ensemble HMM regime detection with BOCPD changepoint detection.

    Creates a unified "Regime Radar" combining both methods:
    - HMM provides regime identity (bull/bear/crash)
    - BOCPD provides changepoint timing (often earlier)

    Outputs per-bar metrics:
    - regime_change_score: 0-1, how likely a regime change is happening
    - early_warning: BOCPD detected change before HMM caught up
    - consensus_confidence: agreement-boosted confidence
    - regime_stability: how stable the current regime is

    Parameters
    ----------
    hmm_states : array of Viterbi-decoded state indices
    hmm_posteriors : (T, n_states) posterior matrix
    hmm_entropy : Shannon entropy per bar
    hmm_confidence : 1 - normalized entropy per bar
    hmm_labels : state index -> label mapping
    bocpd_result : output from BayesianChangepointDetector.detect()
    hmm_transition_bars : bars where HMM detected regime transitions
    agreement_window : bars to look for agreement between methods
    """
    T = len(hmm_states)
    cp_prob = bocpd_result.changepoint_prob

    # Detect HMM transition bars if not provided
    if hmm_transition_bars is None:
        hmm_transition_bars = []
        for t in range(1, T):
            if hmm_states[t] != hmm_states[t - 1]:
                hmm_transition_bars.append(t)

    hmm_transition_set = set(hmm_transition_bars)

    # Build BOCPD changepoint proximity signal (smoothed)
    bocpd_signal = np.zeros(T)
    for cp in bocpd_result.detected_changepoints:
        for offset in range(-agreement_window, agreement_window + 1):
            idx = cp + offset
            if 0 <= idx < T:
                weight = 1.0 - abs(offset) / (agreement_window + 1)
                bocpd_signal[idx] = max(bocpd_signal[idx], weight)

    # Build HMM transition proximity signal
    hmm_signal = np.zeros(T)
    for tr in hmm_transition_bars:
        for offset in range(-agreement_window, agreement_window + 1):
            idx = tr + offset
            if 0 <= idx < T:
                weight = 1.0 - abs(offset) / (agreement_window + 1)
                hmm_signal[idx] = max(hmm_signal[idx], weight)

    # Compute per-bar ensemble metrics
    records = []
    for t in range(T):
        # Regime change score: weighted combination
        hmm_change = hmm_signal[t]
        bocpd_change = cp_prob[t]
        combined_change = 0.6 * bocpd_change + 0.4 * hmm_change

        # Agreement: both methods detect a change nearby
        agreement = min(hmm_signal[t], bocpd_signal[t])

        # Early warning: BOCPD fires but HMM hasn't caught up yet
        early_warning = max(0.0, bocpd_signal[t] - hmm_signal[t])

        # Consensus confidence: boost when both agree, reduce on disagreement
        base_conf = hmm_confidence[t]
        if agreement > 0.5:
            # Both agree a change is happening — high confidence in the transition
            consensus_conf = min(1.0, base_conf * (1.0 + 0.3 * agreement))
        elif early_warning > 0.5:
            # BOCPD sees change but HMM doesn't — reduce confidence
            consensus_conf = base_conf * (1.0 - 0.2 * early_warning)
        else:
            consensus_conf = base_conf

        # Regime stability: inverse of recent changepoint activity
        lookback = min(t + 1, 20)
        recent_cp = np.mean(cp_prob[max(0, t - lookback + 1) : t + 1])
        stability = 1.0 - min(recent_cp * 3.0, 1.0)

        # Regime acceleration: rate of change of run length
        if t >= 2:
            rl = bocpd_result.expected_run_length
            accel = rl[t] - 2 * rl[t - 1] + rl[t - 2]
        else:
            accel = 0.0

        records.append({
            "bar": t,
            "regime": hmm_labels.get(int(hmm_states[t]), "unknown"),
            "hmm_confidence": float(hmm_confidence[t]),
            "bocpd_changepoint_prob": float(cp_prob[t]),
            "expected_run_length": float(bocpd_result.expected_run_length[t]),
            "regime_change_score": float(combined_change),
            "agreement": float(agreement),
            "early_warning": float(early_warning),
            "consensus_confidence": float(consensus_conf),
            "regime_stability": float(stability),
            "run_length_accel": float(accel),
            "is_hmm_transition": t in hmm_transition_set,
            "is_bocpd_changepoint": t in bocpd_result.detected_changepoints,
        })

    return pd.DataFrame(records)


def compute_detection_lead(
    hmm_transitions: list[int],
    bocpd_changepoints: list[int],
    max_distance: int = 20,
) -> dict:
    """
    Compute how many bars BOCPD leads or lags HMM in detecting regime changes.

    For each HMM transition, find the nearest BOCPD changepoint within max_distance.
    Negative lead = BOCPD detected first (earlier = better).

    Returns dict with:
    - matched_pairs: list of (hmm_bar, bocpd_bar, lead_bars)
    - mean_lead: average lead (negative = BOCPD is faster)
    - bocpd_early_pct: % of transitions BOCPD detected first
    - unmatched_hmm: HMM transitions with no BOCPD counterpart
    - unmatched_bocpd: BOCPD changepoints with no HMM counterpart
    """
    if not hmm_transitions or not bocpd_changepoints:
        return {
            "matched_pairs": [],
            "mean_lead": 0.0,
            "bocpd_early_pct": 0.0,
            "n_matched": 0,
            "unmatched_hmm": list(hmm_transitions),
            "unmatched_bocpd": list(bocpd_changepoints),
        }

    matched = []
    used_bocpd = set()
    used_hmm = set()

    for hmm_bar in hmm_transitions:
        best_dist = max_distance + 1
        best_cp = None
        for cp in bocpd_changepoints:
            if cp in used_bocpd:
                continue
            dist = abs(hmm_bar - cp)
            if dist < best_dist:
                best_dist = dist
                best_cp = cp

        if best_cp is not None and best_dist <= max_distance:
            lead = hmm_bar - best_cp  # positive = BOCPD detected first
            matched.append((hmm_bar, best_cp, lead))
            used_bocpd.add(best_cp)
            used_hmm.add(hmm_bar)

    leads = [m[2] for m in matched]
    early_count = sum(1 for l in leads if l > 0)

    return {
        "matched_pairs": matched,
        "mean_lead": float(np.mean(leads)) if leads else 0.0,
        "median_lead": float(np.median(leads)) if leads else 0.0,
        "bocpd_early_pct": early_count / len(matched) if matched else 0.0,
        "n_matched": len(matched),
        "unmatched_hmm": [b for b in hmm_transitions if b not in used_hmm],
        "unmatched_bocpd": [b for b in bocpd_changepoints if b not in used_bocpd],
    }
