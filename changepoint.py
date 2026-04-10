"""
changepoint.py -- Bayesian Online Changepoint Detection (BOCPD).

Implements Adams & MacKay (2007): exact online inference over a
run-length distribution using conjugate Student-t sufficient statistics.
At every bar, computes P(changepoint NOW) -- the probability that the
current observation starts a new segment.

Key insight for HMM regime trading: the HMM tells you WHICH regime
you're in; BOCPD tells you WHEN the regime is changing. Fusing both
gives faster, higher-confidence regime transition signals.

Reference:
  Adams, R.P. & MacKay, D.J.C. (2007).
  "Bayesian Online Changepoint Detection."
  arXiv:0710.3742
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.special import gammaln, logsumexp


@dataclass
class BOCPDResult:
    """Container for BOCPD outputs."""
    # Per-bar changepoint probability: P(r_t = 0)
    changepoint_prob: np.ndarray
    # Per-bar MAP run length (most likely segment length)
    map_run_length: np.ndarray
    # Per-bar expected run length E[r_t]
    expected_run_length: np.ndarray
    # Full run-length distribution (T, max_run) -- sparse, trimmed
    run_length_dist: np.ndarray
    # Detected changepoint indices (where prob > threshold)
    changepoint_indices: np.ndarray
    # Growth probability: P(no changepoint) = 1 - P(changepoint)
    growth_prob: np.ndarray


class StudentTSuffStats:
    """
    Conjugate sufficient statistics for univariate Gaussian with
    unknown mean and variance, yielding Student-t predictive distribution.

    Prior: Normal-Inverse-Gamma(mu0, kappa0, alpha0, beta0)

    Predictive: Student-t with
        df    = 2 * alpha
        loc   = mu
        scale = sqrt(beta * (kappa + 1) / (alpha * kappa))
    """

    __slots__ = ("mu", "kappa", "alpha", "beta")

    def __init__(self, mu0: float, kappa0: float, alpha0: float, beta0: float):
        self.mu = np.array([mu0])
        self.kappa = np.array([kappa0])
        self.alpha = np.array([alpha0])
        self.beta = np.array([beta0])

    def update(self, x: float):
        """Bayesian update: incorporate one observation into all run lengths."""
        new_kappa = self.kappa + 1
        new_mu = (self.kappa * self.mu + x) / new_kappa
        new_alpha = self.alpha + 0.5
        new_beta = self.beta + (self.kappa * (x - self.mu) ** 2) / (2 * new_kappa)

        self.mu = new_mu
        self.kappa = new_kappa
        self.alpha = new_alpha
        self.beta = new_beta

    def pred_logpdf(self, x: float) -> np.ndarray:
        """
        Log-pdf of the Student-t predictive distribution at x,
        evaluated for each current run length hypothesis.

        Student-t(x | df, loc, scale):
          log p = gammaln((df+1)/2) - gammaln(df/2) - 0.5*log(df*pi*scale^2)
                  - (df+1)/2 * log(1 + ((x-loc)/scale)^2 / df)
        """
        df = 2.0 * self.alpha
        loc = self.mu
        scale2 = self.beta * (self.kappa + 1) / (self.alpha * self.kappa)
        scale2 = np.maximum(scale2, 1e-30)  # numerical guard

        z = (x - loc) / np.sqrt(scale2)
        logp = (
            gammaln((df + 1) / 2)
            - gammaln(df / 2)
            - 0.5 * np.log(df * np.pi * scale2)
            - ((df + 1) / 2) * np.log1p(z ** 2 / df)
        )
        return logp

    def prepend_prior(self, mu0: float, kappa0: float, alpha0: float, beta0: float):
        """Prepend fresh prior for the new run-length=0 hypothesis."""
        self.mu = np.concatenate([[mu0], self.mu])
        self.kappa = np.concatenate([[kappa0], self.kappa])
        self.alpha = np.concatenate([[alpha0], self.alpha])
        self.beta = np.concatenate([[beta0], self.beta])

    def trim(self, max_len: int):
        """Trim to keep only the most recent max_len hypotheses."""
        if len(self.mu) > max_len:
            self.mu = self.mu[:max_len]
            self.kappa = self.kappa[:max_len]
            self.alpha = self.alpha[:max_len]
            self.beta = self.beta[:max_len]


class BayesianChangepoint:
    """
    Bayesian Online Changepoint Detection for univariate time series.

    At each timestep, maintains a distribution over the "run length"
    (number of bars since the last changepoint). A spike in P(r_t=0)
    signals that the current observation likely starts a new segment.

    Parameters
    ----------
    hazard_rate : float
        Expected number of bars between changepoints.
        P(changepoint) = 1/hazard_rate per bar.
    mu0 : float
        Prior mean for segment observations.
    kappa0 : float
        Prior precision weight (strength of mean prior).
    alpha0 : float
        Prior shape for inverse-gamma variance prior.
    beta0 : float
        Prior scale for inverse-gamma variance prior.
    max_run_length : int
        Truncate run-length distribution to this length for efficiency.
    threshold : float
        Changepoint probability threshold for flagging detections.
    """

    def __init__(
        self,
        hazard_rate: float = 100.0,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        max_run_length: int = 300,
        threshold: float = 0.3,
    ):
        self.hazard_rate = hazard_rate
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.max_run_length = max_run_length
        self.threshold = threshold

        # Hazard function: constant rate
        self.H = 1.0 / hazard_rate

    def detect(self, data: np.ndarray) -> BOCPDResult:
        """
        Run BOCPD on a 1-D array of observations.

        All message passing is done in log-space to avoid underflow when
        predictive likelihoods are very small (common with financial data).

        Note on changepoint probability: with a constant hazard function,
        the marginal P(r_t=0) is always H (a well-known identity). The true
        changepoint information lives in the full run-length distribution.
        We derive the changepoint signal from the expected run length drop:
        a sharp decrease in E[r_t] indicates a regime change just occurred.

        Returns BOCPDResult with changepoint probabilities, run lengths,
        and detected changepoint indices.
        """
        T = len(data)
        log_H = np.log(self.H)
        log_1mH = np.log(1.0 - self.H)

        # Initialize log run-length distribution: P(r_0 = 0) = 1 -> log = 0
        log_rl = np.array([0.0])

        # Sufficient statistics tracker
        stats = StudentTSuffStats(self.mu0, self.kappa0, self.alpha0, self.beta0)

        # Output arrays
        map_rl = np.zeros(T, dtype=int)
        expected_rl = np.zeros(T)
        rl_dist = np.zeros((T, min(self.max_run_length, T + 1)))

        for t in range(T):
            x = data[t]

            # 1. Evaluate predictive log-probability under each run length
            pred_logp = stats.pred_logpdf(x)

            # 2. Growth log-probabilities: log P(r_t = r+1, x_{1:t})
            log_growth = log_rl + pred_logp + log_1mH

            # 3. Changepoint log-probability: log P(r_t = 0, x_{1:t})
            log_cp_terms = log_rl + pred_logp + log_H
            log_cp = logsumexp(log_cp_terms)

            # 4. Assemble new log run-length distribution
            new_log_rl = np.concatenate([[log_cp], log_growth])

            # 5. Normalize in log-space
            log_evidence = logsumexp(new_log_rl)
            new_log_rl -= log_evidence

            # 6. Update sufficient statistics with the new observation
            stats.update(x)

            # 7. Prepend fresh prior for the new r=0 hypothesis
            stats.prepend_prior(self.mu0, self.kappa0, self.alpha0, self.beta0)

            # 8. Trim for efficiency
            if len(new_log_rl) > self.max_run_length:
                new_log_rl = new_log_rl[:self.max_run_length]
                new_log_rl -= logsumexp(new_log_rl)
                stats.trim(self.max_run_length)

            log_rl = new_log_rl

            # Convert to probabilities for output
            rl_probs = np.exp(log_rl)

            # Record outputs
            map_rl[t] = np.argmax(rl_probs)
            r_vals = np.arange(len(rl_probs))
            expected_rl[t] = np.sum(r_vals * rl_probs)

            # Store run-length distribution (trimmed to output size)
            store_len = min(len(rl_probs), rl_dist.shape[1])
            rl_dist[t, :store_len] = rl_probs[:store_len]

        # Derive changepoint probability from expected run length drops.
        # When E[r_t] drops sharply relative to E[r_{t-1}], a changepoint
        # just occurred. We use an exponential transform to map the
        # fractional drop into [0, 1].
        cp_prob = np.zeros(T)
        for t in range(1, T):
            prev_erl = expected_rl[t - 1]
            if prev_erl > 1.0:
                drop = max(0.0, prev_erl - expected_rl[t])
                # Fractional drop: 1.0 = complete reset, 0.0 = no change
                frac_drop = drop / prev_erl
                # Exponential scaling: sharper response to large drops
                cp_prob[t] = 1.0 - np.exp(-5.0 * frac_drop)

        # Detect changepoints above threshold
        cp_indices = np.where(cp_prob > self.threshold)[0]

        return BOCPDResult(
            changepoint_prob=cp_prob,
            map_run_length=map_rl,
            expected_run_length=expected_rl,
            run_length_dist=rl_dist,
            changepoint_indices=cp_indices,
            growth_prob=1.0 - cp_prob,
        )

    def detect_multivariate(
        self, data: np.ndarray, feature_weights: np.ndarray | None = None
    ) -> BOCPDResult:
        """
        Run BOCPD on multivariate data by computing a weighted combination
        of per-feature changepoint probabilities.

        This is a practical approach that avoids the complexity of
        multivariate conjugate priors while still leveraging all features.

        Parameters
        ----------
        data : ndarray of shape (T, D)
            Multivariate time series.
        feature_weights : ndarray of shape (D,), optional
            Importance weights per feature. Default: equal weights.

        Returns
        -------
        BOCPDResult with combined changepoint probabilities.
        """
        T, D = data.shape
        if feature_weights is None:
            feature_weights = np.ones(D) / D
        else:
            feature_weights = feature_weights / feature_weights.sum()

        # Run BOCPD on each feature
        per_feature_cp = np.zeros((T, D))
        per_feature_rl = np.zeros((T, D))

        for d in range(D):
            result = self.detect(data[:, d])
            per_feature_cp[:, d] = result.changepoint_prob
            per_feature_rl[:, d] = result.expected_run_length

        # Weighted combination of changepoint probabilities
        combined_cp = (per_feature_cp * feature_weights[np.newaxis, :]).sum(axis=1)
        combined_rl = (per_feature_rl * feature_weights[np.newaxis, :]).sum(axis=1)

        # MAP run length from the primary feature (log_return, feature 0)
        primary_result = self.detect(data[:, 0])

        cp_indices = np.where(combined_cp > self.threshold)[0]

        return BOCPDResult(
            changepoint_prob=combined_cp,
            map_run_length=primary_result.map_run_length,
            expected_run_length=combined_rl,
            run_length_dist=primary_result.run_length_dist,
            changepoint_indices=cp_indices,
            growth_prob=1.0 - combined_cp,
        )


class ChangepointHMMFusion:
    """
    Fuses BOCPD changepoint signals with HMM regime posteriors to produce
    faster, higher-confidence regime transition detection.

    The fusion logic:
    1. BOCPD provides P(changepoint at t)
    2. HMM provides regime confidence (1 - normalized entropy)
    3. When BOCPD fires AND HMM confidence drops, we have high-conviction
       evidence of a regime transition in progress
    4. This "transition urgency" score can override hysteresis delay
       in the signal generator

    The key formula:
        urgency_t = cp_prob_t * (1 - hmm_confidence_t) * entropy_gradient_t

    High urgency means: changepoint detected + HMM uncertain + uncertainty rising
    → strong evidence the regime is actively changing.
    """

    def __init__(
        self,
        cp_weight: float = 0.5,
        entropy_weight: float = 0.3,
        posterior_shift_weight: float = 0.2,
        urgency_threshold: float = 0.4,
        entropy_gradient_window: int = 5,
    ):
        self.cp_weight = cp_weight
        self.entropy_weight = entropy_weight
        self.posterior_shift_weight = posterior_shift_weight
        self.urgency_threshold = urgency_threshold
        self.entropy_gradient_window = entropy_gradient_window

    def compute_transition_urgency(
        self,
        cp_prob: np.ndarray,
        entropy: np.ndarray,
        posteriors: np.ndarray,
        n_states: int,
    ) -> np.ndarray:
        """
        Compute per-bar transition urgency score in [0, 1].

        Combines three signals:
        1. Changepoint probability (BOCPD)
        2. Entropy level (normalized, high = uncertain = transitioning)
        3. Posterior shift magnitude (how fast the dominant state is changing)

        Parameters
        ----------
        cp_prob : ndarray (T,)
            BOCPD changepoint probabilities.
        entropy : ndarray (T,)
            Shannon entropy from HMM posteriors.
        posteriors : ndarray (T, n_states)
            HMM posterior probabilities.
        n_states : int
            Number of HMM states.

        Returns
        -------
        urgency : ndarray (T,)
            Transition urgency score, 0 (stable) to 1 (actively transitioning).
        """
        T = len(cp_prob)
        urgency = np.zeros(T)

        # Normalize entropy to [0, 1]
        max_entropy = np.log2(n_states) if n_states > 1 else 1.0
        norm_entropy = np.clip(entropy / max_entropy, 0, 1)

        # Posterior shift: L1 distance between consecutive posteriors
        posterior_shift = np.zeros(T)
        for t in range(1, T):
            posterior_shift[t] = 0.5 * np.sum(np.abs(posteriors[t] - posteriors[t - 1]))

        # Entropy gradient: rising entropy suggests regime destabilization
        entropy_grad = np.zeros(T)
        w = self.entropy_gradient_window
        for t in range(w, T):
            entropy_grad[t] = max(0, entropy[t] - entropy[t - w])

        # Normalize entropy gradient to [0, 1]
        eg_max = entropy_grad.max()
        if eg_max > 0:
            entropy_grad /= eg_max

        # Combine signals
        urgency = (
            self.cp_weight * cp_prob
            + self.entropy_weight * norm_entropy * entropy_grad
            + self.posterior_shift_weight * posterior_shift
        )

        return np.clip(urgency, 0, 1)

    def adaptive_hysteresis(
        self,
        urgency: np.ndarray,
        base_hysteresis: int,
        min_hysteresis: int = 0,
    ) -> np.ndarray:
        """
        Compute per-bar adaptive hysteresis: reduce the required regime
        persistence when transition urgency is high.

        When urgency is 0 (stable): use full base_hysteresis
        When urgency is 1 (actively transitioning): use min_hysteresis

        Parameters
        ----------
        urgency : ndarray (T,)
            Transition urgency scores.
        base_hysteresis : int
            Default hysteresis from config (e.g. 3 bars).
        min_hysteresis : int
            Minimum hysteresis even under max urgency.

        Returns
        -------
        adaptive_hyst : ndarray (T,) of int
            Per-bar hysteresis requirement.
        """
        hyst_range = base_hysteresis - min_hysteresis
        adaptive = base_hysteresis - (urgency * hyst_range)
        return np.maximum(adaptive, min_hysteresis).astype(int)

    def fuse(
        self,
        bocpd_result: BOCPDResult,
        entropy: np.ndarray,
        confidence: np.ndarray,
        posteriors: np.ndarray,
        n_states: int,
        base_hysteresis: int = 3,
    ) -> dict:
        """
        Full fusion pipeline: BOCPD + HMM → transition urgency + adaptive hysteresis.

        Returns dict with all fusion outputs.
        """
        urgency = self.compute_transition_urgency(
            bocpd_result.changepoint_prob, entropy, posteriors, n_states
        )

        adaptive_hyst = self.adaptive_hysteresis(urgency, base_hysteresis)

        # Enhanced confidence: reduce confidence when changepoint is likely
        # (the model should be cautious during transitions)
        transition_penalty = 1.0 - 0.5 * bocpd_result.changepoint_prob
        enhanced_confidence = confidence * transition_penalty

        # Transition alert: high urgency points
        alert_indices = np.where(urgency > self.urgency_threshold)[0]

        return {
            "urgency": urgency,
            "adaptive_hysteresis": adaptive_hyst,
            "enhanced_confidence": enhanced_confidence,
            "transition_alerts": alert_indices,
            "changepoint_prob": bocpd_result.changepoint_prob,
            "map_run_length": bocpd_result.map_run_length,
            "expected_run_length": bocpd_result.expected_run_length,
            "run_length_dist": bocpd_result.run_length_dist,
        }


def create_bocpd_from_config(config: dict) -> BayesianChangepoint:
    """Factory function to create BOCPD from config.yaml."""
    cp_cfg = config.get("changepoint", {})
    return BayesianChangepoint(
        hazard_rate=cp_cfg.get("hazard_rate", 100),
        mu0=cp_cfg.get("mu0", 0.0),
        kappa0=cp_cfg.get("kappa0", 1.0),
        alpha0=cp_cfg.get("alpha0", 1.0),
        beta0=cp_cfg.get("beta0", 1.0),
        max_run_length=cp_cfg.get("max_run_length", 300),
        threshold=cp_cfg.get("threshold", 0.3),
    )


def create_fusion_from_config(config: dict) -> ChangepointHMMFusion:
    """Factory function to create fusion engine from config.yaml."""
    cp_cfg = config.get("changepoint", {})
    fusion_cfg = cp_cfg.get("fusion", {})
    return ChangepointHMMFusion(
        cp_weight=fusion_cfg.get("cp_weight", 0.5),
        entropy_weight=fusion_cfg.get("entropy_weight", 0.3),
        posterior_shift_weight=fusion_cfg.get("posterior_shift_weight", 0.2),
        urgency_threshold=fusion_cfg.get("urgency_threshold", 0.4),
        entropy_gradient_window=fusion_cfg.get("entropy_gradient_window", 5),
    )
