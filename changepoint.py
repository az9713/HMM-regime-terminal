"""
changepoint.py — Bayesian Online Changepoint Detection (BOCD).

Implements the Adams & MacKay (2007) algorithm for real-time detection of
regime changes in financial time series. Unlike the HMM's forward-backward
algorithm (which smooths over past and future), BOCD is fully causal: at
each bar it maintains a posterior distribution over "run lengths" (time
since the last changepoint) and fires when that distribution shifts mass
toward r=0.

Key outputs:
  - run_length_posterior: full (T, T) matrix of P(r_t | x_{1:t})
  - changepoint_probability: P(r_t = 0 | x_{1:t}) per bar
  - map_run_length: most-probable run length per bar
  - detected_changepoints: bars where P(changepoint) exceeds threshold

This provides an orthogonal early-warning signal to the HMM's entropy-based
confidence. While entropy measures "model uncertainty within the current
regime", BOCD measures "probability that the regime itself just changed".

Reference:
  Adams, R.P. & MacKay, D.J.C. (2007). "Bayesian Online Changepoint
  Detection." arXiv:0710.3742.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.stats import t as student_t
from scipy.special import gammaln


@dataclass
class BOCDResult:
    """Container for BOCD outputs."""
    # Per-bar changepoint probability: P(r_t = 0 | x_{1:t})
    changepoint_prob: np.ndarray
    # Most-probable run length at each bar
    map_run_length: np.ndarray
    # Indices of detected changepoints (where prob > threshold)
    changepoints: np.ndarray
    # Growth probability: P(r_t = r_{t-1} + 1 | x_{1:t}) — regime continuation
    growth_prob: np.ndarray
    # Full run-length posterior (sparse representation: list of dicts)
    # Only stored if requested, as it can be large
    run_length_posterior: list[np.ndarray] | None = None


class StudentTPredictor:
    """
    Bayesian predictive model using conjugate Normal-Inverse-Gamma prior.

    For univariate observations, maintains sufficient statistics and
    computes the predictive probability under a Student-t distribution.

    Prior: mu ~ N(mu0, sigma^2 / kappa0), sigma^2 ~ IG(alpha0, beta0)
    Predictive: x_new | x_{1:t} ~ Student-t(2*alpha, mu_n, beta_n*(kappa_n+1)/(alpha_n*kappa_n))
    """

    def __init__(self, mu0: float = 0.0, kappa0: float = 1.0,
                 alpha0: float = 1.0, beta0: float = 1.0):
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

    def predictive_logprob(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """
        Compute log-predictive probability of observation x under each
        run-length hypothesis.

        Returns array of log P(x | r, x_{1:t}) for each active run length.
        """
        df = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))

        # Student-t log-pdf
        z = (x - mu) / scale
        logp = (
            gammaln((df + 1) / 2)
            - gammaln(df / 2)
            - 0.5 * np.log(df * np.pi)
            - np.log(scale)
            - ((df + 1) / 2) * np.log1p(z**2 / df)
        )
        return logp

    def update_sufficient_stats(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Update conjugate sufficient statistics after observing x.

        Returns updated (mu, kappa, alpha, beta) arrays.
        """
        new_kappa = kappa + 1
        new_mu = (kappa * mu + x) / new_kappa
        new_alpha = alpha + 0.5
        new_beta = beta + kappa * (x - mu)**2 / (2 * new_kappa)
        return new_mu, new_kappa, new_alpha, new_beta


class MultivariatePredictor:
    """
    Bayesian predictive model for multivariate observations.

    Uses dimension-independent Normal-Inverse-Gamma priors (one per feature)
    and sums log-predictive probabilities across dimensions.
    This is equivalent to assuming feature independence within a run,
    which is a reasonable approximation that keeps the algorithm O(T*d).
    """

    def __init__(self, n_features: int, mu0: float = 0.0,
                 kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 1.0):
        self.n_features = n_features
        self.predictors = [
            StudentTPredictor(mu0, kappa0, alpha0, beta0)
            for _ in range(n_features)
        ]

    def predictive_logprob(
        self,
        x: np.ndarray,
        stats: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Sum log-predictive across feature dimensions."""
        logp = np.zeros(len(stats[0][0]))
        for d in range(self.n_features):
            mu, kappa, alpha, beta = stats[d]
            logp += self.predictors[d].predictive_logprob(x[d], mu, kappa, alpha, beta)
        return logp

    def update_sufficient_stats(
        self,
        x: np.ndarray,
        stats: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Update sufficient stats for each dimension."""
        new_stats = []
        for d in range(self.n_features):
            mu, kappa, alpha, beta = stats[d]
            new = self.predictors[d].update_sufficient_stats(x[d], mu, kappa, alpha, beta)
            new_stats.append(new)
        return new_stats


class BOCDEngine:
    """
    Bayesian Online Changepoint Detection.

    Maintains a distribution over run lengths P(r_t | x_{1:t}) and
    at each new observation:
      1. Computes predictive probability under each run-length hypothesis
      2. Updates growth probabilities (regime continues)
      3. Computes changepoint probability (regime resets)
      4. Normalizes to get the posterior

    The hazard function H(r) = P(changepoint | run_length = r) controls
    how aggressively changepoints are detected. A constant hazard 1/lambda
    corresponds to a geometric prior on regime duration with mean lambda.
    """

    def __init__(self, config: dict):
        cp_cfg = config.get("changepoint", {})
        self.hazard_lambda = cp_cfg.get("hazard_lambda", 200)
        self.threshold = cp_cfg.get("threshold", 0.5)
        self.mu0 = cp_cfg.get("prior_mu", 0.0)
        self.kappa0 = cp_cfg.get("prior_kappa", 1.0)
        self.alpha0 = cp_cfg.get("prior_alpha", 1.0)
        self.beta0 = cp_cfg.get("prior_beta", 1.0)
        self.store_posterior = cp_cfg.get("store_posterior", False)
        self.min_run_length = cp_cfg.get("min_run_length", 5)

    def hazard(self, r: np.ndarray) -> np.ndarray:
        """Constant hazard function: H(r) = 1/lambda for all r."""
        return np.full_like(r, 1.0 / self.hazard_lambda, dtype=float)

    def detect(self, X: np.ndarray) -> BOCDResult:
        """
        Run BOCD on observation matrix X of shape (T,) or (T, d).

        Auto-scales priors to match data: sets prior_beta to the empirical
        variance of the first 20 observations so the predictive distribution
        is calibrated to the actual signal scale.

        Returns BOCDResult with changepoint probabilities and detections.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T, d = X.shape

        # Auto-scale prior beta to data variance for well-calibrated detection
        warmup = min(20, T)
        empirical_var = np.var(X[:warmup], axis=0).mean()
        if empirical_var > 0:
            beta0 = max(empirical_var, 1e-10)
        else:
            beta0 = self.beta0

        # Use multivariate predictor
        predictor = MultivariatePredictor(
            d, self.mu0, self.kappa0, self.alpha0, beta0
        )

        # Initialize sufficient statistics for run length 0
        # Each dimension: (mu, kappa, alpha, beta) arrays — one entry per active run length
        stats = [
            (
                np.array([self.mu0]),
                np.array([self.kappa0]),
                np.array([self.alpha0]),
                np.array([beta0]),
            )
            for _ in range(d)
        ]

        # Run-length probabilities: start with P(r_0 = 0) = 1
        run_length_probs = np.array([1.0])

        changepoint_prob = np.zeros(T)
        map_run_length = np.zeros(T, dtype=int)
        growth_prob = np.zeros(T)
        posterior_store = [] if self.store_posterior else None

        for t in range(T):
            x = X[t]
            n_rl = len(run_length_probs)

            # 1. Predictive probabilities under each run-length hypothesis
            pred_logp = predictor.predictive_logprob(x, stats)

            # 2. Hazard function
            r_vals = np.arange(n_rl, dtype=float)
            H = self.hazard(r_vals)

            # 3. Growth probabilities: P(r_t = r_{t-1}+1, x_{1:t})
            #    = P(r_{t-1}, x_{1:t-1}) * P(x_t | r_{t-1}) * (1 - H(r_{t-1}))
            pred_probs = np.exp(pred_logp)
            growth = run_length_probs * pred_probs * (1 - H)

            # 4. Changepoint probability: sum over all run lengths that reset
            #    P(r_t = 0, x_{1:t}) = sum_r P(r_{t-1}=r) * P(x_t | r) * H(r)
            cp = np.sum(run_length_probs * pred_probs * H)

            # 5. New run-length distribution: [cp, growth_0, growth_1, ...]
            new_rl_probs = np.empty(n_rl + 1)
            new_rl_probs[0] = cp
            new_rl_probs[1:] = growth

            # 6. Normalize
            evidence = new_rl_probs.sum()
            if evidence > 0:
                new_rl_probs /= evidence
            else:
                new_rl_probs = np.ones(n_rl + 1) / (n_rl + 1)

            # 7. Store results
            #    With constant hazard, P(r=0) = H always, so we compute a
            #    more informative "changepoint score" based on the posterior
            #    mass on short run lengths (r < threshold). When a changepoint
            #    occurs, mass shifts from long to short run lengths.
            map_rl = np.argmax(new_rl_probs)
            map_run_length[t] = map_rl

            # Changepoint score: mass on short run lengths (r <= min_run_length)
            short_cutoff = min(self.min_run_length + 1, len(new_rl_probs))
            short_mass = new_rl_probs[:short_cutoff].sum()
            # Normalize: subtract baseline (expected short mass under no-change)
            # Under steady state, mass on r<k is roughly k/lambda
            baseline = short_cutoff / self.hazard_lambda
            changepoint_prob[t] = np.clip(
                (short_mass - baseline) / max(1 - baseline, 1e-10), 0, 1
            )
            growth_prob[t] = 1.0 - changepoint_prob[t]

            if posterior_store is not None:
                posterior_store.append(new_rl_probs.copy())

            # 8. Update sufficient statistics
            #    - For run length 0 (new segment): reset to prior
            #    - For run lengths > 0: update existing stats with new observation
            updated_stats = predictor.update_sufficient_stats(x, stats)
            new_stats = []
            for dim in range(d):
                new_mu = np.empty(n_rl + 1)
                new_kappa = np.empty(n_rl + 1)
                new_alpha = np.empty(n_rl + 1)
                new_beta = np.empty(n_rl + 1)

                # Run length 0: reset to prior
                new_mu[0] = self.mu0
                new_kappa[0] = self.kappa0
                new_alpha[0] = self.alpha0
                new_beta[0] = beta0

                # Run lengths 1..n_rl: updated stats from previous
                new_mu[1:] = updated_stats[dim][0]
                new_kappa[1:] = updated_stats[dim][1]
                new_alpha[1:] = updated_stats[dim][2]
                new_beta[1:] = updated_stats[dim][3]

                new_stats.append((new_mu, new_kappa, new_alpha, new_beta))

            stats = new_stats
            run_length_probs = new_rl_probs

            # 9. Pruning: remove run lengths with negligible probability
            #    to keep memory O(expected_run_length) instead of O(T)
            if len(run_length_probs) > self.hazard_lambda * 3:
                keep = run_length_probs > 1e-10
                # Always keep the first few entries
                keep[:min(10, len(keep))] = True
                if keep.sum() < len(keep):
                    run_length_probs = run_length_probs[keep]
                    run_length_probs /= run_length_probs.sum()
                    stats = [
                        (mu[keep], kappa[keep], alpha[keep], beta[keep])
                        for mu, kappa, alpha, beta in stats
                    ]

        # Detect changepoints: bars where P(cp) > threshold and min_run_length met
        raw_cps = np.where(changepoint_prob > self.threshold)[0]
        # Filter out changepoints too close together
        if len(raw_cps) > 0 and self.min_run_length > 1:
            filtered = [raw_cps[0]]
            for cp in raw_cps[1:]:
                if cp - filtered[-1] >= self.min_run_length:
                    filtered.append(cp)
            changepoints = np.array(filtered)
        else:
            changepoints = raw_cps

        return BOCDResult(
            changepoint_prob=changepoint_prob,
            map_run_length=map_run_length,
            changepoints=changepoints,
            growth_prob=growth_prob,
            run_length_posterior=posterior_store,
        )

    def detect_on_features(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> BOCDResult:
        """
        Run BOCD on the engineered features from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            Must contain the feature columns (e.g., from compute_features).
        feature_cols : list[str], optional
            Features to use. Defaults to log_return only (most sensitive
            to regime changes). Using multiple features increases robustness
            but may dilute sensitivity.

        Returns
        -------
        BOCDResult
        """
        if feature_cols is None:
            feature_cols = ["log_return"]

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = df[feature_cols].values
        return self.detect(X)

    def changepoint_confirmation(
        self,
        result: BOCDResult,
        window: int = 5,
    ) -> np.ndarray:
        """
        Compute a smoothed changepoint confirmation signal.

        Rather than using the raw P(cp) which is noisy, this computes
        the max changepoint probability in a rolling window. This is
        useful as a confirmation signal: "a changepoint happened recently."

        Returns array of shape (T,) with values in [0, 1].
        """
        T = len(result.changepoint_prob)
        confirmed = np.zeros(T)
        for t in range(T):
            start = max(0, t - window + 1)
            confirmed[t] = np.max(result.changepoint_prob[start:t + 1])
        return confirmed

    def regime_stability_score(self, result: BOCDResult) -> np.ndarray:
        """
        Compute regime stability as normalized MAP run length.

        Longer run lengths → more stable regime → higher score.
        Score = 1 - exp(-map_run_length / hazard_lambda)

        Returns array of shape (T,) with values in [0, 1].
        """
        return 1.0 - np.exp(-result.map_run_length / self.hazard_lambda)
