"""
changepoint.py — Bayesian Online Changepoint Detection (BOCPD).

Implements Adams & MacKay (2007): maintains a posterior distribution
over "run lengths" (bars since last changepoint) using recursive
message passing with conjugate Normal-Inverse-Gamma priors.

At each bar, produces:
  - changepoint_prob: P(r_t < recency_horizon) — probability that a
    regime change occurred recently (within the last few bars)
  - expected_run_length: E[r_t] — expected bars since last changepoint
  - regime_stability: derived from expected_run_length relative to
    the hazard_lambda, usable as a confirmation gate

Complements HMM: HMM tells you WHAT regime you're in;
BOCPD tells you WHEN it's changing.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class BOCPDResult:
    """Output of Bayesian Online Changepoint Detection."""
    changepoint_prob: np.ndarray      # P(recent changepoint) at each bar
    regime_stability: np.ndarray      # 1 - changepoint_prob
    expected_run_length: np.ndarray   # E[run_length] at each bar
    map_run_length: np.ndarray        # argmax run_length at each bar
    growth_probs: list[np.ndarray]    # full run-length posterior per bar (for viz)
    changepoint_indices: np.ndarray   # bars where changepoint_prob > threshold


class BayesianChangepoint:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

    Uses a conjugate Normal-Inverse-Gamma (NIG) model as the
    underlying predictive model (UPM), with a constant hazard
    rate for the changepoint prior.

    The hazard function H(tau) = 1/lambda represents the prior
    belief that a changepoint occurs every ~lambda bars on average.

    Detection signal: With constant hazard, the normalized P(r=0)
    is always 1/lambda. The informative signal is the cumulative
    mass at short run lengths P(r < k), which spikes when the
    posterior believes a changepoint occurred recently. This is
    derived from the expected run length relative to hazard_lambda.
    """

    def __init__(self, config: dict):
        cp_cfg = config.get("changepoint", {})
        self.hazard_lambda = cp_cfg.get("hazard_lambda", 200)
        self.mu0 = cp_cfg.get("prior_mu", 0.0)
        self.kappa0 = cp_cfg.get("prior_kappa", 0.1)
        self.alpha0 = cp_cfg.get("prior_alpha", 1.0)
        self.beta0 = cp_cfg.get("prior_beta", 0.01)
        self.threshold = cp_cfg.get("threshold", 0.3)
        self.feature_index = cp_cfg.get("feature_index", 0)
        self.recency_horizon = cp_cfg.get("recency_horizon", 10)

    def _hazard(self, _r: np.ndarray) -> np.ndarray:
        """Constant hazard rate: H(r) = 1/lambda for all r."""
        return np.full_like(_r, 1.0 / self.hazard_lambda, dtype=float)

    def _student_t_pdf(
        self,
        x: float,
        mu: np.ndarray,
        var: np.ndarray,
        nu: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the Student-t predictive density.

        Under the NIG conjugate model, the predictive distribution
        for the next observation is Student-t with:
          location = mu_n
          scale^2 = beta_n * (kappa_n + 1) / (alpha_n * kappa_n)
          degrees of freedom = 2 * alpha_n
        """
        nu = np.maximum(nu, 1e-8)
        var = np.maximum(var, 1e-16)

        z = (x - mu) ** 2 / var

        log_pdf = (
            _log_gamma((nu + 1) / 2)
            - _log_gamma(nu / 2)
            - 0.5 * np.log(nu * np.pi * var)
            - ((nu + 1) / 2) * np.log(1 + z / nu)
        )

        return np.exp(log_pdf)

    def detect(self, X: np.ndarray) -> BOCPDResult:
        """
        Run BOCPD on a 1D or multi-feature time series.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (T, D) or 1D array of shape (T,).
            Uses self.feature_index to select the monitored feature if 2D.

        Returns
        -------
        BOCPDResult with changepoint probabilities, stability scores,
        run-length statistics, and detected changepoint indices.
        """
        if X.ndim == 2:
            x = X[:, self.feature_index].copy()
        else:
            x = X.copy()

        T = len(x)
        k = min(self.recency_horizon, T)

        # Initialize sufficient statistics arrays
        mu = np.array([self.mu0])
        kappa = np.array([self.kappa0])
        alpha = np.array([self.alpha0])
        beta = np.array([self.beta0])

        # Run-length probabilities: start with P(r_0 = 0) = 1
        R = np.zeros(T + 1)
        R[0] = 1.0

        # Output arrays
        short_run_mass = np.zeros(T)  # P(r < k) — "recent changepoint" signal
        expected_run_length = np.zeros(T)
        map_run_length = np.zeros(T, dtype=int)
        growth_probs = []

        for t in range(T):
            n_rl = t + 1

            # 1. Evaluate predictive probability P(x_t | r_t) for each run length
            nu = 2 * alpha[:n_rl]
            pred_var = beta[:n_rl] * (kappa[:n_rl] + 1) / (alpha[:n_rl] * kappa[:n_rl])
            pred_probs = self._student_t_pdf(x[t], mu[:n_rl], pred_var, nu)

            # 2. Growth probabilities: P(r_t = r_{t-1}+1, x_{1:t})
            H = self._hazard(np.arange(n_rl, dtype=float))
            growth = R[:n_rl] * pred_probs * (1 - H)

            # 3. Changepoint probability: P(r_t = 0, x_{1:t})
            cp = np.sum(R[:n_rl] * pred_probs * H)

            # 4. Assemble new run-length distribution
            new_R = np.zeros(n_rl + 1)
            new_R[0] = cp
            new_R[1:n_rl + 1] = growth

            # 5. Normalize (evidence)
            evidence = new_R.sum()
            if evidence > 0:
                new_R /= evidence

            # Store results
            R = np.zeros(T + 1)
            R[:n_rl + 1] = new_R

            # Cumulative mass at short run lengths: P(r < k)
            rl_limit = min(k, n_rl + 1)
            short_run_mass[t] = np.sum(new_R[:rl_limit])

            expected_run_length[t] = np.sum(np.arange(n_rl + 1) * new_R[:n_rl + 1])
            map_run_length[t] = np.argmax(new_R[:n_rl + 1])
            growth_probs.append(new_R[:n_rl + 1].copy())

            # 6. Update sufficient statistics
            new_mu = np.zeros(n_rl + 1)
            new_kappa = np.zeros(n_rl + 1)
            new_alpha = np.zeros(n_rl + 1)
            new_beta = np.zeros(n_rl + 1)

            # Run length 0: reset to prior
            new_mu[0] = self.mu0
            new_kappa[0] = self.kappa0
            new_alpha[0] = self.alpha0
            new_beta[0] = self.beta0

            # Run lengths 1..n_rl: Bayesian update of NIG parameters
            kn = kappa[:n_rl]
            mn = mu[:n_rl]
            an = alpha[:n_rl]
            bn = beta[:n_rl]

            new_kappa[1:n_rl + 1] = kn + 1
            new_mu[1:n_rl + 1] = (kn * mn + x[t]) / (kn + 1)
            new_alpha[1:n_rl + 1] = an + 0.5
            new_beta[1:n_rl + 1] = bn + kn * (x[t] - mn) ** 2 / (2 * (kn + 1))

            # Pad for next iteration
            mu = np.zeros(T + 1)
            kappa = np.zeros(T + 1)
            alpha = np.zeros(T + 1)
            beta = np.zeros(T + 1)
            mu[:n_rl + 1] = new_mu
            kappa[:n_rl + 1] = new_kappa
            alpha[:n_rl + 1] = new_alpha
            beta[:n_rl + 1] = new_beta

        # Derive changepoint probability from short-run-length mass.
        # Baseline mass under no-changepoint steady state is ~k/lambda.
        # Normalize so that baseline ≈ 0 and full mass at short runs ≈ 1.
        baseline = min(k / self.hazard_lambda, 0.5)
        changepoint_prob = np.clip(
            (short_run_mass - baseline) / (1.0 - baseline), 0.0, 1.0
        )

        regime_stability = 1.0 - changepoint_prob
        changepoint_indices = np.where(changepoint_prob > self.threshold)[0]

        return BOCPDResult(
            changepoint_prob=changepoint_prob,
            regime_stability=regime_stability,
            expected_run_length=expected_run_length,
            map_run_length=map_run_length,
            growth_probs=growth_probs,
            changepoint_indices=changepoint_indices,
        )

    def merge_with_hmm_confidence(
        self,
        hmm_confidence: np.ndarray,
        bocpd_result: BOCPDResult,
        blend_weight: float = 0.3,
    ) -> np.ndarray:
        """
        Fuse HMM confidence with BOCPD regime stability.

        During stable periods (low changepoint prob), HMM confidence passes
        through largely unchanged. During transitions (high changepoint prob),
        the fused confidence drops — signaling the strategy to reduce exposure.

        fused = hmm_confidence * (1 - blend_weight * changepoint_prob)

        Parameters
        ----------
        hmm_confidence : np.ndarray
            Entropy-based confidence from HMM posteriors.
        bocpd_result : BOCPDResult
            Output from self.detect().
        blend_weight : float
            How much changepoint probability dampens confidence (0-1).

        Returns
        -------
        np.ndarray
            Fused confidence scores.
        """
        dampening = 1.0 - blend_weight * bocpd_result.changepoint_prob
        return hmm_confidence * np.clip(dampening, 0.0, 1.0)


def _log_gamma(x: np.ndarray) -> np.ndarray:
    """Log-gamma function, vectorized."""
    from scipy.special import gammaln
    return gammaln(x)
