"""
forecast.py — Analytical Forecast Engine.

Closed-form, posterior-conditioned forward projection of HMM regimes
and returns. Complements the Monte Carlo engine, which samples
*unconditionally* from the stationary distribution: this module
propagates the model's *current belief* forward in time using the
Chapman-Kolmogorov equation and the law of total expectation / variance.

Given the current filtered posterior p_T and the fitted transition
matrix A, it computes (exactly, no sampling):

    p_{T+k} = p_T @ A^k                (regime trajectory)

    E[r_{T+k}]   = sum_s p_{T+k,s} * mu_s            (law of total exp.)
    Var[r_{T+k}] = sum_s p_{T+k,s} * (sigma_s^2 + mu_s^2)
                   - E[r_{T+k}]^2                     (law of total var.)

and derived quantities:

  - cumulative return cone (mean + variance over horizon)
  - expected time spent in each regime
  - expected first passage time matrix (via fundamental matrix of the
    absorbing Markov chain)
  - most-likely regime sequence by forward DP
  - Bayesian-decision-theoretic action selector that maximizes expected
    log-wealth over the forecast distribution

The computations are all O(n_states^3 * horizon) and deterministic,
so they are safe to run interactively on every dashboard refresh.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ─── Result container ──────────────────────────────────────────────────────────


@dataclass
class ForecastResult:
    """Container for a single ForecastEngine.run() call."""

    horizon: int
    n_states: int

    # (horizon + 1, n_states) regime probability trajectory, row 0 = current
    posterior_trajectory: np.ndarray

    # (horizon,) per-step expected log returns and variances
    expected_return_path: np.ndarray
    variance_path: np.ndarray

    # (horizon,) cumulative mean and variance (assuming conditional
    # independence across steps — see docstring on cumulative_return_stats)
    cumulative_mean: np.ndarray
    cumulative_variance: np.ndarray

    # ±1σ and ±2σ bands on cumulative log return
    cone_lower_1: np.ndarray
    cone_upper_1: np.ndarray
    cone_lower_2: np.ndarray
    cone_upper_2: np.ndarray

    # (n_states,) expected bars spent in each regime over the horizon
    expected_time_in_regime: np.ndarray

    # (n_states, n_states) expected first passage time E[T_{i->j}]
    # Diagonal is filled with expected return time E[T_{i->i}].
    first_passage_time: np.ndarray

    # (horizon,) most likely state sequence starting from the current posterior
    most_likely_path: np.ndarray

    # Expected-utility signal optimizer
    eu_best_action: int              # -1, 0, or +1
    eu_values: dict                  # {-1: float, 0: float, 1: float}
    eu_recommended_size: float       # in [0, max_size]

    # Echo of the current state (for plotting convenience)
    current_posterior: np.ndarray
    labels: dict = field(default_factory=dict)


# ─── Core engine ───────────────────────────────────────────────────────────────


class ForecastEngine:
    """
    Analytical, closed-form forecast engine for a fitted Gaussian HMM.

    All methods are pure functions of the fitted model parameters and the
    current posterior, so they can be called repeatedly without side effects.
    """

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("forecast", {})
        self.horizon = int(cfg.get("horizon", 30))
        self.risk_aversion = float(cfg.get("risk_aversion", 1.0))
        self.max_size = float(cfg.get("max_size", 1.0))
        self.min_transition_prob = float(cfg.get("min_transition_prob", 1e-12))

    # ── Regime trajectory ────────────────────────────────────────────────────

    def project_posterior(
        self,
        current_posterior: np.ndarray,
        transmat: np.ndarray,
        horizon: int | None = None,
    ) -> np.ndarray:
        """
        Propagate the current posterior forward using Chapman-Kolmogorov.

        Returns an array of shape (horizon + 1, n_states) where row 0 is the
        current posterior and row k is p_T @ A^k.
        """
        horizon = horizon if horizon is not None else self.horizon
        p0 = np.asarray(current_posterior, dtype=float).flatten()
        if p0.sum() <= 0:
            raise ValueError("current_posterior must sum to a positive value")
        p0 = p0 / p0.sum()

        n = transmat.shape[0]
        if p0.shape[0] != n:
            raise ValueError(
                f"posterior has {p0.shape[0]} states but transmat has {n}"
            )

        traj = np.empty((horizon + 1, n), dtype=float)
        traj[0] = p0
        for k in range(1, horizon + 1):
            traj[k] = traj[k - 1] @ transmat

        # Numerical cleanup: clip and renormalize each row
        traj = np.clip(traj, 0.0, None)
        row_sums = traj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        traj = traj / row_sums
        return traj

    # ── Return path moments (law of total exp. / var. for a mixture) ────────

    @staticmethod
    def _feature_variance(
        covars: np.ndarray,
        state: int,
        feature_idx: int,
        covariance_type: str,
        n_features: int,
    ) -> float:
        """Extract Var(X[feature_idx] | state) from an hmmlearn covars_ array."""
        if covariance_type == "full":
            return float(covars[state][feature_idx, feature_idx])
        if covariance_type == "diag":
            return float(covars[state][feature_idx])
        if covariance_type == "spherical":
            return float(covars[state])
        if covariance_type == "tied":
            return float(covars[feature_idx, feature_idx])
        raise ValueError(f"Unknown covariance_type: {covariance_type}")

    def expected_return_path(
        self,
        posterior_trajectory: np.ndarray,
        means: np.ndarray,
        feature_idx: int = 0,
    ) -> np.ndarray:
        """
        Per-step expected return using the law of total expectation:

            E[r_{T+k}] = sum_s P(s_{T+k} = s) * mu_s[feature_idx]

        Row 0 of the trajectory is the current posterior (k=0), so the
        returned array starts at k=1 and has length horizon.
        """
        per_state_mean = means[:, feature_idx]
        # skip the k=0 row — that is "now", not the next bar
        return posterior_trajectory[1:] @ per_state_mean

    def variance_path(
        self,
        posterior_trajectory: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
        covariance_type: str = "full",
        feature_idx: int = 0,
    ) -> np.ndarray:
        """
        Per-step return variance using the law of total variance for a
        Gaussian mixture of regimes:

            E[r^2 | s] = sigma_s^2 + mu_s^2
            E[r^2]    = sum_s p_s * (sigma_s^2 + mu_s^2)
            Var[r]    = E[r^2] - E[r]^2

        Returns an array of length (horizon,) — excludes row 0 (now).
        """
        n_states, n_features = means.shape
        mu = means[:, feature_idx]
        sigma2 = np.array([
            self._feature_variance(covars, s, feature_idx, covariance_type, n_features)
            for s in range(n_states)
        ])

        p = posterior_trajectory[1:]                    # (H, n_states)
        e_r = p @ mu                                    # (H,)
        e_r2 = p @ (sigma2 + mu * mu)                   # (H,)
        var = e_r2 - e_r * e_r
        return np.clip(var, 0.0, None)

    def cumulative_return_stats(
        self,
        expected_return_path: np.ndarray,
        variance_path: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Cumulative log return over [T, T+k]:

            R_k = sum_{h=1..k} r_{T+h}

        Under the (standard) simplification that residual shocks are
        conditionally independent across bars given regimes, the cumulative
        mean is additive and the cumulative variance is additive:

            E[R_k]   = sum_{h=1..k} E[r_h]
            Var[R_k] = sum_{h=1..k} Var[r_h]

        The regime-driven drift component is already tracked by the Chapman-
        Kolmogorov trajectory, so this gives a tight analytic cone without
        having to compute full joint state marginals.
        """
        cum_mean = np.cumsum(expected_return_path)
        cum_var = np.cumsum(variance_path)
        return cum_mean, cum_var

    # ── Time-in-regime ───────────────────────────────────────────────────────

    def expected_time_in_regime(
        self,
        posterior_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        Expected number of bars the chain spends in each regime over the
        horizon. This is the sum of posterior probabilities across steps 1..H
        (excluding 'now'), which by linearity of expectation equals the
        expected count of visits.
        """
        return posterior_trajectory[1:].sum(axis=0)

    # ── First passage times ─────────────────────────────────────────────────

    def first_passage_time_matrix(
        self,
        transmat: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the full matrix of expected first passage times E[T_{i->j}]
        for every (source, target) pair.

        For target j, make j absorbing and solve

            (I - Q) m = 1

        for the expected hitting times m from each non-absorbing state, where
        Q is the sub-transition matrix over non-absorbing states. The diagonal
        E[T_{j->j}] is the expected return time to j, which for an ergodic
        chain equals 1 / pi_j where pi is the stationary distribution.

        If the chain is not ergodic (or the linear system is singular for a
        particular target) the affected entries fall back to +inf rather than
        raising, so the caller can still render something sensible.
        """
        A = np.asarray(transmat, dtype=float)
        n = A.shape[0]
        fpt = np.full((n, n), np.inf, dtype=float)

        # Diagonal: expected return time via the stationary distribution.
        try:
            eigvals, eigvecs = np.linalg.eig(A.T)
            idx = int(np.argmin(np.abs(eigvals - 1.0)))
            pi = np.real(eigvecs[:, idx])
            pi = np.abs(pi)
            total = pi.sum()
            if total > 0:
                pi = pi / total
                for j in range(n):
                    if pi[j] > 0:
                        fpt[j, j] = 1.0 / pi[j]
        except np.linalg.LinAlgError:
            pass

        # Off-diagonal: solve (I - Q) m = 1 with j absorbed.
        for j in range(n):
            keep = [i for i in range(n) if i != j]
            Q = A[np.ix_(keep, keep)]
            I = np.eye(len(keep))
            rhs = np.ones(len(keep))
            try:
                m = np.linalg.solve(I - Q, rhs)
                for k, i in enumerate(keep):
                    fpt[i, j] = float(m[k]) if np.isfinite(m[k]) and m[k] > 0 else np.inf
            except np.linalg.LinAlgError:
                continue

        return fpt

    def expected_first_passage_from_posterior(
        self,
        current_posterior: np.ndarray,
        first_passage: np.ndarray,
    ) -> np.ndarray:
        """
        Marginalise the first-passage matrix against the current posterior.

        Returns a vector (n_states,) where entry j is the expected number of
        bars until the chain first enters state j, averaged over the current
        belief about the starting state. Infinite entries are preserved.
        """
        p = np.asarray(current_posterior, dtype=float).flatten()
        p = p / p.sum() if p.sum() > 0 else p
        n = first_passage.shape[0]
        out = np.empty(n)
        for j in range(n):
            col = first_passage[:, j]
            if np.any(np.isinf(col) & (p > 0)):
                # If any reachable-with-positive-probability source has infinite
                # first-passage, the expected value is also infinite.
                if np.any(np.isinf(col[p > 0])):
                    out[j] = np.inf
                    continue
            out[j] = float(np.sum(p * col))
        return out

    # ── Most-likely regime path ─────────────────────────────────────────────

    def most_likely_path(
        self,
        current_posterior: np.ndarray,
        transmat: np.ndarray,
        horizon: int | None = None,
    ) -> np.ndarray:
        """
        Forward dynamic program that returns the single most likely regime
        sequence over the next `horizon` bars given the current belief.

        This is NOT the Viterbi decoding of observed data — there are no
        observations in the future — it is the mode of the joint distribution
        over future state sequences implied by p_T and A.

        delta_0(s)       = p_T[s]
        delta_h(s)       = max_s' delta_{h-1}(s') * A[s', s]
        argmax backtrack = most likely sequence
        """
        horizon = horizon if horizon is not None else self.horizon
        A = np.asarray(transmat, dtype=float)
        n = A.shape[0]

        # Work in log space to avoid underflow over long horizons.
        logA = np.log(np.clip(A, self.min_transition_prob, None))
        log_p0 = np.log(np.clip(current_posterior, self.min_transition_prob, None))

        delta = np.empty((horizon + 1, n))
        psi = np.empty((horizon + 1, n), dtype=int)
        delta[0] = log_p0
        psi[0] = np.arange(n)

        for h in range(1, horizon + 1):
            # delta_h[s] = max_s' (delta_{h-1}[s'] + logA[s', s])
            scores = delta[h - 1][:, None] + logA      # (n, n)
            psi[h] = np.argmax(scores, axis=0)
            delta[h] = scores[psi[h], np.arange(n)]

        # Backtrack from the most likely terminal state.
        path = np.empty(horizon, dtype=int)
        s = int(np.argmax(delta[horizon]))
        for h in range(horizon, 0, -1):
            path[h - 1] = s
            s = int(psi[h][s])
        return path

    # ── Expected-utility action selector ─────────────────────────────────────

    def expected_utility_signal(
        self,
        cumulative_mean: np.ndarray,
        cumulative_variance: np.ndarray,
        actions: tuple[int, ...] = (-1, 0, 1),
    ) -> tuple[int, dict, float]:
        """
        Pick the action that maximizes expected log-wealth over the horizon
        under a Gaussian approximation to cumulative log returns.

        For an action a ∈ {-1, 0, +1} and (approximately Gaussian) horizon
        return R_H with mean mu and variance v, the expected log wealth is

            E[log(1 + a * R_H)]  ≈  a*mu - 0.5 * lambda * v  (for |a*mu| << 1)

        where lambda is a risk-aversion coefficient. We use the final-horizon
        cumulative moments as a proxy for long-term expected utility, which
        gives the familiar mean–variance form.

        The chosen action is the argmax. A recommended *size* is returned as

            |mu| / (lambda * v),    clipped to [0, max_size]

        which is the classical Merton/Kelly optimum for a Gaussian risky
        asset — zero when variance dominates drift, max_size when the signal
        is very strong.
        """
        if len(cumulative_mean) == 0:
            return 0, {a: 0.0 for a in actions}, 0.0

        mu = float(cumulative_mean[-1])
        v = float(max(cumulative_variance[-1], 1e-12))
        lam = self.risk_aversion

        eu = {a: a * mu - 0.5 * lam * (a * a) * v for a in actions}
        best = max(eu, key=eu.get)

        # Kelly-style size from the horizon moments.
        raw_size = abs(mu) / (lam * v) if v > 0 else 0.0
        size = float(min(max(raw_size, 0.0), self.max_size))
        if best == 0:
            size = 0.0

        return int(best), eu, size

    # ── Orchestration ────────────────────────────────────────────────────────

    def run(
        self,
        current_posterior: np.ndarray,
        transmat: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
        labels: dict[int, str] | None = None,
        covariance_type: str = "full",
        feature_idx: int = 0,
        horizon: int | None = None,
    ) -> ForecastResult:
        """
        Run the full analytical forecast pipeline and return a ForecastResult.

        Parameters
        ----------
        current_posterior : np.ndarray
            Shape (n_states,). Typically the last row of
            ``detector.decode(X)[1]`` — the filtered posterior at 'now'.
        transmat : np.ndarray
            Fitted transition matrix, shape (n_states, n_states).
        means : np.ndarray
            Emission means, shape (n_states, n_features).
        covars : np.ndarray
            Emission covariances in hmmlearn's layout for the given
            ``covariance_type``.
        labels : dict, optional
            Mapping from state id to semantic label (bull, bear, ...).
        covariance_type : str
            One of 'full', 'diag', 'spherical', 'tied'.
        feature_idx : int
            Which emission feature is the log return (default 0).
        horizon : int, optional
            Override the engine's default horizon for this run.
        """
        horizon = int(horizon if horizon is not None else self.horizon)
        labels = labels or {}

        trajectory = self.project_posterior(current_posterior, transmat, horizon)
        e_path = self.expected_return_path(trajectory, means, feature_idx)
        v_path = self.variance_path(trajectory, means, covars, covariance_type, feature_idx)
        cum_mean, cum_var = self.cumulative_return_stats(e_path, v_path)
        cum_std = np.sqrt(cum_var)

        time_in = self.expected_time_in_regime(trajectory)
        fpt = self.first_passage_time_matrix(transmat)
        path = self.most_likely_path(current_posterior, transmat, horizon)
        best_action, eu_values, size = self.expected_utility_signal(cum_mean, cum_var)

        return ForecastResult(
            horizon=horizon,
            n_states=transmat.shape[0],
            posterior_trajectory=trajectory,
            expected_return_path=e_path,
            variance_path=v_path,
            cumulative_mean=cum_mean,
            cumulative_variance=cum_var,
            cone_lower_1=cum_mean - cum_std,
            cone_upper_1=cum_mean + cum_std,
            cone_lower_2=cum_mean - 2.0 * cum_std,
            cone_upper_2=cum_mean + 2.0 * cum_std,
            expected_time_in_regime=time_in,
            first_passage_time=fpt,
            most_likely_path=path,
            eu_best_action=best_action,
            eu_values=eu_values,
            eu_recommended_size=size,
            current_posterior=np.asarray(current_posterior, dtype=float),
            labels=labels,
        )
