"""
regime_forecast.py — Predictive Regime Forecasting Engine.

Turns the HMM from a backward-looking regime detector into a forward-looking
regime forecaster. Uses Chapman-Kolmogorov (P(t+n) = P(t) * A^n) to project
regime probability distributions across future horizons, then derives:

  - N-step-ahead regime probability curves
  - Regime half-life: expected bars until current regime ends
  - Expected return & volatility cones per forecast horizon
  - Transition countdown: P(regime change within next N bars)
  - Posterior momentum: velocity & acceleration of belief shifts
  - Anticipatory signals: front-run regime transitions before they happen
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class RegimeForecast:
    """Forward-looking regime probability forecast at a single point in time."""
    # Current state
    current_regime: str
    current_confidence: float
    current_posterior: np.ndarray  # (n_states,)

    # N-step-ahead probabilities: (max_horizon, n_states)
    forecast_probs: np.ndarray
    horizons: np.ndarray  # [1, 2, ..., max_horizon]

    # Regime half-life (expected bars until current regime probability < 50%)
    half_life: float

    # Expected return and volatility per horizon
    expected_returns: np.ndarray   # (max_horizon,)
    expected_volatility: np.ndarray  # (max_horizon,)

    # Return cone: (max_horizon, 5) for [5th, 25th, 50th, 75th, 95th] percentiles
    return_cone: np.ndarray

    # Transition countdown: P(regime != current within next N bars)
    transition_prob_curve: np.ndarray  # (max_horizon,)

    # Posterior momentum
    posterior_velocity: np.ndarray  # (n_states,) — rate of change
    posterior_acceleration: np.ndarray  # (n_states,) — acceleration

    # Most likely next regime and its probability at various horizons
    most_likely_next: dict  # {horizon: (regime_label, probability)}

    # Regime labels for reference
    labels: dict


@dataclass
class ForecastSeries:
    """Time series of regime forecasts across the dataset."""
    # Per-bar forecast summaries
    half_lives: np.ndarray              # (T,) bars until regime change
    transition_urgency: np.ndarray      # (T,) P(change within 5 bars)
    posterior_velocity_norm: np.ndarray  # (T,) magnitude of belief shift
    anticipated_regime: list             # (T,) predicted regime at horizon H
    anticipated_confidence: np.ndarray   # (T,) confidence in anticipated regime
    expected_return_5: np.ndarray        # (T,) expected return at 5-step horizon
    expected_return_10: np.ndarray       # (T,) expected return at 10-step horizon
    anticipatory_signals: np.ndarray     # (T,) {-1, 0, 1} front-running signals


class RegimeForecaster:
    """
    Predictive regime forecasting using Chapman-Kolmogorov dynamics.

    Given a fitted HMM's transition matrix A and emission parameters,
    projects the current posterior belief forward in time:

        P(S_{t+n} | O_{1:t}) = P(S_t | O_{1:t}) @ A^n

    This yields a full probability distribution over regimes at each
    future horizon, from which we derive expected returns, risk cones,
    regime half-lives, and anticipatory trading signals.
    """

    def __init__(self, config: dict, max_horizon: int = 20):
        self.max_horizon = max_horizon
        strat = config.get("strategy", {})
        self.hysteresis_bars = strat.get("hysteresis_bars", 3)
        self.min_confidence = strat.get("confirmations", {}).get("min_confidence", 0.6)
        self.config = config

    def _compute_transition_powers(self, transmat: np.ndarray) -> np.ndarray:
        """
        Precompute A^1, A^2, ..., A^max_horizon.

        Returns (max_horizon, n_states, n_states) array where
        result[h] = A^(h+1).
        """
        n = transmat.shape[0]
        powers = np.empty((self.max_horizon, n, n))
        powers[0] = transmat
        for h in range(1, self.max_horizon):
            powers[h] = powers[h - 1] @ transmat
        return powers

    def forecast_at_bar(
        self,
        posterior: np.ndarray,
        transmat: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
        labels: dict[int, str],
        current_state: int,
        covariance_type: str = "full",
        posterior_history: np.ndarray | None = None,
    ) -> RegimeForecast:
        """
        Generate a full regime forecast from a single bar's posterior.

        Parameters
        ----------
        posterior : (n_states,) current posterior probabilities
        transmat : (n_states, n_states) transition matrix
        means : (n_states, n_features) emission means
        covars : emission covariances
        labels : {state_id: regime_label}
        current_state : Viterbi-decoded current state
        covariance_type : covariance parameterization
        posterior_history : (lookback, n_states) recent posteriors for momentum
        """
        n_states = transmat.shape[0]
        n_features = means.shape[1]

        # Chapman-Kolmogorov: project posterior forward
        powers = self._compute_transition_powers(transmat)
        horizons = np.arange(1, self.max_horizon + 1)

        forecast_probs = np.empty((self.max_horizon, n_states))
        for h in range(self.max_horizon):
            forecast_probs[h] = posterior @ powers[h]

        # Regime half-life: bars until P(current_state) < 0.5
        current_label = labels.get(current_state, f"state_{current_state}")
        half_life = float(self.max_horizon)  # default if never drops below 0.5
        for h in range(self.max_horizon):
            if forecast_probs[h, current_state] < 0.5:
                # Interpolate between h and h-1
                if h == 0:
                    half_life = 0.5
                else:
                    p_prev = forecast_probs[h - 1, current_state]
                    p_curr = forecast_probs[h, current_state]
                    # Linear interpolation
                    frac = (p_prev - 0.5) / (p_prev - p_curr) if p_prev != p_curr else 0.5
                    half_life = float(h) + frac
                break

        # Expected return per horizon (weighted by regime probabilities)
        regime_returns = means[:, 0]  # log_return is feature 0
        expected_returns = forecast_probs @ regime_returns

        # Expected volatility per horizon
        regime_vols = np.empty(n_states)
        for s in range(n_states):
            if covariance_type == "full":
                regime_vols[s] = np.sqrt(covars[s][0, 0])
            elif covariance_type == "diag":
                regime_vols[s] = np.sqrt(covars[s][0])
            elif covariance_type == "spherical":
                regime_vols[s] = np.sqrt(covars[s])
            else:  # tied
                regime_vols[s] = np.sqrt(covars[0, 0])

        # Regime-probability-weighted volatility per horizon
        # Vol^2 = E[Vol_s^2] + E[mu_s^2] - E[mu_s]^2  (law of total variance)
        expected_volatility = np.empty(self.max_horizon)
        for h in range(self.max_horizon):
            p = forecast_probs[h]
            e_var = np.sum(p * regime_vols ** 2)
            e_mu2 = np.sum(p * regime_returns ** 2)
            e_mu_sq = (np.sum(p * regime_returns)) ** 2
            expected_volatility[h] = np.sqrt(e_var + e_mu2 - e_mu_sq)

        # Return cone: cumulative return percentiles
        # Use regime-mixture distribution at each horizon
        return_cone = self._compute_return_cone(
            forecast_probs, regime_returns, regime_vols
        )

        # Transition countdown: P(regime != current within next N bars)
        # P(change by step h) = 1 - P(still in current at step h)
        # P(still in current) uses only the diagonal element propagation
        transition_prob_curve = np.empty(self.max_horizon)
        for h in range(self.max_horizon):
            p_still_current = forecast_probs[h, current_state]
            transition_prob_curve[h] = 1.0 - p_still_current

        # Posterior momentum
        velocity = np.zeros(n_states)
        acceleration = np.zeros(n_states)
        if posterior_history is not None and len(posterior_history) >= 3:
            # Velocity: finite difference of last 2 posteriors
            velocity = posterior_history[-1] - posterior_history[-2]
            # Acceleration: second derivative
            v_prev = posterior_history[-2] - posterior_history[-3]
            acceleration = velocity - v_prev

        # Most likely next regime at key horizons
        key_horizons = [1, 3, 5, 10, min(20, self.max_horizon)]
        most_likely_next = {}
        for h in key_horizons:
            if h <= self.max_horizon:
                probs_h = forecast_probs[h - 1]
                best_state = int(np.argmax(probs_h))
                most_likely_next[h] = (
                    labels.get(best_state, f"state_{best_state}"),
                    float(probs_h[best_state]),
                )

        return RegimeForecast(
            current_regime=current_label,
            current_confidence=float(posterior[current_state]),
            current_posterior=posterior,
            forecast_probs=forecast_probs,
            horizons=horizons,
            half_life=half_life,
            expected_returns=expected_returns,
            expected_volatility=expected_volatility,
            return_cone=return_cone,
            transition_prob_curve=transition_prob_curve,
            posterior_velocity=velocity,
            posterior_acceleration=acceleration,
            most_likely_next=most_likely_next,
            labels=labels,
        )

    def _compute_return_cone(
        self,
        forecast_probs: np.ndarray,
        regime_returns: np.ndarray,
        regime_vols: np.ndarray,
        n_samples: int = 5000,
    ) -> np.ndarray:
        """
        Monte Carlo return cone from regime-mixture distributions.

        At each horizon h, the return distribution is a Gaussian mixture
        with weights = forecast_probs[h], means = regime_returns,
        stds = regime_vols. We sample from this mixture and compute
        cumulative return percentiles.

        Returns (max_horizon, 5) for [5th, 25th, 50th, 75th, 95th].
        """
        rng = np.random.default_rng(42)
        n_states = len(regime_returns)
        cone = np.empty((self.max_horizon, 5))

        # Sample cumulative paths
        cumulative = np.zeros(n_samples)

        for h in range(self.max_horizon):
            p = forecast_probs[h]
            # Sample regime for each path at this horizon
            regimes = rng.choice(n_states, size=n_samples, p=p)
            # Sample return from regime-conditional Gaussian
            step_returns = np.empty(n_samples)
            for s in range(n_states):
                mask = regimes == s
                count = mask.sum()
                if count > 0:
                    step_returns[mask] = rng.normal(
                        regime_returns[s], regime_vols[s], size=count
                    )
            cumulative += step_returns
            cone[h] = np.percentile(cumulative, [5, 25, 50, 75, 95])

        return cone

    def forecast_series(
        self,
        posteriors: np.ndarray,
        states: np.ndarray,
        transmat: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
        labels: dict[int, str],
        covariance_type: str = "full",
        anticipation_horizon: int = 5,
    ) -> ForecastSeries:
        """
        Compute forecast metrics across the full time series.

        This is the bulk computation that powers the dashboard tab,
        producing per-bar metrics like half-life, transition urgency,
        posterior momentum, and anticipatory signals.

        Parameters
        ----------
        posteriors : (T, n_states) posterior probabilities per bar
        states : (T,) Viterbi-decoded states
        transmat : (n_states, n_states) transition matrix
        means : (n_states, n_features) emission means
        covars : emission covariances
        labels : {state_id: regime_label}
        covariance_type : covariance parameterization
        anticipation_horizon : bars ahead for anticipatory signal
        """
        T = len(states)
        n_states = transmat.shape[0]
        regime_returns = means[:, 0]

        # Precompute transition matrix powers
        powers = self._compute_transition_powers(transmat)

        # Extract per-regime volatilities
        regime_vols = np.empty(n_states)
        for s in range(n_states):
            if covariance_type == "full":
                regime_vols[s] = np.sqrt(covars[s][0, 0])
            elif covariance_type == "diag":
                regime_vols[s] = np.sqrt(covars[s][0])
            elif covariance_type == "spherical":
                regime_vols[s] = np.sqrt(covars[s])
            else:
                regime_vols[s] = np.sqrt(covars[0, 0])

        # Output arrays
        half_lives = np.full(T, float(self.max_horizon))
        transition_urgency = np.zeros(T)
        posterior_velocity_norm = np.zeros(T)
        anticipated_regime = ["unknown"] * T
        anticipated_confidence = np.zeros(T)
        expected_return_5 = np.zeros(T)
        expected_return_10 = np.zeros(T)
        anticipatory_signals = np.zeros(T, dtype=int)

        # Determine bullish/bearish states for signal generation
        bull_states = {s for s, l in labels.items() if l in ("bull", "bull_run")}
        bear_states = {s for s, l in labels.items() if l in ("bear", "crash")}

        for t in range(T):
            posterior = posteriors[t]
            current_state = states[t]

            # N-step forecast
            forecast_probs = np.empty((self.max_horizon, n_states))
            for h in range(self.max_horizon):
                forecast_probs[h] = posterior @ powers[h]

            # Half-life
            for h in range(self.max_horizon):
                if forecast_probs[h, current_state] < 0.5:
                    if h == 0:
                        half_lives[t] = 0.5
                    else:
                        p_prev = forecast_probs[h - 1, current_state]
                        p_curr = forecast_probs[h, current_state]
                        denom = p_prev - p_curr
                        frac = (p_prev - 0.5) / denom if denom != 0 else 0.5
                        half_lives[t] = float(h) + frac
                    break

            # Transition urgency: P(change within 5 bars)
            urgency_h = min(5, self.max_horizon) - 1
            transition_urgency[t] = 1.0 - forecast_probs[urgency_h, current_state]

            # Posterior velocity (norm)
            if t >= 2:
                vel = posteriors[t] - posteriors[t - 1]
                posterior_velocity_norm[t] = np.linalg.norm(vel)

            # Anticipated regime at horizon
            ah = min(anticipation_horizon, self.max_horizon) - 1
            probs_ahead = forecast_probs[ah]
            best_future_state = int(np.argmax(probs_ahead))
            anticipated_regime[t] = labels.get(best_future_state, f"state_{best_future_state}")
            anticipated_confidence[t] = float(probs_ahead[best_future_state])

            # Expected returns at horizons 5 and 10
            h5 = min(5, self.max_horizon) - 1
            h10 = min(10, self.max_horizon) - 1
            expected_return_5[t] = float(forecast_probs[h5] @ regime_returns)
            expected_return_10[t] = float(forecast_probs[h10] @ regime_returns)

            # Anticipatory signal: trade based on where we're GOING, not where we ARE
            # Signal long if anticipated regime is bullish with sufficient confidence
            # Signal short if anticipated regime is bearish with sufficient confidence
            if anticipated_confidence[t] >= self.min_confidence:
                if best_future_state in bull_states:
                    anticipatory_signals[t] = 1
                elif best_future_state in bear_states:
                    anticipatory_signals[t] = -1

        return ForecastSeries(
            half_lives=half_lives,
            transition_urgency=transition_urgency,
            posterior_velocity_norm=posterior_velocity_norm,
            anticipated_regime=anticipated_regime,
            anticipated_confidence=anticipated_confidence,
            expected_return_5=expected_return_5,
            expected_return_10=expected_return_10,
            anticipatory_signals=anticipatory_signals,
        )

    def regime_survival_curve(
        self,
        transmat: np.ndarray,
        state: int,
        max_steps: int | None = None,
    ) -> np.ndarray:
        """
        Compute the survival function for a regime: P(still in state s at step h).

        This is simply the (state, state) element of A^h for each h.
        The survival curve decays geometrically for a single-state Markov chain
        but can have more complex dynamics when there are absorbing-like states.

        Returns (max_steps,) array of survival probabilities.
        """
        max_steps = max_steps or self.max_horizon
        survival = np.empty(max_steps)
        power = np.eye(transmat.shape[0])
        for h in range(max_steps):
            power = power @ transmat
            survival[h] = power[state, state]
        return survival

    def regime_absorption_times(
        self,
        transmat: np.ndarray,
        labels: dict[int, str],
    ) -> pd.DataFrame:
        """
        Compute expected first-passage times between all regime pairs.

        The mean first-passage time from state i to state j is derived from
        the fundamental matrix of the Markov chain. This tells you, on average,
        how many bars it takes to transition from one regime to another.

        Returns DataFrame with from-regime rows and to-regime columns.
        """
        n = transmat.shape[0]

        # Stationary distribution
        eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = np.abs(pi)
        pi /= pi.sum()

        # Mean first passage times using the fundamental matrix approach
        # M_ij = (Z_jj - Z_ij) / pi_j  where Z = (I - A + Pi)^{-1}
        Pi_mat = np.tile(pi, (n, 1))  # each row is pi
        Z = np.linalg.inv(np.eye(n) - transmat + Pi_mat)

        mfpt = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    mfpt[i, j] = 0.0
                else:
                    mfpt[i, j] = (Z[j, j] - Z[i, j]) / pi[j] if pi[j] > 1e-10 else np.inf

        regime_labels = [labels.get(s, f"state_{s}") for s in range(n)]
        return pd.DataFrame(mfpt, index=regime_labels, columns=regime_labels)
