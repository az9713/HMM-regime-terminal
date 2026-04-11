"""
regime_forecast.py — Forward-looking regime prediction & fragility engine.

Transforms the HMM from a retrospective regime-detector into a predictive
engine. Provides closed-form (non-Monte-Carlo) forecasts of:

  • N-step-ahead regime probability distributions (Markov chain evolution).
  • Per-regime survival curves: P(still in state i at step h | now in i).
  • First-passage distributions: P(first enter state j at step h).
  • Hitting probabilities: P(ever visit state j within horizon).
  • Regime-change probability over an arbitrary horizon.
  • Expected cumulative return and return-variance using the law of total
    variance over the forecast path.
  • Optimal holding horizon — the horizon h* that maximises forecast Sharpe.
  • Regime Fragility Index — novel composite metric in [0, 1] combining
    posterior gap, entropy gradient, KL divergence from stationary
    distribution, and Markov regime-change probability.
  • Fragility-adjusted position sizing.

All computations are analytical (matrix powers / spectral decomposition),
complementing the stochastic Monte Carlo module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────
# Markov chain primitives
# ──────────────────────────────────────────────────────────────────────────

def stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """
    Stationary distribution π of a row-stochastic matrix.

    Solves πP = π with Σπ = 1 by taking the left-eigenvector corresponding
    to the unit eigenvalue.
    """
    P = np.asarray(transmat, dtype=float)
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi = np.real(eigvecs[:, idx])
    total = pi.sum()
    if abs(total) < EPS:
        # Fallback: uniform (degenerate matrix).
        return np.full(P.shape[0], 1.0 / P.shape[0])
    pi = pi / total
    return np.clip(pi, 0.0, 1.0)


def n_step_distribution(
    transmat: np.ndarray,
    initial_belief: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Forward-evolve a belief distribution under the HMM transition matrix.

    Returns an array of shape ``(horizon + 1, n_states)`` where row ``h``
    holds ``P(state = i at step h)``. Row 0 is the initial belief itself.
    """
    P = np.asarray(transmat, dtype=float)
    belief = np.asarray(initial_belief, dtype=float).flatten()
    total = belief.sum()
    if total < EPS:
        belief = np.full_like(belief, 1.0 / len(belief))
    else:
        belief = belief / total

    n = P.shape[0]
    out = np.zeros((horizon + 1, n))
    out[0] = belief

    current = belief.copy()
    for h in range(1, horizon + 1):
        current = current @ P
        s = current.sum()
        if s > EPS:
            current = current / s
        out[h] = current
    return out


def expected_regime_durations(transmat: np.ndarray) -> np.ndarray:
    """
    Expected holding time in each state: ``1 / (1 - a_ii)``.

    For ``a_ii → 1`` this returns a large finite value rather than ``inf``.
    """
    diag = np.clip(np.diag(np.asarray(transmat, dtype=float)), 0.0, 1.0 - 1e-9)
    return 1.0 / (1.0 - diag)


def survival_curve(
    transmat: np.ndarray,
    state_idx: int,
    horizon: int,
) -> np.ndarray:
    """
    ``P(chain has not left state_idx for any of the first h steps)``.

    Under the Markov property this is ``a_ii ** h`` (geometric survival).
    Returns a length-``(horizon + 1)`` array; index 0 equals ``1.0``.
    """
    a_ii = float(np.asarray(transmat)[state_idx, state_idx])
    a_ii = min(max(a_ii, 0.0), 1.0)
    h_arr = np.arange(horizon + 1)
    return a_ii ** h_arr


def first_passage_distribution(
    transmat: np.ndarray,
    from_state: int,
    to_state: int,
    horizon: int,
) -> np.ndarray:
    """
    ``P(first visit to to_state occurs exactly at step h | start in from_state)``.

    Computed by making ``to_state`` absorbing and tracking the incremental
    probability mass absorbed at each step.
    """
    P = np.asarray(transmat, dtype=float)
    n = P.shape[0]
    out = np.zeros(horizon + 1)

    if from_state == to_state:
        out[0] = 1.0
        return out

    T = P.copy()
    # Make `to_state` absorbing.
    T[to_state, :] = 0.0
    T[to_state, to_state] = 1.0

    belief = np.zeros(n)
    belief[from_state] = 1.0

    prev_absorbed = 0.0
    for h in range(1, horizon + 1):
        belief = belief @ T
        absorbed = float(belief[to_state])
        out[h] = max(absorbed - prev_absorbed, 0.0)
        prev_absorbed = absorbed
    return out


def hitting_probability(
    transmat: np.ndarray,
    from_state: int,
    to_state: int,
    horizon: int,
) -> float:
    """
    ``P(chain visits to_state at least once within horizon steps)``.

    Equal to the cumulative sum of the first-passage distribution.
    """
    fpd = first_passage_distribution(transmat, from_state, to_state, horizon)
    return float(np.clip(fpd.sum(), 0.0, 1.0))


def regime_change_probability(
    transmat: np.ndarray,
    initial_belief: np.ndarray,
    horizon: int,
) -> float:
    """
    ``P(regime flips at least once within horizon steps)`` weighted by the
    initial belief over current state.

    Uses the geometric-survival approximation ``P(no change | start in i)
    = a_ii ** horizon`` marginalised over the initial belief.
    """
    belief = np.asarray(initial_belief, dtype=float).flatten()
    total = belief.sum()
    if total < EPS:
        return 0.0
    belief = belief / total
    diag = np.clip(np.diag(np.asarray(transmat, dtype=float)), 0.0, 1.0)
    no_change = float(np.sum(belief * diag ** max(horizon, 0)))
    return float(np.clip(1.0 - no_change, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────
# Return / risk forecasts
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ForecastResult:
    """Container for a full horizon forecast."""
    horizon: int
    distribution: np.ndarray            # (H + 1, n_states)
    step_expected_return: np.ndarray    # (H + 1,)
    step_variance: np.ndarray           # (H + 1,)
    cum_expected_return: np.ndarray     # (H + 1,)
    cum_variance: np.ndarray            # (H + 1,)
    cum_sharpe: np.ndarray              # (H + 1,)
    optimal_horizon: int
    optimal_sharpe: float
    regime_change_prob: np.ndarray      # (H + 1,)
    regime_labels: list[str]

    def to_dataframe(self) -> pd.DataFrame:
        records: dict = {
            "step": np.arange(self.horizon + 1),
            "expected_return_step": self.step_expected_return,
            "variance_step": self.step_variance,
            "cum_expected_return": self.cum_expected_return,
            "cum_std": np.sqrt(np.maximum(self.cum_variance, 0.0)),
            "cum_sharpe": self.cum_sharpe,
            "regime_change_prob": self.regime_change_prob,
        }
        for idx, lbl in enumerate(self.regime_labels):
            records[f"P({lbl})"] = self.distribution[:, idx]
        return pd.DataFrame(records)


def _step_mean_variance(
    distribution_row: np.ndarray,
    regime_means: np.ndarray,
    regime_vars: np.ndarray,
) -> tuple[float, float]:
    """
    One-step mean and variance of a regime-mixture return.

        E[r]  = Σ p_i μ_i
        Var[r] = Σ p_i (σ_i² + μ_i²) − E[r]²       (law of total variance)
    """
    p = distribution_row
    mean = float(np.sum(p * regime_means))
    second = float(np.sum(p * (regime_vars + regime_means ** 2)))
    var = max(second - mean ** 2, 0.0)
    return mean, var


def horizon_forecast(
    transmat: np.ndarray,
    initial_belief: np.ndarray,
    regime_means: np.ndarray,
    regime_vars: np.ndarray,
    labels: dict[int, str] | None = None,
    horizon: int = 60,
    annualization: float = 1.0,
) -> ForecastResult:
    """
    Full forward forecast over ``horizon`` bars.

    Parameters
    ----------
    transmat : (n, n) row-stochastic HMM transition matrix.
    initial_belief : length-n posterior at "now" (row of HMM posteriors).
    regime_means : length-n per-regime mean log-return.
    regime_vars : length-n per-regime return variance (σ², not σ).
    labels : optional mapping ``state_idx -> human label``.
    horizon : forecast horizon in bars.
    annualization : optional Sharpe scaling factor (e.g. ``√(bars/year)``).

    Returns
    -------
    ForecastResult with step and cumulative moments, Sharpe trajectory,
    optimal holding horizon, and the regime-change probability curve.
    """
    regime_means = np.asarray(regime_means, dtype=float).flatten()
    regime_vars = np.asarray(regime_vars, dtype=float).flatten()

    dist = n_step_distribution(transmat, initial_belief, horizon)

    step_mean = np.zeros(horizon + 1)
    step_var = np.zeros(horizon + 1)
    for h in range(horizon + 1):
        step_mean[h], step_var[h] = _step_mean_variance(
            dist[h], regime_means, regime_vars
        )

    # Cumulative mean / variance assuming conditional independence across
    # steps — a standard HMM-forecast approximation. Step 0 contributes
    # nothing (it is the "before any step is taken" snapshot).
    cum_mean = np.concatenate([[0.0], np.cumsum(step_mean[1:])])
    cum_var = np.concatenate([[0.0], np.cumsum(step_var[1:])])
    cum_std = np.sqrt(np.maximum(cum_var, 0.0))

    cum_sharpe = np.zeros(horizon + 1)
    nonzero = cum_std > EPS
    cum_sharpe[nonzero] = (cum_mean[nonzero] / cum_std[nonzero]) * annualization

    if horizon >= 1:
        opt_h = int(np.argmax(cum_sharpe[1:]) + 1)
        opt_sharpe = float(cum_sharpe[opt_h])
    else:
        opt_h = 0
        opt_sharpe = 0.0

    rc_prob = np.array([
        regime_change_probability(transmat, initial_belief, h)
        for h in range(horizon + 1)
    ])

    label_list = [
        (labels.get(i, f"state_{i}") if labels else f"state_{i}")
        for i in range(np.asarray(transmat).shape[0])
    ]

    return ForecastResult(
        horizon=horizon,
        distribution=dist,
        step_expected_return=step_mean,
        step_variance=step_var,
        cum_expected_return=cum_mean,
        cum_variance=cum_var,
        cum_sharpe=cum_sharpe,
        optimal_horizon=opt_h,
        optimal_sharpe=opt_sharpe,
        regime_change_prob=rc_prob,
        regime_labels=label_list,
    )


# ──────────────────────────────────────────────────────────────────────────
# Fragility index (novel)
# ──────────────────────────────────────────────────────────────────────────

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) with safety clipping."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    q = np.clip(np.asarray(q, dtype=float), EPS, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


@dataclass
class FragilityReport:
    """Regime Fragility Index decomposition."""
    index: float                 # composite fragility ∈ [0, 1]
    posterior_gap: float         # top1 − top2 posterior
    entropy_gradient: float      # d/dt entropy over a recent window
    kl_from_stationary: float    # KL(current_belief || π)
    change_probability: float    # P(regime flips within `horizon` bars)
    components: dict[str, float]


def regime_fragility_index(
    posteriors: np.ndarray,
    transmat: np.ndarray,
    entropy_series: np.ndarray,
    horizon: int = 10,
    grad_window: int = 5,
) -> FragilityReport:
    """
    Novel composite fragility metric in ``[0, 1]`` where 0 means rock-solid
    regime conviction and 1 means an imminent regime change.

    Components (each normalised to ``[0, 1]`` then averaged):

        1. Posterior closeness   — small top1-top2 gap ⇒ fragile.
        2. Entropy trend         — rising entropy ⇒ fragile.
        3. Stationary convergence — belief close to π ⇒ no discriminating
           edge (chain drifting to long-run average) ⇒ fragile.
        4. Horizon change prob.  — high ``P(flip)`` over ``horizon`` ⇒ fragile.

    The composite is the arithmetic mean of the four sub-scores.
    """
    posteriors = np.asarray(posteriors, dtype=float)
    if posteriors.ndim != 2 or len(posteriors) == 0:
        raise ValueError("posteriors must be a non-empty 2-D array")

    last_post = posteriors[-1]
    sorted_post = np.sort(last_post)[::-1]
    top_gap = float(sorted_post[0] - sorted_post[1]) if len(sorted_post) > 1 else 1.0

    # 1. Posterior closeness — larger gap ⇒ lower fragility.
    closeness = float(1.0 - np.clip(top_gap, 0.0, 1.0))

    # 2. Entropy gradient normalised by a characteristic entropy scale.
    entropy_series = np.asarray(entropy_series, dtype=float)
    if len(entropy_series) >= grad_window + 1:
        recent = entropy_series[-(grad_window + 1):]
        raw_grad = float(recent[-1] - recent[0])
    else:
        raw_grad = 0.0
    n_states = int(np.asarray(transmat).shape[0])
    denom = max(np.log2(max(n_states, 2)) / 4.0, EPS)
    grad_score = float(np.clip(0.5 + 0.5 * np.tanh(raw_grad / denom), 0.0, 1.0))

    # 3. KL from stationary — closer to π ⇒ higher fragility. Map via exp(-KL).
    pi = stationary_distribution(transmat)
    kl = kl_divergence(last_post, pi)
    stationary_score = float(np.exp(-kl))  # ≈ 1 when belief ≈ π

    # 4. Regime change probability over the horizon.
    rc_prob = regime_change_probability(transmat, last_post, horizon)

    components = {
        "posterior_closeness": closeness,
        "entropy_gradient": grad_score,
        "stationary_convergence": stationary_score,
        "horizon_change_probability": rc_prob,
    }
    index = float(np.mean(list(components.values())))

    return FragilityReport(
        index=index,
        posterior_gap=top_gap,
        entropy_gradient=raw_grad,
        kl_from_stationary=kl,
        change_probability=rc_prob,
        components=components,
    )


# ──────────────────────────────────────────────────────────────────────────
# Position sizing
# ──────────────────────────────────────────────────────────────────────────

def fragility_adjusted_position(
    base_size: float,
    fragility: float,
    expected_sharpe: float,
    min_size: float = 0.0,
    max_size: float = 1.0,
) -> float:
    """
    Scale a base position size by ``(1 − fragility)`` and the positive part
    of ``tanh(expected_sharpe)``.

    Monotone in both conviction and forecast edge, bounded by ``[min, max]``,
    and returns ``0`` for negative-Sharpe forecasts.
    """
    conviction = float(np.clip(1.0 - fragility, 0.0, 1.0))
    sharpe_scale = float(np.tanh(expected_sharpe))  # ∈ [−1, 1]
    sharpe_scale = max(sharpe_scale, 0.0)
    size = base_size * conviction * sharpe_scale
    return float(np.clip(size, min_size, max_size))


# ──────────────────────────────────────────────────────────────────────────
# Convenience wrapper operating directly on RegimeDetector outputs
# ──────────────────────────────────────────────────────────────────────────

def forecast_from_detector(
    detector,
    posteriors: np.ndarray,
    regime_stats: pd.DataFrame,
    horizon: int = 60,
    annualization: float = 1.0,
    fragility_horizon: int = 10,
) -> tuple[ForecastResult, FragilityReport]:
    """
    One-call convenience: build forecast & fragility from a fitted
    ``RegimeDetector`` plus its decoded posteriors.

    Uses the ``mean_return`` and ``volatility`` columns from ``regime_stats``
    (sorted by state index) for per-regime first and second moments.
    """
    transmat = detector.transition_matrix()
    labels = detector.labels

    stats_sorted = regime_stats.sort_values("state")
    regime_means = stats_sorted["mean_return"].to_numpy(dtype=float)
    regime_vars = stats_sorted["volatility"].to_numpy(dtype=float) ** 2

    last_post = posteriors[-1]
    clipped = np.clip(posteriors, EPS, 1.0)
    entropy_series = -np.sum(clipped * np.log2(clipped), axis=1)

    forecast = horizon_forecast(
        transmat=transmat,
        initial_belief=last_post,
        regime_means=regime_means,
        regime_vars=regime_vars,
        labels=labels,
        horizon=horizon,
        annualization=annualization,
    )
    fragility = regime_fragility_index(
        posteriors=posteriors,
        transmat=transmat,
        entropy_series=entropy_series,
        horizon=fragility_horizon,
    )
    return forecast, fragility
