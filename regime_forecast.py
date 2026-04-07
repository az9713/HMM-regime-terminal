"""
regime_forecast.py — Forward-looking regime projection via Markov chain analytics.

While the rest of the terminal answers "what regime are we in now?", this module
answers "what regime will we be in next?" using closed-form Markov chain math
on the fitted HMM transition matrix:

    1. k-step posterior propagation:    π_{t+k} = π_t · T^k
    2. Spectral mixing time:            from second eigenvalue of T
    3. Per-state half-life:             ln(0.5) / ln(a_ii)
    4. Expected first-passage times:    solve (I - T_Q) h = 1
    5. Hit-by-horizon probabilities:    absorbing-chain construction
    6. Expected horizon return:         Σ_k (π_t T^k) · μ
    7. Forecast entropy cone:           uncertainty growth over the horizon

No simulation, no retraining — these are exact analytic forecasts derived from
the same HMM the rest of the dashboard already fits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ForecastTrajectory:
    """k-step ahead posterior trajectory for a single starting distribution."""

    horizons: np.ndarray            # shape (H+1,) with 0..H
    distributions: np.ndarray       # shape (H+1, n_states)
    entropies: np.ndarray           # shape (H+1,)
    expected_returns: np.ndarray    # shape (H+1,)  per-step expected log return
    cumulative_returns: np.ndarray  # shape (H+1,)  expected cumulative log return


@dataclass
class HittingAnalysis:
    """First-passage / absorbing chain analysis to one or more target states."""

    target_states: list[int]
    expected_bars: dict[int, float]                    # state_id -> E[T_hit]
    hit_prob_by_horizon: pd.DataFrame                  # rows=horizon, cols=from-state label


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


class RegimeForecastEngine:
    """
    Forward-projection analytics on an HMM transition matrix.

    Parameters
    ----------
    transmat : (n, n) ndarray
        Row-stochastic transition matrix from a fitted HMM.
    state_means : (n,) ndarray
        Mean log-return per state (typically RegimeDetector.model.means_[:, 0]).
    labels : dict[int, str]
        Mapping state-id -> human readable label.
    """

    def __init__(
        self,
        transmat: np.ndarray,
        state_means: np.ndarray,
        labels: dict[int, str],
    ):
        T = np.asarray(transmat, dtype=float)
        if T.ndim != 2 or T.shape[0] != T.shape[1]:
            raise ValueError("transmat must be a square 2-D array")
        if not np.allclose(T.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("transmat rows must sum to 1")
        if T.shape[0] != len(state_means):
            raise ValueError("state_means length must match transmat dimension")

        self.transmat = T
        self.state_means = np.asarray(state_means, dtype=float)
        self.labels = dict(labels)
        self.n_states = T.shape[0]

    # ------------------------------------------------------------------
    # 1) k-step posterior propagation
    # ------------------------------------------------------------------

    def propagate(self, pi0: np.ndarray, horizon: int) -> ForecastTrajectory:
        """
        Project the starting distribution π0 forward `horizon` steps using
        the analytic recursion π_{k+1} = π_k · T.

        Returns horizon+1 distributions (including the starting one at k=0),
        per-step Shannon entropy, and the expected log-return trajectory.
        """
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        pi0 = np.asarray(pi0, dtype=float).reshape(-1)
        if pi0.shape[0] != self.n_states:
            raise ValueError("pi0 length must equal n_states")
        s = pi0.sum()
        if s <= 0:
            raise ValueError("pi0 must have positive mass")
        pi0 = pi0 / s

        H = horizon
        n = self.n_states
        dists = np.zeros((H + 1, n))
        dists[0] = pi0
        for k in range(1, H + 1):
            dists[k] = dists[k - 1] @ self.transmat

        eps = 1e-12
        clipped = np.clip(dists, eps, 1.0)
        entropies = -np.sum(clipped * np.log2(clipped), axis=1)

        per_step_returns = dists @ self.state_means
        cumulative_returns = np.cumsum(per_step_returns) - per_step_returns[0]

        return ForecastTrajectory(
            horizons=np.arange(H + 1),
            distributions=dists,
            entropies=entropies,
            expected_returns=per_step_returns,
            cumulative_returns=cumulative_returns,
        )

    # ------------------------------------------------------------------
    # 2) Spectral mixing time
    # ------------------------------------------------------------------

    def stationary_distribution(self) -> np.ndarray:
        """Left eigenvector of T corresponding to eigenvalue 1, normalized."""
        eigvals, eigvecs = np.linalg.eig(self.transmat.T)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        v = np.real(eigvecs[:, idx])
        v = v / v.sum()
        # Numerical safety
        v = np.clip(v, 0.0, None)
        return v / v.sum()

    def spectral_gap(self) -> float:
        """
        1 - |λ_2|, where λ_2 is the second-largest-modulus eigenvalue of T.
        Larger spectral gap => faster mixing.
        """
        eigvals = np.linalg.eigvals(self.transmat)
        moduli = np.sort(np.abs(eigvals))[::-1]
        if len(moduli) < 2:
            return 1.0
        return float(1.0 - moduli[1])

    def mixing_time(self, epsilon: float = 0.01) -> float:
        """
        Approximate ε-mixing time from the spectral gap:

            t_mix(ε)  ≈  ln(1/ε) / (1 - |λ_2|)

        Interpretation: the number of bars after which the chain's distribution
        is within ε of stationary in total variation, regardless of starting
        state. This is the *theoretical horizon of predictability* — beyond
        this many bars, the HMM has effectively forgotten today's state.
        """
        gap = self.spectral_gap()
        if gap <= 1e-12:
            return float("inf")
        return float(np.log(1.0 / epsilon) / gap)

    def regime_half_life(self) -> pd.DataFrame:
        """
        Per-state half-life: expected number of bars until the probability
        of *still being in this state* (under the geometric self-loop model)
        decays to 0.5.

            half_life_i = ln(0.5) / ln(a_ii)

        Returns a DataFrame with state, label, self_loop_prob, expected_duration,
        and half_life columns.
        """
        n = self.n_states
        rows = []
        diag = np.diag(self.transmat)
        for i in range(n):
            a_ii = float(diag[i])
            label = self.labels.get(i, f"state_{i}")
            duration = 1.0 / (1.0 - a_ii) if a_ii < 1.0 else float("inf")
            if 0.0 < a_ii < 1.0:
                half_life = float(np.log(0.5) / np.log(a_ii))
            elif a_ii >= 1.0:
                half_life = float("inf")
            else:
                half_life = 0.0
            rows.append({
                "state": i,
                "label": label,
                "self_loop_prob": a_ii,
                "expected_duration": duration,
                "half_life": half_life,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 3) Hitting time / absorbing chain analysis
    # ------------------------------------------------------------------

    def expected_hitting_times(self, target_states: list[int]) -> dict[int, float]:
        """
        Expected first-passage times from every non-target state to the
        (union of) target_states, computed by absorbing-chain analysis.

        Construction: split states into transient Q (non-target) and
        absorbing R (target). Then h = (I - Q)^(-1) · 1 gives the expected
        number of steps to absorption from each transient state.

        Returns dict {state_id: E[T_hit]}, with target states mapped to 0.
        """
        n = self.n_states
        targets = sorted(set(int(s) for s in target_states))
        for s in targets:
            if s < 0 or s >= n:
                raise ValueError(f"target state {s} out of range")

        result: dict[int, float] = {s: 0.0 for s in targets}
        transient = [i for i in range(n) if i not in targets]
        if not transient:
            return result

        Q = self.transmat[np.ix_(transient, transient)]
        I = np.eye(len(transient))

        # If the spectral radius of Q is ~1, some transient states never reach
        # the target set (closed sub-chain). Mark those as +inf.
        try:
            spectral_radius = float(np.max(np.abs(np.linalg.eigvals(Q))))
        except np.linalg.LinAlgError:
            spectral_radius = 1.0

        if spectral_radius >= 1.0 - 1e-9:
            # Absorption probability < 1 for at least some states. Compute
            # which transient states are "trapped" by checking absorption
            # probability via large-power propagation of the absorbing chain.
            T_abs = self.transmat.copy()
            for t in targets:
                T_abs[t, :] = 0.0
                T_abs[t, t] = 1.0
            T_pow = np.linalg.matrix_power(T_abs, 10_000)
            for state_id in transient:
                absorb_prob = float(sum(T_pow[state_id, t] for t in targets))
                if absorb_prob >= 1.0 - 1e-6:
                    # Recoverable: solve restricted system below
                    pass
                else:
                    result[state_id] = float("inf")
            # Solve the well-posed sub-system for the recoverable states
            recoverable = [
                s for s in transient
                if s not in result or not np.isinf(result[s])
            ]
            if recoverable:
                Qr = self.transmat[np.ix_(recoverable, recoverable)]
                Ir = np.eye(len(recoverable))
                try:
                    hr = np.linalg.solve(Ir - Qr, np.ones(len(recoverable)))
                except np.linalg.LinAlgError:
                    hr = np.full(len(recoverable), np.inf)
                for idx, state_id in enumerate(recoverable):
                    val = float(hr[idx])
                    if val < 0 or not np.isfinite(val):
                        val = float("inf")
                    result[state_id] = val
            return result

        # Standard well-posed case: I - Q is non-singular.
        h = np.linalg.solve(I - Q, np.ones(len(transient)))
        for idx, state_id in enumerate(transient):
            val = float(h[idx])
            if val < 0 or not np.isfinite(val):
                val = float("inf")
            result[state_id] = val
        return result

    def hit_probability_by_horizon(
        self,
        target_states: list[int],
        horizon: int,
    ) -> pd.DataFrame:
        """
        For each non-target starting state, the probability of hitting the
        target set within k bars, for k = 1..horizon.

        Built by making the target set absorbing and propagating each
        canonical basis vector forward.
        """
        n = self.n_states
        targets = set(int(s) for s in target_states)
        T_abs = self.transmat.copy()
        for t in targets:
            T_abs[t, :] = 0.0
            T_abs[t, t] = 1.0

        rows = []
        for k in range(1, horizon + 1):
            T_pow = np.linalg.matrix_power(T_abs, k)
            row = {"horizon": k}
            for i in range(n):
                if i in targets:
                    continue
                p_hit = float(sum(T_pow[i, t] for t in targets))
                row[self.labels.get(i, f"state_{i}")] = p_hit
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 4) Convenience / reporting
    # ------------------------------------------------------------------

    def regime_probabilities_at(
        self,
        pi0: np.ndarray,
        horizons: list[int],
    ) -> pd.DataFrame:
        """
        Tabular regime-probability snapshot at requested forecast horizons,
        starting from distribution pi0. Used by the dashboard.
        """
        max_h = max(horizons) if horizons else 0
        traj = self.propagate(pi0, max_h)
        rows = []
        for h in horizons:
            row = {"horizon": h, "entropy": float(traj.entropies[h])}
            for i in range(self.n_states):
                row[self.labels.get(i, f"state_{i}")] = float(traj.distributions[h, i])
            row["expected_log_return"] = float(traj.expected_returns[h])
            row["cumulative_log_return"] = float(traj.cumulative_returns[h])
            rows.append(row)
        return pd.DataFrame(rows)

    def forecast_summary(
        self,
        pi0: np.ndarray,
        target_label: str = "crash",
        horizon: int = 168,
    ) -> dict:
        """
        High-level one-call summary used by the dashboard. Bundles
        propagation, hitting times, mixing time, and half-lives.
        """
        # Resolve target label -> state ids
        targets = [
            sid for sid, lab in self.labels.items() if lab == target_label
        ]

        traj = self.propagate(pi0, horizon)
        stationary = self.stationary_distribution()
        gap = self.spectral_gap()
        t_mix = self.mixing_time()
        half_lives = self.regime_half_life()

        if targets:
            hit_times = self.expected_hitting_times(targets)
            hit_curve = self.hit_probability_by_horizon(targets, horizon)
            current_state = int(np.argmax(pi0))
            expected_bars_to_target = hit_times.get(current_state, float("nan"))
        else:
            hit_times = {}
            hit_curve = pd.DataFrame()
            expected_bars_to_target = float("nan")

        return {
            "trajectory": traj,
            "stationary": stationary,
            "spectral_gap": gap,
            "mixing_time": t_mix,
            "half_lives": half_lives,
            "target_label": target_label,
            "target_states": targets,
            "expected_hitting_times": hit_times,
            "hit_probability_curve": hit_curve,
            "expected_bars_to_target": expected_bars_to_target,
        }
