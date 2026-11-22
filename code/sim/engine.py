"""Design engine: everything that can be precomputed for one (problem, simulator) pair.

The engine turns stances and simulator distributions into affine utilities, base-set bounds and
mis-set masses, scores all packages under the truth and under the simulator, and evaluates any
selected package against the truth.

Information discipline: `true_org`, `s_minus` and `s_plus` cover *all* organisations because the
experiments need the truth for evaluation. A decision rule or certificate may only read the rows
of known (pilot) organisations; `tests/test_information_flow.py` enforces this.

Methodology map (03_methodology.md): sections 1.1-1.2 (welfare, plug-in design, overestimate,
regret, group decomposition, illusory fairness) and 4.2 step 4 (base sets, mis-set mass).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps

import numpy as np

from . import certify, welfare
from .optimize import PackageSpace, best_of_n_pmf
from .simulators.base import SimulatorOutput
from .welfare import Affine


def _memo(method):
    """Cache a method's result per instance and argument (the cache dies with the engine)."""
    @wraps(method)
    def wrapper(self, *args):
        key = (method.__name__, *args)
        if key not in self._cache:
            self._cache[key] = method(self, *args)
        return self._cache[key]
    return wrapper


@dataclass(frozen=True)
class EngineSettings:
    """Defaults follow the methodology (section 7): delta = 0.1 split equally across groups,
    top-1 base sets, equal group weights."""
    delta: float = 0.1
    group_weights: str = "equal"        # 'equal' | 'size'
    base_set: str = "top1"              # 'top1' | 'topk' | 'mass'
    base_set_k: int = 1
    base_set_mass: float = 0.9
    miss_rule: str = "sigma_hull"       # 'sigma_hull' | 'label'
    ucb: str = "betting"                # main certificate: 'betting' | 'hoeffding_serfling'
    betting_strategy: str = "prpl"      # 'prpl' | 'dkelly'
    betting_c: float = 0.75
    split_two_sided: bool = False
    clip_trivial: bool = True
    count_rank: str | float = "best"    # GUARD-count: 'best' | 'max' | quantile in (0, 1]
    cons_delta: float = 0.1             # GUARD-cons: error budget of the improvement test
    cons_bound: str = "t"               # 't' (moderated t, asymptotic) | 'hoeffding_serfling' (finite-sample)
    cons_prior_obs: float = 2.0         # moderated t: prior degrees of freedom on the worst-case variance
    ens_kappa: float = 1.0              # Ensemble-LCB: mean - kappa * sd
    hedgetune_log2n: tuple = ()         # candidate pressures; () means 0, 2, ..., Q
    hedgetune_estimate: str = "pilot_only"  # welfare estimate used to pick N: 'pilot_only' | 'rectified'

    def delta_per_group(self, n_groups: int) -> np.ndarray:
        return np.full(n_groups, self.delta / n_groups)


class DesignEngine:
    """Precomputed tables for one problem and one simulator."""

    def __init__(self, prob, sim: SimulatorOutput, settings: EngineSettings = None,
                 space: PackageSpace = None, board: np.ndarray = None):
        if sim.n_cells != prob.n_cells:
            raise ValueError(f"simulator {sim.name!r} has {sim.n_cells} cells, the problem has {prob.n_cells}")
        self._cache = {}
        self.prob = prob
        self.sim = sim
        self.settings = settings or EngineSettings()
        self.space = space or PackageSpace(prob.n_dec)
        self.G, self.Q, self.n_org = prob.n_groups, prob.n_dec, prob.n_org
        self.gi = prob.g_index
        self.sizes = prob.group_sizes.astype(float)
        self.weights = welfare.group_weights(self.sizes, self.settings.group_weights)
        self.board_index = None if board is None else self.space.index(board)

        sv = prob.sigma_vec
        self.true_org = welfare.org_affine_from_sigma(prob, prob.sigma)
        self.sim_sigma = sim.expected_sigma(sv)
        self.sim_org = welfare.org_affine_from_sigma(prob, self.sim_sigma)
        self.member_org = ([welfare.org_affine_from_sigma(prob, ms) for ms in sim.member_sigma(sv)]
                           if sim.n_members else [])

        s = self.settings
        self.base = certify.base_sets(sim.probs, sv, s.base_set, s.base_set_k, s.base_set_mass)
        self.lam_org, self.ups_org = certify.org_bound_affines(prob.cell_org, prob.cell_dec,
                                                               self.n_org, self.Q, self.base)
        miss = certify.cell_miss(self.base, prob.sigma, prob.stance_idx, s.miss_rule)
        self.s_minus, self.s_plus, self.s_max = certify.org_misset_mass(prob.cell_org, self.n_org,
                                                                        self.base, miss)
        # |D_iq| <= cells of org i on decision q divided by |Q_i|: known from the response pattern
        self.coef_range = np.zeros((self.n_org, self.Q))
        np.add.at(self.coef_range, (prob.cell_org, prob.cell_dec), 1.0 / prob.cells_per_org[prob.cell_org])

        self.true_group = welfare.group_mean(self.true_org, self.gi, self.G)
        self.sim_group = welfare.group_mean(self.sim_org, self.gi, self.G)
        self.W_true = self.true_group.table()
        self.W_sim = self.sim_group.table()
        self._group_onehot = (self.gi[None, :] == np.arange(self.G)[:, None]).astype(float)

    # ------------------------------------------------------------------ objectives
    def values(self, table: np.ndarray, objective) -> np.ndarray:
        """(M,) social welfare M_p of a (M, G) table of group values."""
        return welfare.power_mean(table, welfare.objective_p(objective), self._w())

    def value_at(self, group_values: np.ndarray, objective) -> float:
        return float(welfare.power_mean(np.asarray(group_values, dtype=float),
                                        welfare.objective_p(objective), self._w()))

    def _w(self):
        return None if self.settings.group_weights == "equal" else self.weights

    @_memo
    def true_values(self, objective: str) -> np.ndarray:
        return self.values(self.W_true, objective)

    @_memo
    def sim_values(self, objective: str) -> np.ndarray:
        return self.values(self.W_sim, objective)

    @_memo
    def oracle_index(self, objective: str) -> int:
        return self.space.argmax(self.true_values(objective))

    @_memo
    def plugin_index(self, objective: str) -> int:
        """Simulator-only plug-in design: argmax of the simulated welfare (section 1.2)."""
        return self.space.argmax(self.sim_values(objective))

    @_memo
    def sim_order(self, objective: str) -> np.ndarray:
        """Packages sorted from best to worst simulated welfare (ties: more ED proposals first)."""
        return np.lexsort((-self.space.popcount, -self.sim_values(objective)))

    @_memo
    def best_of_n_pmf(self, log2_n: int) -> np.ndarray:
        return best_of_n_pmf(self.space.size, 2 ** int(log2_n))

    # ------------------------------------------------------------------ group sums
    def group_sum(self, mask: np.ndarray, org: Affine) -> Affine:
        """Per-group sum of organisation-level affine functions over the organisations in `mask`."""
        Z = self._group_onehot * mask[None, :]
        return Affine(Z @ org.c, Z @ org.D)

    def group_count(self, mask: np.ndarray) -> np.ndarray:
        return self._group_onehot @ mask.astype(float)

    # ------------------------------------------------------------------ evaluation against truth
    def evaluate(self, k: int, objective: str, est_table: np.ndarray = None) -> dict:
        """Truth-based metrics of package k (evaluation only).

        est_table: optional (M, G) table of the *estimate* the rule optimised (e.g. simulated
        welfare); adds the overestimate, its group decomposition and the illusory-fairness index.
        """
        k = int(k)
        true_g = self.W_true[k]
        true_val = float(self.true_values(objective)[k])
        oracle_val = float(self.true_values(objective)[self.oracle_index(objective)])
        out = {
            "package": k,
            "n_revised": int(self.Q - self.space.popcount[k]),
            "true_welfare": true_val,
            "regret": oracle_val - true_val,
            "true_min_group": float(true_g.min()),
            "true_worst_group": self.prob.group_names[int(np.argmin(true_g))],
            "hamming_to_oracle": self.space.hamming_between(k, self.oracle_index(objective)),
        }
        if self.board_index is not None:
            out["hamming_to_board"] = self.space.hamming_between(k, self.board_index)
        for g, name in enumerate(self.prob.group_names):
            out[f"true_W[{name}]"] = float(true_g[g])
        if est_table is not None:
            est_g = est_table[k]
            delta_g = est_g - true_g
            est_val = self.value_at(est_g, objective)
            out.update({
                "est_welfare": est_val,
                "overestimate": est_val - true_val,                      # Delta(a-hat)
                "illusory_fairness": float(est_g.min() - true_g.min()),  # IF(a-hat)
                "est_worst_group": self.prob.group_names[int(np.argmin(est_g))],
                "most_overestimated_group": self.prob.group_names[int(np.argmax(delta_g))],
                "worst_is_most_overestimated": bool(np.argmin(true_g) == np.argmax(delta_g)),
            })
            for g, name in enumerate(self.prob.group_names):
                out[f"delta[{name}]"] = float(delta_g[g])
        return out
