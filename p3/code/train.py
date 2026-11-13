"""
train.py — PLTR-MA experiment orchestration (build step 6; Algorithm 1).

Ties the five mechanisms together and produces every result the paper needs:

  (A) Convergence ablation   : vanilla MAPPO / +Lyapunov / +CPO / full PLTR-MA,
                               each over >= n_seeds seeds, evaluated on a common
                               footing (uniform reference omega) every eval_every
                               iterations -> mean +/- std learning curves.
  (B) Pareto front           : sweep omega over the simplex for the trained full
                               PLTR-MA model -> non-dominated trade-off points.
  (C) V / delta ablation      : final performance vs the Lyapunov weight V and the
                               trust-region radius delta.
  (D) Queue stability        : mean backlog + virtual-queue trace (property P1).

All results are saved to results/ as a single .npz (plus a small JSON of the
config) so plot.py (step 7) can render figures without re-training.

SCALING: everything is driven by ExperimentConfig below. To go from N=3 to
N=50 servers, change `n_servers`; nothing else needs editing. Heavier settings
(more seeds / iterations / eval episodes) are where the Kaggle/GPU-or-many-core
run pays off; the defaults here are a fast smoke configuration.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, replace

import numpy as np

from cloud_env import CloudAllocationEnv, LEDGER, ledger_phase
from mappo import MAPPO, MAPPOConfig, scalarize
from lyapunov import LyapunovTracker
from trust_region import TrustRegionMAPPO
from lagrangian_ppo import LagrangianMAPPO
from trace import WorkloadTrace
from pareto import (ParetoMAPPO, pareto_mask, simplex_grid,
                    simplex_corners_center, hypervolume)


# --------------------------------------------------------------------------- #
# Experiment configuration  (the ONE place to edit for scaling / heavier runs)
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    # environment / scaling
    n_servers: int = 3
    arrival_rate: float = 1.4       # rho~1.12: cluster runs hot so constraints bind
    horizon: int = 150

    # training budget
    n_iterations: int = 300
    n_seeds: int = 3
    ablation_seeds: int = 3        # seeds averaged in the V / delta ablations
    eval_every: int = 20
    eval_episodes: int = 6

    # PLTR-MA hyperparameters
    V: float = 0.5             # Lyapunov reward/constraint trade-off
    delta: float = 0.02        # trust-region KL radius
    alpha: float = 1.0         # Dirichlet concentration for omega sampling

    # ablation grids
    V_grid: tuple = (0.1, 0.3, 0.5, 1.0, 3.0)
    delta_grid: tuple = (0.002, 0.01, 0.02, 0.05)
    pareto_grid: int = 8       # simplex resolution for the front sweep
    pareto_seeds: int = 3      # seeds for the PLTR-MA front (hypervolume mean +/- std)
    baseline_seeds: int = 3    # seeds per fixed-weight scalarized-MAPPO policy
    hv_mc: int = 200_000       # Monte-Carlo samples for hypervolume
    baseline_grid: int = 0     # >0 => densify the fixed-weight MORL front to a
                               # simplex_grid(M, baseline_grid); 0 => corners+centre
    extra_fronts: tuple = ()   # extra conditioned variants to sweep a front for,
                               # e.g. ("cn_lyap", "cn_tr") for the component
                               # attribution table. Empty reproduces the
                               # original PLTR-vs-CN-vs-fixed-weight comparison.
    extra_conv: tuple = ()     # extra variants to add to the convergence block
    baseline_iterations: int | None = None  # per-policy budget for the
                               # fixed-weight baseline. None = same as the
                               # conditioned network (equal PER-POLICY budget,
                               # baseline gets k x the total). Set to
                               # n_iterations/k to equalise TOTAL budget instead.
    also_greedy: bool = False  # ALSO evaluate every learned policy greedily
                               # (argmax actions), strictly after its normal
                               # stochastic evaluation, on the same weights.
                               # Stochastic results are unchanged.
    omega_per_iter: int = 1    # preferences sampled per training iteration for
                               # the conditioned variants. The per-iteration step
                               # budget is SPLIT across them, so environment
                               # interaction is unchanged and only preference
                               # coverage differs. 1 reproduces the main tables.
    actor_hidden: int = 64     # width of the decentralized actors. Fixed at 64
                               # in the main tables at every cluster size; raised
                               # only in the capacity study, which asks whether
                               # one conditioned network needs capacity
                               # proportional to N to cover the preference
                               # simplex as the joint problem grows.
    trace_load: int = 1        # replay the trace at this intensity (1 = the
                               # unmodified Alibaba slice). >1 raises utilization
                               # so the capacity/SLA constraints bind.
    store_costs: bool = False  # ALSO record the mean per-step constraint costs
                               # c_k of every front point and baseline policy
                               # (env cost recorder; returns are unchanged).
    lyap_d: tuple | None = None  # Lyapunov tolerances d_k for every variant with
                               # virtual queues. None = (0,0,0), the main tables.
    contention: bool = False   # P5 revision: overcommit slows service (env option)
    simultaneous_actions: bool = False  # audit A4: env option (False = P3/v1 order)
    eval_seed_offset: int = 0   # calibration: every seeded (evaluation) episode shifted (env option)
    time_limit: str = "legacy"  # audit A11: GAE at the horizon, "legacy" | "bootstrap" (MAPPOConfig)
    isolate_eval: bool = False  # audit A1: monitoring evaluation inside fit() runs on a
                               # deep copy of the env with all RNG states restored, so it
                               # cannot change the training stream (False = P3/v1 path)
    monitor_eval: bool = True   # False: no monitoring evaluation inside fit() at all
    ledger_dir: str | None = None  # audit A3: per-job interaction ledger (JSON) written here
    baseline_eval_seed_base: int = 90000  # evaluation episodes of the fixed-weight /
                               # Lagrangian policies; 30000 = same as the swept fronts
    extra_heuristics: tuple = ()  # non-learning fronts (heuristics.py), e.g.
                               # ("heur_leastloaded", "heur_local"); evaluated once
                               # per run on the fronts' episodes (deterministic)
    extra_baselines: tuple = ()  # extra fixed-weight baselines trained with
                               # another variant, e.g. ("lagrangian",) for
                               # PPO-Lagrangian at cost_idx / cost_limit.
    with_cn: bool = True       # also train the conditioned-network (CN) MORL
                               # baseline in run_pareto; False reproduces the
                               # original two-method comparison

    # --- P3: environment axis --------------------------------------------- #
    # None => synthetic Poisson arrivals (the conference env). A path to a
    # normalised trace CSV (see make_trace.py) => real-workload replay, which is
    # the second, Q1-credible environment.
    trace_path: str | None = None

    # --- P3: PPO-Lagrangian baseline -------------------------------------- #
    cost_idx: int = 0          # which constraint to bound (0 = capacity c1)
    cost_limit: float = 0.0    # target E[c_k] <= cost_limit
    lam_lr: float = 0.2        # dual-ascent step on the Lagrange multiplier
    lam_init: float = 2.0      # initial multiplier

    # --- parallelism ------------------------------------------------------- #
    # Runs are embarrassingly parallel (independent variant/seed/weight) and the
    # bottleneck is single-threaded Python env-stepping, so PROCESS parallelism is
    # what buys speed. 0/1 => sequential. On Kaggle (~4 cores, 12h cap) use 4.
    n_workers: int = 1

    out_dir: str = "results"
    device: str = "cpu"

    def mappo_cfg(self, seed: int) -> MAPPOConfig:
        return MAPPOConfig(
            n_servers=self.n_servers, arrival_rate=self.arrival_rate,
            horizon=self.horizon, steps_per_update=self.horizon,
            n_iterations=self.n_iterations, seed=seed, device=self.device,
            hidden=self.actor_hidden, time_limit=getattr(self, "time_limit", "legacy"),
        )


VARIANTS = ("mappo", "lyapunov", "cpo", "lagrangian", "pltr_ma")


def make_env(xc: ExperimentConfig, seed: int) -> CloudAllocationEnv:
    """Fresh environment for ONE algo instance.

    Trace-driven when `xc.trace_path` is set (real-workload replay, P3's second
    environment); otherwise the synthetic Poisson env of the conference paper.
    Each algo needs its own env instance, so this is called per variant/seed.
    """
    trace = (WorkloadTrace(xc.trace_path, horizon=xc.horizon,
                           load_scale=xc.trace_load) if xc.trace_path else None)
    return CloudAllocationEnv(
        n_servers=xc.n_servers, arrival_rate=xc.arrival_rate,
        horizon=xc.horizon, seed=seed, trace=trace,
        contention=getattr(xc, "contention", False),
        simultaneous_actions=getattr(xc, "simultaneous_actions", False),
        reset_seed_offset=getattr(xc, "eval_seed_offset", 0),
    )


# --------------------------------------------------------------------------- #
# Variant factory
# --------------------------------------------------------------------------- #
def _lyap(V, xc):
    """LyapunovTracker with the configured tolerances (default d = 0)."""
    lyap_d = getattr(xc, "lyap_d", None)
    return LyapunovTracker(V=V) if lyap_d is None else LyapunovTracker(V=V, d=tuple(lyap_d))


def build_variant(name: str, xc: ExperimentConfig, seed: int,
                  V: float | None = None, delta: float | None = None):
    """Return (algo, tracker_or_None) for a named ablation variant."""
    cfg = xc.mappo_cfg(seed)
    V = xc.V if V is None else V
    delta = xc.delta if delta is None else delta
    env = make_env(xc, seed)          # honours xc.trace_path (environment axis)

    if name == "mappo":
        return MAPPO(cfg, env=env), None
    if name == "lyapunov":
        lyap = _lyap(V, xc)
        return MAPPO(cfg, env=env, reward_fn=lyap.make_reward_fn()), lyap
    if name == "cpo":
        return TrustRegionMAPPO(cfg, delta=delta, env=env), None
    if name == "lagrangian":
        # PPO-Lagrangian safe-RL baseline (learned dual on the constraint).
        return LagrangianMAPPO(cfg, env=env, cost_idx=xc.cost_idx,
                               cost_limit=xc.cost_limit, lam_lr=xc.lam_lr,
                               lam_init=xc.lam_init), None
    if name == "pltr_ma":
        lyap = _lyap(V, xc)
        return ParetoMAPPO(cfg, delta=delta, alpha=xc.alpha, env=env,
                           reward_fn=lyap.make_reward_fn(),
                           omega_per_iter=xc.omega_per_iter), lyap
    if name == "cn":
        # Conditioned-network MORL baseline (Abels et al. 2019): one
        # weight-conditioned network swept at inference, WITHOUT the Lyapunov
        # drift-plus-penalty and WITHOUT the per-agent trust region.
        # delta=None disables the KL early-stop; no reward_fn => no virtual queues.
        return ParetoMAPPO(cfg, delta=None, alpha=xc.alpha, env=env,
                           omega_per_iter=xc.omega_per_iter), None
    if name == "cn_lyap":
        # Conditioning + Lyapunov, WITHOUT the trust region (delta=None).
        # Isolates what the drift-plus-penalty term adds on top of conditioning.
        lyap = _lyap(V, xc)
        return ParetoMAPPO(cfg, delta=None, alpha=xc.alpha, env=env,
                           reward_fn=lyap.make_reward_fn()), lyap
    if name == "cn_tr":
        # Conditioning + trust region, WITHOUT the Lyapunov term (no reward_fn).
        # Isolates what the per-agent trust region adds on top of conditioning.
        return ParetoMAPPO(cfg, delta=delta, alpha=xc.alpha, env=env), None
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# Common-footing evaluation (uniform reference omega)
# --------------------------------------------------------------------------- #
UNIFORM = None  # filled per-env


def eval_policy(algo, omega, n_ep: int, seed_base: int = 90000, greedy: bool = False):
    """Mean scalar return, per-objective return, cost vector and backlog."""
    M, K = algo.env.M, algo.env.K
    sca, vec, cost, backlog = [], [], [], []
    for e in range(n_ep):
        obs, _ = algo.env.reset(seed=seed_base + e)
        rv = np.zeros(M); cs = np.zeros(K); bl = []
        done = False
        while not done:
            a, _ = (algo.act(obs, cond=omega, greedy=True) if greedy
                    else algo.act(obs, cond=omega))
            obs, r, term, trunc, info = algo.env.step(a)
            rv += r; cs += info["cost"]; bl.append(info["total_backlog"])
            done = term or trunc
        sca.append(scalarize(rv, omega)); vec.append(rv)
        cost.append(cs / max(1, len(bl))); backlog.append(np.mean(bl))
    return (float(np.mean(sca)), np.mean(vec, 0),
            np.mean(cost, 0), float(np.mean(backlog)))


class _isolated_rng:
    """Save and restore torch / numpy-global / python RNG states (audit A1)."""
    def __enter__(self):
        import random
        import torch
        self._t = torch.get_rng_state()
        self._n = np.random.get_state()
        self._p = random.getstate()
        return self

    def __exit__(self, *exc):
        import random
        import torch
        torch.set_rng_state(self._t)
        np.random.set_state(self._n)
        random.setstate(self._p)
        return False


def isolated_eval_policy(algo, omega, n_ep, **kw):
    """eval_policy on a deep copy of algo.env with every RNG state restored afterwards:
    the training env (state, RNG, recorder) and algo._obs are untouched."""
    import copy
    real = algo.env
    algo.env = copy.deepcopy(real)
    algo.env.cost_log = None
    try:
        with _isolated_rng():
            return eval_policy(algo, omega, n_ep, **kw)
    finally:
        algo.env = real


def fit(algo, xc: ExperimentConfig, omega_sampler):
    """Unified training loop with periodic common-footing evaluation."""
    uni = np.full(algo.env.M, 1.0 / algo.env.M, dtype=np.float32)
    algo._obs, _ = algo.env.reset()
    curve = {"iter": [], "eval_scalar": [], "eval_cost": [],
             "eval_backlog": [], "eval_vec": []}
    for it in range(xc.n_iterations):
        omega = omega_sampler()
        roll = algo.collect_rollout(xc.horizon, omega=omega)
        algo.update(roll)
        if not getattr(xc, "monitor_eval", True):
            continue
        if it % xc.eval_every == 0 or it == xc.n_iterations - 1:
            _ev = isolated_eval_policy if getattr(xc, "isolate_eval", False) else eval_policy
            with ledger_phase("monitor_eval"):
                s, v, c, b = _ev(algo, uni, xc.eval_episodes)
            curve["iter"].append(it)
            curve["eval_scalar"].append(s)
            curve["eval_vec"].append(v)
            curve["eval_cost"].append(c)
            curve["eval_backlog"].append(b)
    return curve


# --------------------------------------------------------------------------- #
# Parallel execution helper
# --------------------------------------------------------------------------- #
def _worker_init():
    """Pin each worker to one torch thread so N processes don't oversubscribe."""
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _pmap(fn, items, n_workers: int):
    """map(fn, items), in parallel when n_workers > 1. `fn` must be module-level.

    Start method matters: a *forked* child cannot re-initialize CUDA, so when the
    parent has CUDA available we must use "spawn". Forking is faster to start, so
    we keep it for the CPU case (where it is also perfectly safe).
    """
    if n_workers and n_workers > 1 and len(items) > 1:
        import multiprocessing as mp
        import torch
        if torch.cuda.is_available():
            method = "spawn"        # required: fork + CUDA => RuntimeError
        else:
            method = "fork" if hasattr(os, "fork") else "spawn"
        ctx = mp.get_context(method)
        with ctx.Pool(processes=min(n_workers, len(items)),
                      initializer=_worker_init) as pool:
            return pool.map(fn, items)
    return [fn(x) for x in items]


# --------------------------------------------------------------------------- #
# Experiment blocks
# --------------------------------------------------------------------------- #
def _conv_job(args):
    """One (variant, seed) convergence run. Module-level so it is picklable."""
    name, seed, xc = args
    algo, _ = build_variant(name, xc, seed)
    M = algo.env.M
    sampler = ((lambda a=algo: a.sample_omega()) if name == "pltr_ma"
               else (lambda M=M: np.full(M, 1.0 / M, np.float32)))
    t0 = time.time()
    curve = fit(algo, xc, sampler)
    print(f"  [{name:10s}] seed {seed} done ({time.time()-t0:5.1f}s) "
          f"final eval={curve['eval_scalar'][-1]:.2f}", flush=True)
    return name, seed, curve


def run_convergence(xc: ExperimentConfig) -> dict:
    """(A) all variants x seeds, common-footing learning curves."""
    jobs = [(name, seed, xc) for name in tuple(VARIANTS) + tuple(xc.extra_conv)
            for seed in range(xc.n_seeds)]
    results = _pmap(_conv_job, jobs, xc.n_workers)

    by_variant: dict[str, list] = {n: [] for n in tuple(VARIANTS) + tuple(xc.extra_conv)}
    for name, _seed, curve in sorted(results, key=lambda r: (r[0], r[1])):
        by_variant[name].append(curve)

    out = {}
    for name, per_seed in by_variant.items():
        iters = np.array(per_seed[0]["iter"])
        scal = np.array([p["eval_scalar"] for p in per_seed])      # (seeds, T)
        cost = np.array([p["eval_cost"] for p in per_seed])        # (seeds, T, K)
        back = np.array([p["eval_backlog"] for p in per_seed])
        out[name] = {
            "iter": iters,
            "scalar_mean": scal.mean(0), "scalar_std": scal.std(0),
            "cost_mean": cost.mean(0), "cost_std": cost.std(0),    # (T, K)
            "backlog_mean": back.mean(0), "backlog_std": back.std(0),
            "n_seeds": np.array(xc.n_seeds),
        }
    return out


def _train_fixed_weight_baseline(xc: ExperimentConfig, omega: np.ndarray,
                                 seed: int, variant: str = "mappo") -> np.ndarray:
    """Train ONE vanilla scalarized MAPPO at a fixed preference `omega` and
    return its mean per-objective return vector. This is the obvious
    alternative to weight-conditioning: one network per weight, retrained."""
    # The baseline's per-policy budget can be set independently of the
    # conditioned network's. Both defaults are xc.n_iterations, which keeps the
    # per-policy budget equal and leaves the baseline with k times the TOTAL
    # interaction; setting baseline_iterations instead equalises the total.
    # Whichever convention a table uses must be stated in that table.
    xcb = xc if xc.baseline_iterations is None else replace(
        xc, n_iterations=xc.baseline_iterations)
    if "@" in variant:                      # "mappo@100": per-policy iterations override
        variant, _it = variant.split("@")
        xcb = replace(xcb, n_iterations=int(_it))
    algo, _ = build_variant(variant, xcb, seed=seed)
    fit(algo, xcb, lambda w=omega: w.astype(np.float32))
    seed_base = getattr(xc, "baseline_eval_seed_base", 90000)
    if getattr(xc, "store_costs", False):
        _arm_recorder(algo.env)
        with ledger_phase("final_eval"):
            _, vec, cost, _ = eval_policy(algo, omega, xcb.eval_episodes, seed_base=seed_base)
        log = algo.env.flush_cost_log()
        algo.env.cost_log = None
        metrics = np.array([np.concatenate([c, x]) for sd, c, x in log if sd is not None])
        return vec, cost, metrics
    with ledger_phase("final_eval"):
        _, vec, cost, _ = eval_policy(algo, omega, xcb.eval_episodes, seed_base=seed_base)
        if xc.also_greedy:
            _, vec_g, _, _ = eval_policy(algo, omega, xcb.eval_episodes, greedy=True)
            return vec, vec_g
    return vec


def _sweep(algo, xc):
    """algo.sweep_front, plus (with xc.store_costs) the mean per-step cost
    vector of every front point, read from the env cost recorder. Each point's
    evaluation starts at episode seed 30000, which segments the log; padded
    rows (PGMORL) repeat point 0, exactly as the returns do."""
    if not getattr(xc, "store_costs", False):
        with ledger_phase("front_eval"):
            return algo.sweep_front(n_grid=xc.pareto_grid, n_ep=xc.eval_episodes)[1]
    env = algo.env
    _arm_recorder(env)
    with ledger_phase("front_eval"):
        _, returns, _ = algo.sweep_front(n_grid=xc.pareto_grid, n_ep=xc.eval_episodes)
    log = env.flush_cost_log()
    env.cost_log = None
    costs, metrics = _segment_log(log, len(returns))
    return returns, costs, metrics


def _arm_recorder(env):
    """Switch the env cost recorder on with an empty log, after finalizing and
    discarding any episode still open from training (e.g. PGMORL's last placement
    evaluation). Only env.t is touched, and every evaluation episode resets the env."""
    env.cost_log = []
    env.flush_cost_log()
    env.cost_log = []


def _segment_log(log, n_points):
    """Split a cost-recorder log into front points (each point's evaluation
    starts at episode seed 30000). Returns mean costs (P, K) and per-episode
    metrics (P, n_ep, len(env.LOG_FIELDS)): c1, c2, c3 and env.LOG_FIELDS[3:]. Padded rows repeat
    point 0, exactly as PGMORL's padded returns do."""
    groups = []
    for seed_, c, extra in log:
        if seed_ is None:                   # a training episode left open: not an evaluation
            continue
        if seed_ == 30_000 or not groups:
            groups.append([])
        groups[-1].append(np.concatenate([c, extra]))
    costs = np.array([np.mean(g, axis=0)[:3] for g in groups])
    metrics = np.array([np.array(g) for g in groups])
    assert 1 <= len(costs) <= n_points, (len(costs), n_points)
    if len(costs) < n_points:
        k = n_points - len(costs)
        costs = np.vstack([costs, np.repeat(costs[:1], k, axis=0)])
        metrics = np.concatenate([metrics, np.repeat(metrics[:1], k, axis=0)])
    return costs, metrics


def _front_job(args):
    """One PLTR-MA front (train once, then sweep omega). Module-level = picklable."""
    xc, seed = args
    algo, _ = build_variant("pltr_ma", xc, seed=seed)
    algo.train(verbose=False)
    if getattr(xc, "store_costs", False):
        out = _sweep(algo, xc)
        print(f"  [pltr_ma front] seed {seed} done ({len(out[0])} pts, costs)", flush=True)
        return out
    with ledger_phase("front_eval"):
        _, returns, _ = algo.sweep_front(n_grid=xc.pareto_grid, n_ep=xc.eval_episodes)
    print(f"  [pltr_ma front] seed {seed} done ({len(returns)} pts)", flush=True)
    if xc.also_greedy:
        with ledger_phase("front_eval"):
            _, returns_g, _ = algo.sweep_front(n_grid=xc.pareto_grid,
                                               n_ep=xc.eval_episodes, greedy=True)
        return returns, returns_g
    return returns


def _cn_job(args):
    """One conditioned-network (CN) front: same protocol as _front_job but the
    network has neither the Lyapunov term nor the trust region."""
    xc, seed = args
    algo, _ = build_variant("cn", xc, seed=seed)
    algo.train(verbose=False)
    if getattr(xc, "store_costs", False):
        out = _sweep(algo, xc)
        print(f"  [cn front] seed {seed} done ({len(out[0])} pts, costs)", flush=True)
        return out
    with ledger_phase("front_eval"):
        _, returns, _ = algo.sweep_front(n_grid=xc.pareto_grid, n_ep=xc.eval_episodes)
    print(f"  [cn front] seed {seed} done ({len(returns)} pts)", flush=True)
    if xc.also_greedy:
        with ledger_phase("front_eval"):
            _, returns_g, _ = algo.sweep_front(n_grid=xc.pareto_grid,
                                               n_ep=xc.eval_episodes, greedy=True)
        return returns, returns_g
    return returns


def _variant_front_job(args):
    """One Pareto front for a named conditioned variant (cn_lyap, cn_tr, ...).
    Same sweep protocol as _front_job; only the variant differs."""
    xc, name, seed = args
    algo, _ = build_variant(name, xc, seed=seed)
    algo.train(verbose=False)
    if getattr(algo, "last_front_size", None) is not None or hasattr(algo, "ep"):
        _JOB_INFO["front_members_trained"] = len(getattr(algo, "ep", []) or [])
    if hasattr(algo, "selection_diagnostics"):      # PGMORL: selection-time sparsity weight
        _JOB_INFO["selection_diagnostics"] = algo.selection_diagnostics()
    if hasattr(algo, "cost_report"):
        _rep = algo.cost_report()
        _JOB_INFO["allowance_unused"] = _rep.get("allowance_unused")
    if getattr(xc, "store_costs", False):
        out = _sweep(algo, xc)
        if getattr(algo, "last_front_size", None) is not None:
            _JOB_INFO["front_size_real"] = int(algo.last_front_size)
        print(f"  [{name} front] seed {seed} done ({len(out[0])} pts, costs)", flush=True)
        return out
    with ledger_phase("front_eval"):
        _, returns, _ = algo.sweep_front(n_grid=xc.pareto_grid, n_ep=xc.eval_episodes)
    if getattr(algo, "last_front_size", None) is not None:
        _JOB_INFO["front_size_real"] = int(algo.last_front_size)
    print(f"  [{name} front] seed {seed} done ({len(returns)} pts)", flush=True)
    return returns


def _heur_job(args):
    """One non-learning front: sweep the heuristic's parameter on the swept
    fronts' episodes (seed 30000+e). Returns returns (P, M) and, with
    store_costs, mean costs (P, K) and per-episode metrics (P, n_ep, 9)."""
    xc, name = args
    from heuristics import heuristic_front, FRONT_SEED_BASE
    env = make_env(xc, seed=0)
    store = getattr(xc, "store_costs", False)
    if store:
        _arm_recorder(env)
    with ledger_phase("front_eval"):
        _, returns, _, _ = heuristic_front(name, env, n_ep=xc.eval_episodes,
                                           seed_base=FRONT_SEED_BASE)
    print(f"  [{name} front] done ({len(returns)} pts)", flush=True)
    if not store:
        return (returns,)
    log = env.flush_cost_log()
    env.cost_log = None
    costs, metrics = _segment_log(log, len(returns))
    return returns, costs, metrics


def _base_job(args):
    """One fixed-weight baseline policy (train at a fixed omega, then evaluate)."""
    xc, omega, seed = args[:3]
    variant = args[3] if len(args) > 3 else "mappo"
    vec = _train_fixed_weight_baseline(xc, omega, seed, variant)
    print(f"  [baseline] seed {seed} w={np.round(omega, 2)} done", flush=True)
    return vec


# --------------------------------------------------------------------------- #
# Interaction ledger sidecars (audit A3). One JSON per job when xc.ledger_dir is set.
# --------------------------------------------------------------------------- #
_JOB_INFO: dict = {}
_CODE_VERSION = None


def code_version() -> str:
    """sha256 over every source a revision run depends on: simulator / harness
    (p3/code/*.py), methods (p5/methods/*.py), the entry point p5/run_p5_comparison.py, the
    analysis / validation / calibration / queue code (p5/analysis/*.py) and the queue
    wrapper script (p5/queue21*.sh; the superseded queue20 draft is not deployed). Paths relative to the tree root are hashed with the
    content. Data inputs (trace, CALIBRATION.env) are hashed separately into PROVENANCE."""
    global _CODE_VERSION
    if _CODE_VERSION is None:
        import glob
        import hashlib
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", ".."))
        files = sorted(glob.glob(os.path.join(here, "*.py")))
        for pat in (("p5", "methods", "*.py"), ("p5", "run_p5_comparison.py"), ("p5", "analysis", "*.py"),
                    ("p5", "queue21*.sh")):
            files += sorted(glob.glob(os.path.join(root, *pat)))
        h = hashlib.sha256()
        for f in files:
            h.update(os.path.relpath(f, root).replace(os.sep, "/").encode())
            with open(f, "rb") as fh:
                h.update(fh.read())
        _CODE_VERSION = h.hexdigest()
    return _CODE_VERSION


def _ledgered(kind):
    """Wrap a job: count its env steps per phase and write a sidecar JSON."""
    import functools

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(args):
            xc = args[0]
            before = dict(LEDGER)
            _JOB_INFO.clear()
            t0 = time.time()
            out = fn(args)
            ldir = getattr(xc, "ledger_dir", None)
            if ldir:
                steps = {k: LEDGER.get(k, 0) - before.get(k, 0) for k in LEDGER}
                steps = {k: v for k, v in steps.items() if v}
                rec = {"kind": kind, "code_version": code_version(), "wall_s": time.time() - t0,
                       "env_steps": steps, "env_steps_total": sum(steps.values()),
                       "env_steps_real": sum(v for k, v in steps.items() if k != "lookahead"),
                       **_JOB_INFO}
                if kind in ("front", "cn"):
                    rec.update(method="pltr_ma" if kind == "front" else "cn", seed=args[1])
                elif kind == "variant":
                    rec.update(method=args[1], seed=args[2])
                elif kind == "heur":
                    rec.update(method=args[1], seed=None)
                elif kind == "base":
                    rec.update(method=args[3] if len(args) > 3 else "mappo", seed=args[2],
                               omega=[float(w) for w in args[1]])
                os.makedirs(ldir, exist_ok=True)
                tag = "_".join(str(x) for x in (rec["method"], rec.get("seed"),
                               "-".join(f"{w:.3f}" for w in rec.get("omega", []))) if x not in (None, ""))
                with open(os.path.join(ldir, f"{kind}_{tag}.json"), "w") as fh:
                    json.dump(rec, fh, indent=1)
            return out
        return wrapper
    return deco


_front_job = _ledgered("front")(_front_job)
_cn_job = _ledgered("cn")(_cn_job)
_variant_front_job = _ledgered("variant")(_variant_front_job)
_heur_job = _ledgered("heur")(_heur_job)
_base_job = _ledgered("base")(_base_job)


def run_pareto(xc: ExperimentConfig) -> dict:
    """(B) PLTR-MA omega-sweep front vs. a fixed-weight scalarized-MAPPO front.

    Offense, not a standalone scatter: ONE weight-conditioned PLTR-MA network
    (swept over the simplex) is compared, in the same objective space and via
    hypervolume, against k separately-trained fixed-weight MAPPO policies. The
    claim is comparable-or-broader coverage at ~1/k the training cost. Both
    fronts and the HV are reported as mean +/- std over seeds so the comparison
    is a real gap, not noise.
    """
    # --- PLTR-MA front, one per seed (single network, swept) ----------------
    pareto_seeds = max(1, xc.pareto_seeds)
    store_costs = getattr(xc, "store_costs", False)
    assert not (store_costs and xc.also_greedy), "store_costs and also_greedy are exclusive"
    cost_store = {}

    def _split(res, key):
        # with store_costs every job returns (returns, costs)
        if not store_costs:
            return res
        cost_store[key] = np.array([r[1] for r in res])
        cost_store[key.replace("costs_all", "metrics_all")] = np.array([r[2] for r in res])
        return [r[0] for r in res]

    _fr = _split(_pmap(_front_job, [(xc, s) for s in range(pareto_seeds)], xc.n_workers),
                 "costs_all")
    greedy_store = {}
    if xc.also_greedy:
        greedy_store["returns_all_greedy"] = np.array([r[1] for r in _fr])
        _fr = [r[0] for r in _fr]
    fronts = np.array(_fr)                          # (seeds, P, M)

    # --- fixed-weight scalarized-MAPPO baseline front -----------------------
    # Fixed-weight multi-policy front = the MORL baseline (linear scalarization,
    # one network retrained per preference). `baseline_grid > 0` densifies it
    # beyond corners+centre for a stronger P3 comparison.
    M_obj = fronts.shape[-1]
    base_omegas = (simplex_grid(M_obj, xc.baseline_grid) if xc.baseline_grid > 0
                   else simplex_corners_center(M_obj))            # (k, M)
    base_seeds = max(1, xc.baseline_seeds)
    base_jobs = [(xc, w, s) for s in range(base_seeds) for w in base_omegas]
    base_vecs = _pmap(_base_job, base_jobs, xc.n_workers)
    if store_costs:
        cost_store["baseline_costs_all"] = np.array([v[1] for v in base_vecs]).reshape(
            base_seeds, len(base_omegas), -1)
        _m = np.array([v[2] for v in base_vecs])
        cost_store["baseline_metrics_all"] = _m.reshape(base_seeds, len(base_omegas), *_m.shape[1:])
        base_vecs = [v[0] for v in base_vecs]
    if xc.also_greedy:
        greedy_store["baseline_returns_all_greedy"] = np.array(
            [v[1] for v in base_vecs]).reshape(base_seeds, len(base_omegas), M_obj)
        base_vecs = [v[0] for v in base_vecs]
    base = np.array(base_vecs).reshape(base_seeds, len(base_omegas), M_obj)

    # --- conditioned-network (CN) MORL baseline, one front per seed ---------
    cn_fronts = None
    if xc.with_cn:
        _cf = _split(_pmap(_cn_job, [(xc, s) for s in range(pareto_seeds)], xc.n_workers),
                     "cn_costs_all")
        if xc.also_greedy:
            greedy_store["cn_returns_all_greedy"] = np.array([r[1] for r in _cf])
            _cf = [r[0] for r in _cf]
        cn_fronts = np.array(_cf)                       # (seeds, P, M)

    # --- component-attribution fronts (cn_lyap, cn_tr), one front per seed ---
    # Same protocol as PLTR-MA; each switches on exactly one mechanism on top of
    # conditioning, so the four fronts separate what each mechanism contributes.
    extra_fronts = {}
    for _name in xc.extra_fronts:
        extra_fronts[_name] = np.array(_split(
            _pmap(_variant_front_job, [(xc, _name, s) for s in range(pareto_seeds)],
                  xc.n_workers), f"{_name}_costs_all"))  # (seeds, P, M)

    # --- extra fixed-weight baselines trained with another variant ----------
    # e.g. PPO-Lagrangian at (cost_idx, cost_limit): same omegas, seeds, budget.
    extra_bases = {}
    for _v in getattr(xc, "extra_baselines", ()):
        _res = _pmap(_base_job, [(xc, w, s, _v) for s in range(base_seeds) for w in base_omegas],
                     xc.n_workers)
        _key = _v.replace("@", "_it")
        if store_costs:
            cost_store[f"{_key}_baseline_costs_all"] = np.array([r[1] for r in _res]).reshape(
                base_seeds, len(base_omegas), -1)
            _m = np.array([r[2] for r in _res])
            cost_store[f"{_key}_baseline_metrics_all"] = _m.reshape(base_seeds, len(base_omegas),
                                                                   *_m.shape[1:])
            _res = [r[0] for r in _res]
        extra_bases[_key] = np.array(_res).reshape(base_seeds, len(base_omegas), M_obj)

    # --- non-learning fronts (deterministic: evaluated once, on the fronts' episodes)
    heur_fronts = {}
    for _name, _res in zip(getattr(xc, "extra_heuristics", ()),
                           _pmap(_heur_job, [(xc, n) for n in getattr(xc, "extra_heuristics", ())],
                                 xc.n_workers)):
        heur_fronts[_name] = np.asarray(_res[0])[None]                 # (1, P, M)
        if store_costs:
            cost_store[f"{_name}_costs_all"] = np.asarray(_res[1])[None]
            cost_store[f"{_name}_metrics_all"] = np.asarray(_res[2])[None]

    # --- shared normalization box for a fair hypervolume comparison ---------
    # The box MUST span every method being compared (pareto.hypervolume requires
    # identical lo/hi), so adding CN widens it and shifts the absolute HV of all
    # methods. Ratios are what carry the claim.
    parts = [fronts.reshape(-1, fronts.shape[-1]), base.reshape(-1, base.shape[-1])]
    if cn_fronts is not None:
        parts.append(cn_fronts.reshape(-1, cn_fronts.shape[-1]))
    for _f in extra_fronts.values():
        parts.append(_f.reshape(-1, _f.shape[-1]))
    for _b in extra_bases.values():
        parts.append(_b.reshape(-1, _b.shape[-1]))
    for _h in heur_fronts.values():
        parts.append(_h.reshape(-1, _h.shape[-1]))
    allpts = np.concatenate(parts, axis=0)
    lo, hi = allpts.min(0), allpts.max(0)

    hv_pltr = np.array([hypervolume(f, lo, hi, n_mc=xc.hv_mc) for f in fronts])
    hv_base = np.array([hypervolume(b, lo, hi, n_mc=xc.hv_mc) for b in base])
    hv_cn = (np.array([hypervolume(f, lo, hi, n_mc=xc.hv_mc) for f in cn_fronts])
             if cn_fronts is not None else None)

    # seed 0 front + its mask for the scatter; means for the baseline markers
    omegas0 = simplex_grid(fronts.shape[-1], xc.pareto_grid)
    returns0 = fronts[0]
    out = {
        "omegas": omegas0, "returns": returns0, "mask": pareto_mask(returns0),
        "returns_all": fronts,                       # (seeds, P, M)
        "baseline_omegas": base_omegas,              # (k, M)
        "baseline_returns": base.mean(0),            # (k, M)
        "baseline_returns_all": base,                # (seeds, k, M)
        "hv_pltr_mean": np.array(hv_pltr.mean()), "hv_pltr_std": np.array(hv_pltr.std()),
        "hv_base_mean": np.array(hv_base.mean()), "hv_base_std": np.array(hv_base.std()),
        "hv_ref_lo": lo, "hv_ref_hi": hi,
    }
    if cn_fronts is not None:
        out["cn_returns_all"] = cn_fronts                # (seeds, P, M)
        out["cn_returns"] = cn_fronts[0]
        out["hv_cn_mean"] = np.array(hv_cn.mean())
        out["hv_cn_std"] = np.array(hv_cn.std())
    out.update(greedy_store)          # empty unless also_greedy
    for _v, _b in extra_bases.items():
        out[f"{_v}_baseline_returns_all"] = _b           # (seeds, k, M)
    for _n, _h in heur_fronts.items():
        out[f"{_n}_returns_all"] = _h                    # (1, P, M): deterministic
    out.update(cost_store)                               # *_costs_all (seeds, P|k, K)
    for _name, _f in extra_fronts.items():
        hv_x = np.array([hypervolume(f, lo, hi, n_mc=xc.hv_mc) for f in _f])
        out[f"{_name}_returns_all"] = _f                 # (seeds, P, M)
        out[f"hv_{_name}_mean"] = np.array(hv_x.mean())
        out[f"hv_{_name}_std"] = np.array(hv_x.std())
    return out


def _train_uniform(algo, xc):
    """Train a variant at the fixed uniform reference omega (used by ablations)."""
    uni = np.full(algo.env.M, 1.0 / algo.env.M, np.float32)
    algo._obs, _ = algo.env.reset()
    for _ in range(xc.n_iterations):
        algo.update(algo.collect_rollout(xc.horizon, omega=uni))
    return uni


def run_ablation(xc: ExperimentConfig) -> dict:
    """(C) final performance vs V (Lyapunov) and delta (trust region).

    Averaged over `ablation_seeds` seeds — single-seed ablations are dominated by
    run-to-run variance, which buries the V / delta trend (see experiments/test2).
    Stores the full constraint-cost vector so the plot can pick whichever
    constraint is actually binding.
    """
    n_seeds = max(1, xc.ablation_seeds)

    V_res, V_sd = [], []
    for V in xc.V_grid:
        rows = []
        for seed in range(n_seeds):
            algo, lyap = build_variant("lyapunov", xc, seed=seed, V=V)
            uni = _train_uniform(algo, xc)
            s, v, c, b = eval_policy(algo, uni, xc.eval_episodes)
            zmax = lyap.stability_report()["Z_max"].max()
            rows.append((s, c[0], c[1], c[2], b, float(zmax)))
        m, sd = np.mean(rows, axis=0), np.std(rows, axis=0)
        V_res.append((V, *m))          # (V, scalar, c1, c2, c3, backlog, zmax)
        V_sd.append((V, *sd))          # std in the same column layout (col 0 = V, std 0)

    d_res, d_sd = [], []
    for d in xc.delta_grid:
        rows = []
        for seed in range(n_seeds):
            algo, _ = build_variant("cpo", xc, seed=seed, delta=d)
            uni = _train_uniform(algo, xc)
            s, v, c, b = eval_policy(algo, uni, xc.eval_episodes)
            rows.append((s, algo.kl_report()["kl_max"]))
        m, sd = np.mean(rows, axis=0), np.std(rows, axis=0)
        d_res.append((d, *m))          # (delta, scalar, kl_max)
        d_sd.append((d, *sd))

    return {"V": np.array(V_res), "V_std": np.array(V_sd),
            "delta": np.array(d_res), "delta_std": np.array(d_sd),
            "n_seeds": np.array(n_seeds)}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(xc: ExperimentConfig | None = None,
         blocks=("convergence", "pareto", "ablation")) -> dict:
    xc = xc or ExperimentConfig()
    os.makedirs(xc.out_dir, exist_ok=True)
    print(f"PLTR-MA experiments | N={xc.n_servers} seeds={xc.n_seeds} "
          f"iters={xc.n_iterations} | blocks={blocks}")

    results: dict = {}
    if "convergence" in blocks:
        print("[A] convergence ablation ...")
        results["convergence"] = run_convergence(xc)
    if "pareto" in blocks:
        print("[B] Pareto-front sweep ...")
        results["pareto"] = run_pareto(xc)
    if "ablation" in blocks:
        print("[C] V / delta ablation ...")
        results["ablation"] = run_ablation(xc)

    # flatten to a savez-friendly dict
    flat = {}
    for blk, d in results.items():
        _flatten(blk, d, flat)
    np.savez(os.path.join(xc.out_dir, "results.npz"), **flat)
    with open(os.path.join(xc.out_dir, "config.json"), "w") as f:
        json.dump(asdict(xc), f, indent=2, default=str)
    print(f"saved -> {os.path.join(xc.out_dir, 'results.npz')}")
    return results


def _flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}/{k}", v, out)
    else:
        out[prefix] = np.asarray(obj)


if __name__ == "__main__":
    main()
