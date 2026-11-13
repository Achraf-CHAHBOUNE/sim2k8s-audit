"""
heuristics.py — NON-LEARNING baseline schedulers for P3 ("is RL needed at all?").

Two hand-designed controllers that act in CloudAllocationEnv with NO training:

  heur_leastloaded  LeastLoadedPolicy(env, tau)
      Per agent, per step, on the head task of its arrival buffer:
        * ACCEPT  if the local server's projected utilization after taking the
                  task, max(cpu, mem), is <= tau;
        * MIGRATE else, if the peer the env would route the task to
                  (env._least_loaded_other(i), i.e. the other server with the
                  smallest running backlog) could take it within tau AND that
                  peer is currently less utilized than the local server by at
                  least `margin`;
        * DEFER   otherwise.
      tau is the swept parameter (tau > 1 deliberately allows overcommit,
      which the env permits and charges through the capacity cost c1).

  heur_dpp          DriftPlusPenaltyPolicy(env, V)
      Online drift-plus-penalty (Neely 2010), greedy per step. Each agent picks
      the action minimising

          Delta_hat(a)  -  V * omega^T r_hat(a)

      where r_hat(a) is the (M,) one-step vector reward and Delta_hat(a) the
      one-step change of the quadratic Lyapunov function

          L = 1/2 sum_i (S_i^2 + A_i^2)  +  1/2 sum_k Z_k^2,

      S_i = running backlog of server i, A_i = its arrival-buffer length (both
      task counts, all servers), Z_k = the K virtual queues of lyapunov.py,
      Z_k <- max(0, Z_k + c_k - d_k) with d_k = 0 as in LyapunovTracker.
      The physical queue terms make the rule a backpressure rule (accept when
      the buffer is longer than the service queue, migrate when the target
      buffer is shorter); the Z terms carry the constraint history.

      HOW THE ONE-STEP EFFECT IS ESTIMATED (stated approximation):
        - EXACT env dynamics, not a model: for each candidate action a of agent
          i, the environment is cloned (`clone_env`) and the clone is stepped
          once with env.step(); r_hat, the realised costs c_hat and the
          post-step queues are read from the clone. Nothing is learned.
        - PER-AGENT GREEDY WITH OTHERS AT A DEFAULT (Jacobi): while agent i
          evaluates its 3 actions, every other agent's action is held at DEFER
          (a no-op on their head task). All agents then commit their choices
          simultaneously. Joint effects of simultaneous actions (e.g. two
          agents migrating to the same peer in the same step) are therefore
          NOT seen by the evaluation. This is an approximation of the joint
          argmin over 3^N actions, which is intractable at N=10+.
        - Common random numbers: the clone carries a copy of the env RNG, so
          the arrivals spawned at the end of the simulated step are identical
          for every candidate action; differences between actions come only
          from the action itself. The real env's RNG is never advanced.
        - Strictly one step (myopic): consequences beyond t+1 (e.g. a deferred
          task expiring 10 steps later) enter only through the queues and the
          virtual queues, as in the standard drift-plus-penalty argument.
      omega defaults to the uniform preference (the reference preference used
      by train.fit); V is the swept parameter.

  heur_local / heur_local_urgent   LocalOnlyPolicy / LocalUrgentPolicy(env, tau)
      INFORMATION-MATCHED comparators: they read only the agent's own 9-dim
      observation (what a decentralized PLTR-MA actor sees), never peer state,
      env internals or a simulator; accept-or-defer only (no MIGRATE).
      The two heuristics above read the global cluster state (and DPP an exact
      simulator), so they are NOT information-matched to the learned actors.

A heuristic yields ONE operating point per parameter setting. `heuristic_front`
sweeps the parameter and evaluates every setting with train.eval_policy, so the
resulting set of mean vector returns is directly comparable (via
pareto.hypervolume on a shared box) to the learned fronts.

Both policies expose the attributes eval_policy needs: `.env` and
`.act(obs, cond=None) -> (actions, None)`.
"""

from __future__ import annotations

import copy

import numpy as np

from cloud_env import CloudAllocationEnv, Server, ledger_phase
from lyapunov import LyapunovTracker

HEURISTICS = ("heur_leastloaded", "heur_dpp", "heur_local", "heur_local_urgent")

# Default sweep grids. tau: utilization threshold (1.0 = full capacity);
# V: four orders of magnitude. On N=3 synthetic the controller is saturated
# (pure backpressure, identical returns) for V <~ 3 and saturated the other way
# (energy-only, nearly everything expires) by V ~ 1000; the trade-off lives in
# between, so the grid is denser there.
TAU_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)
V_GRID = (0.1, 1.0, 3.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0, 1000.0)

# Seed base of the learned-front protocol (ParetoMAPPO.evaluate_omega uses
# env.reset(seed=30000 + e)). train.eval_policy defaults to 90000, which is what
# the fixed-weight baseline used; pass seed_base explicitly to choose.
FRONT_SEED_BASE = 30000


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _util_after(srv: Server, task) -> float:
    """Projected utilization max(cpu, mem) of `srv` if it accepted `task`."""
    cpu = (sum(t.cpu for t in srv.running) + task.cpu) / srv.cpu_cap
    mem = (sum(t.mem for t in srv.running) + task.mem) / srv.mem_cap
    return max(cpu, mem)


def _util(srv: Server) -> float:
    return max(srv.cpu_util(), srv.mem_util())


def clone_env(env: CloudAllocationEnv) -> CloudAllocationEnv:
    """Independent copy of the env's MUTABLE state for one-step look-ahead.

    Equivalent to copy.deepcopy(env) for the purpose of env.step() (verified in
    test_heuristics.py) but shares the read-only parts (trace table, fleet spec,
    gym spaces), which makes it much cheaper on the trace environment.
    """
    sim = copy.copy(env)
    sim.servers = [
        Server(cpu_cap=s.cpu_cap, mem_cap=s.mem_cap, speed=s.speed,
               queue_cap=s.queue_cap,
               running=[copy.copy(t) for t in s.running],
               arrivals=[copy.copy(t) for t in s.arrivals])
        for s in env.servers
    ]
    sim.rng = copy.deepcopy(env.rng)
    # audit A2: the look-ahead copy must not share the real recorder (cost_log list and
    # _log_extra dict were shared by the shallow copy, so hypothetical steps wrote into
    # the real episode's recorded statistics). Recording is off in the copy.
    sim.cost_log = None
    sim._log_extra = None
    m = env._ep_metrics
    sim._ep_metrics = {**m, "queue_trace": [], "cost_sum": m["cost_sum"].copy()}
    return sim


# --------------------------------------------------------------------------- #
# 1. Least-loaded threshold heuristic
# --------------------------------------------------------------------------- #
class LeastLoadedPolicy:
    """Accept below utilization tau, else migrate to a less-loaded peer, else defer."""

    name = "heur_leastloaded"

    def __init__(self, env: CloudAllocationEnv, tau: float, margin: float = 0.1):
        self.env = env
        self.tau = float(tau)
        self.margin = float(margin)

    def act(self, obs=None, cond=None):
        env = self.env
        a = np.full(env.N, env.DEFER, dtype=int)
        for i, srv in enumerate(env.servers):
            if not srv.arrivals:
                continue
            task = srv.arrivals[0]
            if _util_after(srv, task) <= self.tau:
                a[i] = env.ACCEPT
                continue
            if env.N > 1:
                j = env._least_loaded_other(i)      # where MIGRATE would send it
                peer = env.servers[j]
                if (_util_after(peer, task) <= self.tau
                        and _util(peer) + self.margin <= _util(srv)):
                    a[i] = env.MIGRATE
        return a, None


# --------------------------------------------------------------------------- #
# 2. Online drift-plus-penalty controller
# --------------------------------------------------------------------------- #
class DriftPlusPenaltyPolicy:
    """Greedy one-step drift-plus-penalty; see module docstring for the
    approximation (exact cloned one-step dynamics, per-agent Jacobi greedy with
    the other agents held at DEFER)."""

    name = "heur_dpp"

    def __init__(self, env: CloudAllocationEnv, V: float, omega=None,
                 d=(0.0, 0.0, 0.0)):
        self.env = env
        self.V = float(V)
        self.omega = (np.full(env.M, 1.0 / env.M) if omega is None
                      else np.asarray(omega, float))
        self.lyap = LyapunovTracker(K=env.K, d=tuple(d), V=self.V)
        self._last_cost_sum = np.zeros(env.K)

    # --- virtual queues follow the REAL costs of the real env ------------- #
    def _sync_virtual_queues(self) -> None:
        env = self.env
        if env.t == 0:                       # new episode (reset sets t = 0)
            self.lyap.reset()
            self._last_cost_sum = np.zeros(env.K)
            return
        cs = env._ep_metrics["cost_sum"]
        c_prev = cs - self._last_cost_sum    # exact cost of the previous step
        self._last_cost_sum = cs.copy()
        self.lyap.advance(c_prev)

    @staticmethod
    def _phys_L(env) -> float:
        return 0.5 * sum(len(s.running) ** 2 + len(s.arrivals) ** 2
                         for s in env.servers)

    def _virt_L_after(self, cost) -> float:
        Z1 = np.maximum(0.0, self.lyap.Z + np.asarray(cost, float) - self.lyap.d)
        return 0.5 * float(np.dot(Z1, Z1))

    def score(self, joint) -> float:
        """Drift-plus-penalty of a joint action, from one cloned env step.
        The pre-step L is common to all candidates and is omitted."""
        sim = clone_env(self.env)
        with ledger_phase("lookahead"):              # simulator-as-model steps (audit A3)
            _, r, _, _, info = sim.step(joint)
        drift = self._phys_L(sim) + self._virt_L_after(info["cost"])
        return drift - self.V * float(np.dot(self.omega, r))

    def act(self, obs=None, cond=None):
        env = self.env
        self._sync_virtual_queues()
        default = np.full(env.N, env.DEFER, dtype=int)
        a = default.copy()
        cands = ((env.ACCEPT, env.DEFER, env.MIGRATE) if env.N > 1
                 else (env.ACCEPT, env.DEFER))
        for i, srv in enumerate(env.servers):
            if not srv.arrivals:
                continue                      # action is a no-op; keep DEFER
            best, best_s = env.DEFER, np.inf
            for c in cands:                   # ties -> first in (ACCEPT, DEFER, MIGRATE)
                joint = default.copy()
                joint[i] = c
                s = self.score(joint)
                if s < best_s - 1e-12:
                    best, best_s = c, s
            a[i] = best
        return a, None


# --------------------------------------------------------------------------- #
# 3. Information-matched local-only rules (same 9-dim obs as a PLTR-MA actor)
# --------------------------------------------------------------------------- #
def _local_util_after(o) -> float:
    """Utilization proxy after accepting the head task, from the 9-dim local
    observation ONLY:  max(2*o[0] + o[4], 2*o[1] + o[5]).

    o[0], o[1] are cpu/mem utilization / 2 (env clips at 2.0); o[4], o[5] are
    the head task's raw cpu/mem demand. The server's capacity is NOT in the
    observation, so the task's share is added as if capacity were 1.0. Since
    every capacity in the fleet is >= 1, this OVER-estimates the post-accept
    utilization (exact on server 0, conservative on the larger servers). No
    attempt is made to recover capacity from anything outside the observation.
    """
    return max(2.0 * float(o[0]) + float(o[4]), 2.0 * float(o[1]) + float(o[5]))


class LocalOnlyPolicy:
    """ACCEPT iff the local post-accept utilization proxy <= tau, else DEFER.

    Uses ONLY obs[i] (the agent's own 9-dim observation, as a decentralized
    PLTR-MA actor sees it): no peer state, no env internals, no simulator.
    Never MIGRATEs, because picking a target needs peer state. `self.env` is
    held only because train.eval_policy reads env.M / env.K and steps the env;
    act() reads nothing from it but N."""

    name = "heur_local"

    def __init__(self, env: CloudAllocationEnv, tau: float):
        self.env = env
        self.tau = float(tau)

    def _rule(self, o) -> bool:
        return _local_util_after(o) <= self.tau

    def act(self, obs, cond=None):
        obs = np.asarray(obs, dtype=np.float64)
        a = np.full(obs.shape[0], CloudAllocationEnv.DEFER, dtype=int)
        for i, o in enumerate(obs):
            if o[8] > 0.5 and self._rule(o):          # o[8] = has_task
                a[i] = CloudAllocationEnv.ACCEPT
        return a, None


class LocalUrgentPolicy(LocalOnlyPolicy):
    """LocalOnly + deadline urgency, still from the 9-dim observation only:
    ACCEPT iff  proxy utilization <= tau  OR  normalized slack o[7] <= s0,
    else DEFER.

    Rationale (fixed a priori, not tuned on evaluation episodes): DEFER never
    lets a task leave except by expiring, which costs sla_penalty = 5 on the SLA
    objective; once its remaining slack is short, deferring further mostly
    converts it into an SLA violation, so it is accepted even above tau and
    the server is given the chance to finish it. s0 = 1/3 (10 steps at the
    env's normalization of 30) equals the SHORTEST initial slack the synthetic
    generator draws, i.e. "at most as much time left as the tightest new task".
    tau remains the single swept parameter."""

    name = "heur_local_urgent"

    def __init__(self, env: CloudAllocationEnv, tau: float, s0: float = 1.0 / 3.0):
        super().__init__(env, tau)
        self.s0 = float(s0)

    def _rule(self, o) -> bool:
        return _local_util_after(o) <= self.tau or float(o[7]) <= self.s0


# --------------------------------------------------------------------------- #
# Fronts from parameter sweeps
# --------------------------------------------------------------------------- #
def make_policy(name: str, env: CloudAllocationEnv, param: float):
    if name == "heur_leastloaded":
        return LeastLoadedPolicy(env, tau=param)
    if name == "heur_dpp":
        return DriftPlusPenaltyPolicy(env, V=param)
    if name == "heur_local":
        return LocalOnlyPolicy(env, tau=param)
    if name == "heur_local_urgent":
        return LocalUrgentPolicy(env, tau=param)
    raise ValueError(name)


def default_grid(name: str) -> tuple:
    return {"heur_leastloaded": TAU_GRID, "heur_dpp": V_GRID,
            "heur_local": TAU_GRID, "heur_local_urgent": TAU_GRID}[name]


def evaluate_setting(name: str, env: CloudAllocationEnv, param: float,
                     n_ep: int, seed_base: int = FRONT_SEED_BASE) -> dict:
    """One operating point, evaluated with train.eval_policy (same episode
    loop, same reset(seed=seed_base + e) convention)."""
    from train import eval_policy           # local import: keeps torch out of
                                            # module import for light callers
    pol = make_policy(name, env, param)
    uni = np.full(env.M, 1.0 / env.M, dtype=np.float32)
    s, vec, cost, backlog = eval_policy(pol, uni, n_ep, seed_base=seed_base)
    return {"scalar": s, "vec": vec, "cost": cost, "backlog": backlog}


def heuristic_front(name: str, env: CloudAllocationEnv, grid=None,
                    n_ep: int = 6, seed_base: int = FRONT_SEED_BASE):
    """Sweep the heuristic's parameter; return (params, returns (P, M),
    costs (P, K), backlogs (P,)). `returns` is the heuristic's front."""
    grid = tuple(default_grid(name) if grid is None else grid)
    rows = [evaluate_setting(name, env, p, n_ep, seed_base) for p in grid]
    return (np.array(grid, float),
            np.array([r["vec"] for r in rows]),
            np.array([r["cost"] for r in rows]),
            np.array([r["backlog"] for r in rows]))


__all__ = ["LeastLoadedPolicy", "DriftPlusPenaltyPolicy", "LocalOnlyPolicy",
           "LocalUrgentPolicy", "clone_env",
           "heuristic_front", "evaluate_setting", "make_policy", "default_grid",
           "HEURISTICS", "TAU_GRID", "V_GRID", "FRONT_SEED_BASE"]
