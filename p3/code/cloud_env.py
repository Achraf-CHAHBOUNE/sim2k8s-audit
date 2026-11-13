"""
cloud_env.py — Custom N=3 cloud resource-allocation environment for PLTR-MA.

Implements the Constrained Multi-Objective Markov Game (CMOMG) of Section V.A
of the conference paper. This is a multi-agent, vector-reward, constrained
environment built on the Gymnasium API conventions (reset/step), but with two
deliberate departures from the scalar single-agent Gym contract:

  1. step() returns a VECTOR reward of shape (M,) with M=3 objectives
     [ -latency, -energy, -SLA_violations ].  The MAPPO backbone scalarizes
     it with a preference vector omega (see pareto.py); the env itself stays
     preference-agnostic.
  2. The constraint costs c_k (K=3) are returned in info["cost"], shape (K,).
     The Lyapunov module (lyapunov.py) turns these into virtual queues.

Per-agent observation (9 dims, roughly normalized to [0,1]):
    [ cpu_util, mem_util, queue_backlog, energy,
      task_cpu, task_mem, task_proc, task_deadline_slack, has_task ]

Actions per agent (Discrete(3)):
    0 = accept   enqueue the incoming task on this server
    1 = defer    keep the task in the arrival buffer for one more step
    2 = migrate  hand the task to the currently least-loaded other server

There is deliberately NO "reject" action. With voluntary dropping removed, a
task leaves the system only by completing or by missing its deadline, which
forces genuine queue management instead of the degenerate "reject everything to
keep queues empty" policy (see experiments/test1 for that failure mode).

Honesty note: this is a custom analytical simulator, not CloudSim/Kubernetes.
That limitation is stated openly in the paper (Section VI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # Gymnasium is optional at runtime; we only use its spaces for metadata.
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
except Exception:  # pragma: no cover - fallback when gymnasium is unavailable
    _HAS_GYM = False


# --------------------------------------------------------------------------- #
# Interaction ledger (audit A3). Counts env.step() calls per phase, process-wide.
# Bookkeeping only: it never touches an RNG or the env state.
# --------------------------------------------------------------------------- #
LEDGER: dict = {}
_PHASE = ["train"]


class ledger_phase:
    """Context manager: attribute env steps taken inside it to `name`."""
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        _PHASE.append(self.name)
        return self

    def __exit__(self, *exc):
        _PHASE.pop()
        return False


# --------------------------------------------------------------------------- #
# Task and server data structures
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    """A single workload unit routed through the cluster."""

    cpu: float            # CPU demand (normalized units)
    mem: float            # memory footprint (normalized units)
    work: float           # total processing work required (in "work units")
    remaining: float      # remaining work not yet processed
    arrival_t: int        # step index at which the task entered the system
    deadline_t: int       # step index by which it must complete


@dataclass
class Server:
    """A heterogeneous compute node, one per agent."""

    cpu_cap: float        # CPU capacity
    mem_cap: float        # memory capacity
    speed: float          # work units processed per step (FIFO head task)
    queue_cap: int        # soft queue-length cap (Lyapunov backlog ceiling)
    running: list[Task] = field(default_factory=list)   # accepted, in-service
    arrivals: list[Task] = field(default_factory=list)   # waiting to be decided

    # --- derived quantities ------------------------------------------------ #
    def cpu_util(self) -> float:
        return sum(t.cpu for t in self.running) / self.cpu_cap

    def mem_util(self) -> float:
        return sum(t.mem for t in self.running) / self.mem_cap

    def backlog(self) -> int:
        return len(self.running)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class CloudAllocationEnv:
    """N-server constrained multi-objective cloud allocation Markov game."""

    OBS_DIM = 9
    N_ACTIONS = 3               # accept, defer, migrate (no voluntary reject)
    M = 3                       # objectives: latency, energy, SLA
    K = 3                       # constraints: capacity, queue-overflow, SLA-breach

    ACCEPT, DEFER, MIGRATE = 0, 1, 2

    def __init__(
        self,
        n_servers: int = 3,
        arrival_rate: float = 1.0,      # Poisson lambda (tasks/step, cluster-wide)
        horizon: int = 200,
        queue_cap: int = 15,
        sla_penalty: float = 5.0,
        seed: int | None = None,
        trace=None,                     # optional WorkloadTrace (trace.py); overrides Poisson
        contention: bool = False,       # P5 revision: overcommit slows service (default off)
        simultaneous_actions: bool = False,  # audit A4: decide on pre-step state (default off)
        reset_seed_offset: int = 0,     # calibration runs: separate evaluation episodes (default 0)
    ) -> None:
        self.simultaneous_actions = bool(simultaneous_actions)
        # Added to every explicit reset seed for the RNG only; the recorder keeps the nominal
        # seed (front segmentation at 30000 is unchanged). 0 = original episodes.
        self.reset_seed_offset = int(reset_seed_offset)
        self.N = n_servers
        # Contention model (off = original dynamics). When on, a server whose
        # utilization max(cpu, mem) exceeds 1 serves at speed / utilization, so
        # overcommitting no longer comes for free.
        self.contention = bool(contention)
        self.lam = arrival_rate
        # When a trace is supplied, arrivals are REPLAYED from it and `arrival_rate`
        # is ignored; the env dynamics are otherwise identical (see trace.py).
        self.trace = trace
        self.horizon = horizon
        self.queue_cap = queue_cap
        # Weight on unserved tasks in the SLA objective. Without a strong penalty
        # the policy exploits "latency = backlog" by rejecting every task to keep
        # queues empty; this makes serving worthwhile. (See experiments/test1.)
        self.sla_penalty = sla_penalty
        self.rng = np.random.default_rng(seed)

        # Heterogeneous server fleet (capacities/speeds increase with index).
        base_cpu = np.linspace(1.0, 2.0, self.N)
        base_mem = np.linspace(1.0, 2.0, self.N)
        base_spd = np.linspace(1.0, 1.5, self.N)
        self._fleet_spec = list(zip(base_cpu, base_mem, base_spd))

        # Energy model constants.
        self.p_idle = 0.1
        self.p_dyn = 0.9
        self.migrate_energy = 0.05

        # Task-generation ranges.
        self.task_cpu_range = (0.1, 0.5)
        self.task_mem_range = (0.1, 0.5)
        self.task_work_range = (1.0, 5.0)
        self.task_slack_range = (10, 30)    # deadline = arrival + slack (longer => tasks queue)
        self._max_work = self.task_work_range[1]
        self._max_slack = self.task_slack_range[1]

        # Gymnasium spaces (informational; the env runs without gymnasium too).
        if _HAS_GYM:
            self.single_observation_space = spaces.Box(
                low=0.0, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
            )
            self.single_action_space = spaces.Discrete(self.N_ACTIONS)
            self.action_space = spaces.MultiDiscrete([self.N_ACTIONS] * self.N)
            self.observation_space = spaces.Box(
                low=0.0, high=np.inf, shape=(self.N, self.OBS_DIM), dtype=np.float32
            )

        self.t = 0
        self.servers: list[Server] = []
        self._ep_metrics: dict[str, Any] = {}
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    # Optional cost recorder (P5 constraint study). None = off = original
    # behaviour. When a list, every finished episode appends
    # (reset seed, mean per-step cost vector c_k). Pure bookkeeping: it reads
    # the episode metrics the env already keeps and never touches the RNG.
    cost_log = None

    # Per-episode recorded fields: the mean legacy costs (c1, c2, c3) followed by `extra`.
    # Definitions (audit A5, 2026-10-19):
    #   c1, c2, c3            legacy per-step costs, episode means. c3 is the SLA COST PROXY
    #                         expired_this_step / max(1, tasks still in system) -- NOT a
    #                         probability and NOT a deadline-miss rate (it can exceed 1).
    #   max_c1_step           max over steps of c1 (post-service sample)
    #   max_overcommit_post   max over steps and servers of max(cpu,mem) reservation - 1,
    #                         sampled AFTER service (legacy sampling point)
    #   completed, expired    tasks completed / dropped at their deadline in the episode
    #   latency_completed     mean (completion step - arrival step) over completed tasks;
    #                         NaN if none completed (completion-only: censored tasks excluded)
    #   util_cpu_post         mean over steps of mean-over-servers CPU reservation, post-service
    #   arrived               tasks that arrived in the episode (incl. the reset batch)
    #   unfinished_tasks/work tasks (and remaining work) still in the system at truncation
    #   peak_res_max_pre      max over steps of the peak server reservation max(cpu,mem),
    #                         sampled after admission and BEFORE expiry/service
    #   peak_res_p95_pre      95th percentile over steps of that per-step peak
    #   frac_steps_over_pre   fraction of steps whose pre-service peak reservation exceeds 1
    #   ret_latency/energy/sla  episode objective returns (sum of the reward vector)
    LOG_FIELDS = ("c1", "c2", "c3", "max_c1_step", "max_overcommit_post", "completed", "expired",
                  "latency_completed", "util_cpu_post", "arrived", "unfinished_tasks",
                  "unfinished_work", "peak_res_max_pre", "peak_res_p95_pre", "frac_steps_over_pre",
                  "ret_latency", "ret_energy", "ret_sla")

    def _new_log_extra(self) -> dict:
        return {"max_c1": 0.0, "max_over": 0.0, "util_sum": 0.0, "arrived": 0,
                "peaks": [], "ret": np.zeros(self.M)}

    def _log_episode(self) -> None:
        if (self.cost_log is not None and getattr(self, "_ep_metrics", None) is not None
                and self.t > 0):
            m = self._ep_metrics
            x = getattr(self, "_log_extra", None) or self._new_log_extra()
            peaks = np.asarray(x["peaks"], float) if x["peaks"] else np.zeros(1)
            unfinished = [t for srv in self.servers for t in srv.running + srv.arrivals]
            extra = np.array([
                x["max_c1"], x["max_over"], m["completed"], m["sla_violations"],
                (m["latency_sum"] / m["completed"]) if m["completed"] else np.nan,
                x["util_sum"] / self.t, x["arrived"], len(unfinished),
                float(sum(t.remaining for t in unfinished)),
                float(peaks.max()), float(np.percentile(peaks, 95)), float((peaks > 1.0).mean()),
                *x["ret"]], dtype=float)
            self.cost_log.append((self._log_seed, m["cost_sum"] / self.t, extra))

    def flush_cost_log(self) -> list:
        """Finalize the running episode and return the recorded episodes."""
        self._log_episode()
        self.t = 0
        return list(self.cost_log or [])

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self._log_episode()
        self._log_seed = seed
        self._log_extra = self._new_log_extra()
        if seed is not None:
            self.rng = np.random.default_rng(seed + getattr(self, "reset_seed_offset", 0))
        self.t = 0
        self.servers = [
            Server(cpu_cap=c, mem_cap=m, speed=s, queue_cap=self.queue_cap)
            for (c, m, s) in self._fleet_spec
        ]
        self._ep_metrics = {
            "completed": 0,
            "latency_sum": 0.0,
            "energy_sum": 0.0,
            "sla_violations": 0,
            "queue_trace": [],          # list of per-step total backlog
            "cost_sum": np.zeros(self.K),
        }
        # Seed an initial batch of arrivals so step 0 has decisions to make.
        if self.trace is not None:
            self.trace.reset(self.rng)      # pick this episode's trace window
        self._spawn_arrivals()
        return self._build_obs(), self._info(np.zeros(self.K), reward=np.zeros(self.M))

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, actions) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        actions = np.asarray(actions, dtype=int).reshape(self.N)
        self.t += 1
        LEDGER[_PHASE[-1]] = LEDGER.get(_PHASE[-1], 0) + 1   # audit A3 (bookkeeping)

        sla_step = 0          # SLA violations incurred this step
        migrate_count = 0

        # --- 1. apply each agent's action to its head incoming task -------- #
        if self.simultaneous_actions:
            migrate_count = self._apply_actions_simultaneous(actions)
        for i, srv in enumerate(self.servers if not self.simultaneous_actions else ()):
            if not srv.arrivals:
                continue                      # no task to decide on -> no-op
            task = srv.arrivals[0]
            a = int(actions[i])

            if a == self.ACCEPT:
                srv.arrivals.pop(0)
                srv.running.append(task)
            elif a == self.DEFER:
                pass                          # leave it at the head, re-decide next step
            elif a == self.MIGRATE:
                srv.arrivals.pop(0)
                j = self._least_loaded_other(i)
                self.servers[j].arrivals.append(task)
                migrate_count += 1

        if self.cost_log is not None:                    # recorder: pre-service reservation peak
            self._log_extra["peaks"].append(max(max(s.cpu_util(), s.mem_util()) for s in self.servers))

        # --- 2. drop tasks whose deadline has already passed -------------- #
        sla_step += self._expire_overdue()

        # --- 3. process running queues (FIFO, one head task per server) --- #
        latency_completed = self._process()

        # --- 4. compute objectives, costs, and reward --------------------- #
        backlog_total = sum(s.backlog() for s in self.servers)
        energy_step = self._energy(migrate_count)

        # latency objective (Little's law): total number of tasks currently in
        # the system, counting BOTH those running and those still waiting in the
        # arrival buffer. This is the key design point that makes the env
        # non-degenerate: a task can only leave the count by completing (good) or
        # expiring (penalised via SLA). With no reject action and waiting tasks
        # still counted, "defer everything" no longer hides tasks — the only way
        # to lower this penalty is to actually serve them. (See experiments/test1.)
        in_system = sum(s.backlog() + len(s.arrivals) for s in self.servers)
        latency_step = in_system

        reward = np.array(
            [-latency_step, -energy_step, -self.sla_penalty * float(sla_step)],
            dtype=np.float32,
        )

        cost = self._costs(sla_step)

        # --- 5. accounting + spawn next arrivals -------------------------- #
        self._ep_metrics["energy_sum"] += energy_step
        self._ep_metrics["sla_violations"] += sla_step
        self._ep_metrics["queue_trace"].append(backlog_total)
        self._ep_metrics["cost_sum"] += cost
        if self.cost_log is not None:                    # recorder only; no RNG, no state
            x = self._log_extra
            x["max_c1"] = max(x["max_c1"], float(cost[0]))
            x["max_over"] = max(x["max_over"], max(max(s.cpu_util(), s.mem_util()) - 1.0
                                                   for s in self.servers))
            x["util_sum"] += float(np.mean([s.cpu_util() for s in self.servers]))
            x["ret"] += reward

        self._spawn_arrivals()

        terminated = False
        truncated = self.t >= self.horizon
        return self._build_obs(), reward, terminated, truncated, self._info(cost, reward)

    # ------------------------------------------------------------------ #
    # Internal dynamics helpers
    # ------------------------------------------------------------------ #
    def _apply_actions_simultaneous(self, actions) -> int:
        """Audit A4: all agents act on the pre-step state. Targets of migrations come from
        the pre-step backlogs; migrated tasks join the END of their destination's buffer
        after every decision, so none can be accepted in the step it moves."""
        moves = []
        targets = {i: self._least_loaded_other(i) for i, srv in enumerate(self.servers)
                   if srv.arrivals and int(actions[i]) == self.MIGRATE}
        for i, srv in enumerate(self.servers):
            if not srv.arrivals:
                continue
            a = int(actions[i])
            if a == self.ACCEPT:
                srv.running.append(srv.arrivals.pop(0))
            elif a == self.MIGRATE:
                moves.append((targets[i], srv.arrivals.pop(0)))
        for j, task in moves:
            self.servers[j].arrivals.append(task)
        return len(moves)

    def _spawn_arrivals(self) -> None:
        """New arrivals: replayed from the trace if one is set, else Poisson(lambda).
        Each new task is routed to a uniformly random server (initial placement;
        the agents then decide accept/defer/migrate)."""
        if self.cost_log is not None:                    # recorder: count arrivals
            before = sum(len(s.arrivals) for s in self.servers)
            self._spawn_arrivals_impl()
            self._log_extra["arrived"] += sum(len(s.arrivals) for s in self.servers) - before
            return
        self._spawn_arrivals_impl()

    def _spawn_arrivals_impl(self) -> None:
        if self.trace is not None:
            for (cpu, mem, work, slack) in self.trace.tasks_at(self.t):
                task = Task(cpu=cpu, mem=mem, work=work, remaining=work,
                            arrival_t=self.t, deadline_t=self.t + int(slack))
                self.servers[int(self.rng.integers(self.N))].arrivals.append(task)
            return
        n_new = self.rng.poisson(self.lam)
        for _ in range(int(n_new)):
            cpu = self.rng.uniform(*self.task_cpu_range)
            mem = self.rng.uniform(*self.task_mem_range)
            work = self.rng.uniform(*self.task_work_range)
            slack = int(self.rng.integers(self.task_slack_range[0],
                                          self.task_slack_range[1] + 1))
            task = Task(
                cpu=cpu, mem=mem, work=work, remaining=work,
                arrival_t=self.t, deadline_t=self.t + slack,
            )
            j = int(self.rng.integers(self.N))
            self.servers[j].arrivals.append(task)

    def _least_loaded_other(self, i: int) -> int:
        loads = [s.backlog() if k != i else np.inf for k, s in enumerate(self.servers)]
        return int(np.argmin(loads))

    def _expire_overdue(self) -> int:
        """Drop tasks (queued or waiting) whose deadline has passed. Returns count."""
        expired = 0
        for srv in self.servers:
            keep_run = []
            for tsk in srv.running:
                if self.t > tsk.deadline_t:
                    expired += 1
                else:
                    keep_run.append(tsk)
            srv.running = keep_run

            keep_arr = []
            for tsk in srv.arrivals:
                if self.t > tsk.deadline_t:
                    expired += 1
                else:
                    keep_arr.append(tsk)
            srv.arrivals = keep_arr
        return expired

    def _process(self) -> float:
        """Advance FIFO head task of each server by its speed. Return completion latency."""
        latency_completed = 0.0
        for srv in self.servers:
            budget = srv.speed
            if self.contention:
                budget /= max(1.0, srv.cpu_util(), srv.mem_util())
            while budget > 1e-9 and srv.running:
                head = srv.running[0]
                step_work = min(head.remaining, budget)
                head.remaining -= step_work
                budget -= step_work
                if head.remaining <= 1e-9:
                    srv.running.pop(0)
                    latency = self.t - head.arrival_t
                    latency_completed += latency
                    self._ep_metrics["completed"] += 1
                    self._ep_metrics["latency_sum"] += latency
                else:
                    break  # head not finished; rest of budget is lost this step
        return latency_completed

    def _energy(self, migrate_count: int) -> float:
        e = 0.0
        for srv in self.servers:
            util = min(srv.cpu_util(), 2.0)
            e += self.p_idle + self.p_dyn * util
        e += self.migrate_energy * migrate_count
        return e

    def _costs(self, sla_step: int) -> np.ndarray:
        """Per-step constraint costs c_k (capacity, queue-overflow, SLA-breach)."""
        # c1: capacity violation — mean overcommit beyond 100% utilization.
        cap = np.mean([
            max(0.0, max(s.cpu_util(), s.mem_util()) - 1.0) for s in self.servers
        ])
        # c2: queue-overflow risk — mean backlog beyond 80% of the queue cap.
        soft = 0.8 * self.queue_cap
        ovf = np.mean([
            max(0.0, s.backlog() - soft) / self.queue_cap for s in self.servers
        ])
        # c3: SLA-breach probability this step.
        active = sum(s.backlog() + len(s.arrivals) for s in self.servers)
        sla_prob = sla_step / max(1, active)
        return np.array([cap, ovf, sla_prob], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Observation / state construction
    # ------------------------------------------------------------------ #
    def _build_obs(self) -> np.ndarray:
        obs = np.zeros((self.N, self.OBS_DIM), dtype=np.float32)
        for i, srv in enumerate(self.servers):
            has_task = 1.0 if srv.arrivals else 0.0
            if srv.arrivals:
                tsk = srv.arrivals[0]
                t_cpu, t_mem = tsk.cpu, tsk.mem
                t_work = tsk.remaining / self._max_work
                t_slack = max(0, tsk.deadline_t - self.t) / self._max_slack
            else:
                t_cpu = t_mem = t_work = t_slack = 0.0
            energy_i = (self.p_idle + self.p_dyn * min(srv.cpu_util(), 2.0)) / (
                self.p_idle + self.p_dyn * 2.0
            )
            obs[i] = [
                min(srv.cpu_util(), 2.0) / 2.0,
                min(srv.mem_util(), 2.0) / 2.0,
                min(srv.backlog(), self.queue_cap) / self.queue_cap,
                energy_i,
                t_cpu, t_mem, t_work, t_slack, has_task,
            ]
        return obs

    def global_state(self) -> np.ndarray:
        """Flat joint observation for the centralized critic (CTDE training)."""
        return self._build_obs().reshape(-1)

    # ------------------------------------------------------------------ #
    # Info / metrics
    # ------------------------------------------------------------------ #
    def _info(self, cost: np.ndarray, reward: np.ndarray) -> dict:
        return {
            "cost": cost.astype(np.float64),               # shape (K,)
            "queue_lengths": np.array([s.backlog() for s in self.servers]),
            "total_backlog": int(sum(s.backlog() for s in self.servers)),
            "step": self.t,
        }

    def episode_summary(self) -> dict:
        m = self._ep_metrics
        completed = max(1, m["completed"])
        qtrace = np.array(m["queue_trace"]) if m["queue_trace"] else np.array([0])
        return {
            "completed_tasks": m["completed"],
            "mean_latency": m["latency_sum"] / completed,
            "total_energy": m["energy_sum"],
            "sla_violations": m["sla_violations"],
            "mean_backlog": float(qtrace.mean()),
            "max_backlog": int(qtrace.max()),
            "mean_cost": m["cost_sum"] / max(1, self.t),     # avg per-step c_k
        }


__all__ = ["CloudAllocationEnv", "Task", "Server", "LEDGER", "ledger_phase"]
