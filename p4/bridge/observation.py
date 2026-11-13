#!/usr/bin/env python3
"""observation.py — Kubernetes cluster state -> cloud_env observation vector.

Reproduces ``CloudAllocationEnv._build_obs`` (p3/code/cloud_env.py:349-370) from
a Kubernetes snapshot, so a policy trained in the simulator can be driven from a
real cluster without retraining and without silently changing the meaning of any
input.

The nine observation components, their Kubernetes sources and — most importantly —
which of them are NOT observable in Kubernetes are documented in ``MAPPING.md``.
``OBS_COMPONENTS`` below is the machine-readable version of that table, and
``test_bridge.py`` fails if the two drift apart.

Design rules this module obeys
------------------------------
1. **Requests, not usage.** cloud_env's ``cpu_util`` sums ``t.cpu`` over running
   tasks (cloud_env.py:77-84) — i.e. *declared demand*. The bridge therefore sums
   pod **requests**, never metrics-server usage. Feeding measured utilisation into
   slot 0 would put the policy off-distribution while looking like an
   improvement.
2. **Formula-preserving, not truth-preserving, for the energy slot.** Slot 3 in
   the simulator is a deterministic function of slot 0 (cloud_env.py:360-362),
   not a measurement. The bridge computes the same function. Real measured energy
   is reported as an *outcome*, never injected into the observation. See
   MAPPING.md O-4.
3. **Normalisation constants come from the simulator, capacities come from the
   cluster.** ``queue_cap``, ``max_work``, ``max_slack``, ``p_idle``, ``p_dyn``
   are copied from cloud_env. ``sim_cpu_cap``/``sim_mem_cap`` are derived from the
   cluster's *observed* node allocatable (cluster/nodes.json), never hard-coded.
4. **No silent substitution.** Anything that cannot be observed raises or is
   filled from an explicitly named synthesized source; see ``ObservationReport``.

Nothing in this module contacts a cluster. ``snapshot_from_kubernetes()`` is the
only function that would, and it is an optional, guarded import.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

# --------------------------------------------------------------------------- #
# Constants copied verbatim from p3/code/cloud_env.py. Keep in sync.
# --------------------------------------------------------------------------- #
OBS_DIM = 9                 # cloud_env.py:92
SIM_QUEUE_CAP = 15          # cloud_env.py:115 (default queue_cap)
SIM_MAX_WORK = 5.0          # cloud_env.py:137,139 task_work_range[1]
SIM_MAX_SLACK = 30.0        # cloud_env.py:138,140 task_slack_range[1]
SIM_P_IDLE = 0.1            # cloud_env.py:130
SIM_P_DYN = 0.9             # cloud_env.py:131
UTIL_CLAMP = 2.0            # cloud_env.py:364-365 min(util, 2.0) / 2.0

Category = Literal["observable", "approximable", "not-observable"]


@dataclass(frozen=True)
class ObsComponent:
    """One slot of the 9-dim per-agent observation vector."""

    index: int
    name: str
    sim_meaning: str
    k8s_source: str
    category: Category
    caveat: str


#: The mapping table. MAPPING.md row ids O-1..O-9 correspond to `index`.
OBS_COMPONENTS: tuple[ObsComponent, ...] = (
    ObsComponent(
        0, "cpu_util",
        "sum(t.cpu for t in running) / cpu_cap, clamped at 2.0 then halved",
        "sum of container CPU *requests* of non-terminal pods bound to the node, "
        "divided by node .status.allocatable.cpu",
        "approximable",
        "Values > 1.0 are unreachable: kube-scheduler/kubelet refuse a binding "
        "that would exceed allocatable, so the simulator's overcommit regime "
        "(and therefore constraint c1) cannot be entered without lying about "
        "requests. System/DaemonSet pods also consume allocatable and must be "
        "excluded by label or the baseline drifts.",
    ),
    ObsComponent(
        1, "mem_util",
        "sum(t.mem for t in running) / mem_cap, clamped at 2.0 then halved",
        "sum of container memory requests of non-terminal pods bound to the node, "
        "divided by node .status.allocatable.memory",
        "approximable",
        "Same unreachability as cpu_util, and worse: memory is incompressible, so "
        "the real consequence of over-subscription is an OOMKill, not the graceful "
        "degradation the simulator models.",
    ),
    ObsComponent(
        2, "queue_backlog",
        "len(srv.running) / queue_cap — tasks accepted onto the server",
        "count of pods bound to the node in Pending|Running with our run label",
        "approximable",
        "Semantics differ: in the simulator all `running` tasks hold cpu/mem but "
        "only the FIFO head is served; in Kubernetes every bound pod executes "
        "concurrently under CFS. Same integer, different dynamics. queue_cap=15 "
        "is a simulator constant with no Kubernetes counterpart (kubelet's "
        "max-pods default is 110).",
    ),
    ObsComponent(
        3, "energy",
        "(p_idle + p_dyn*min(cpu_util,2)) / (p_idle + p_dyn*2) — a closed-form "
        "function of slot 0, NOT a measurement",
        "recomputed from slot 0 with the simulator's own constants",
        "not-observable",
        "Per-node energy is not measurable in this topology at all: the three "
        "'nodes' are containers on one physical CPU package. Even with RAPL there "
        "is no per-container energy counter. The slot is reproduced by FORMULA to "
        "keep the policy in-distribution; measured energy is reported separately "
        "as an outcome. Reporting this slot as 'energy measured on Kubernetes' "
        "would be a fabrication.",
    ),
    ObsComponent(
        4, "task_cpu",
        "cpu demand of the head undecided arrival at this server",
        "CPU request of the head pending pod assigned to this agent, in sim units "
        "(millis / cpu_unit_millis)",
        "observable",
        "Observable only because we authored the pod from the trace; it is the "
        "declared request, which is exactly what the simulator uses.",
    ),
    ObsComponent(
        5, "task_mem",
        "memory footprint of the head undecided arrival",
        "memory request of the same pod, in sim units (bytes / mem_unit)",
        "observable",
        "As task_cpu.",
    ),
    ObsComponent(
        6, "task_proc",
        "task.remaining / 5.0 — remaining work of the head arrival",
        "pltr-ma.io/work-units annotation / 5.0",
        "approximable",
        "Exact for an *undecided* arrival, because cloud_env reads arrivals[0], "
        "which has never been processed, so remaining == work. It is NOT "
        "observable for a pod already running: a container's remaining work is "
        "not exposed by Kubernetes. The value is only known here because the "
        "replay PRESCRIBED it (a fixed instruction count); it is an input, not a "
        "measurement.",
    ),
    ObsComponent(
        7, "task_deadline_slack",
        "max(0, deadline_t - t) / 30.0 for the head arrival",
        "(pltr-ma.io/deadline-step annotation - current replay step) / 30.0",
        "approximable",
        "The number is available, but Kubernetes has no deadline semantics and "
        "will never act on it. Expiry (cloud_env._expire_overdue) has to be "
        "implemented by the replay controller; spec.activeDeadlineSeconds is NOT "
        "equivalent (it clocks from pod start, not from arrival).",
    ),
    ObsComponent(
        8, "has_task",
        "1.0 if this server's arrival buffer is non-empty",
        "1.0 if any pending pod carries pltr-ma.io/initial-agent == this agent",
        "not-observable",
        "Kubernetes has no per-node arrival buffer: an unbound pod belongs to no "
        "node. The simulator pre-routes each arrival to a uniformly random server "
        "BEFORE the agent decides (cloud_env.py:261). That routing is SYNTHESIZED "
        "by trace_to_pods.py and carried in an annotation. The signal is "
        "therefore an artefact of the simulator that the harness re-injects, not "
        "something read from the cluster.",
    ),
)

assert len(OBS_COMPONENTS) == OBS_DIM
assert [c.index for c in OBS_COMPONENTS] == list(range(OBS_DIM))


# --------------------------------------------------------------------------- #
# Snapshot data model — a pure-Python description of cluster state
# --------------------------------------------------------------------------- #
@dataclass
class UnitScale:
    """Kubernetes unit <-> simulator unit conversion.

    Must match the ScaleConfig used by replay/trace_to_pods.py, otherwise the
    observation is silently mis-scaled. `assert_consistent_with` checks that.
    """

    cpu_unit_millis: int = 1000
    mem_unit_bytes: int = 512 * 1024 * 1024      # 512 MiB per sim memory unit

    def cpu_to_sim(self, millis: float) -> float:
        return millis / self.cpu_unit_millis

    def mem_to_sim(self, nbytes: float) -> float:
        return nbytes / self.mem_unit_bytes

    def assert_consistent_with(self, cpu_unit_millis: int, mem_unit_mi: int) -> None:
        if (cpu_unit_millis, mem_unit_mi * 1024 * 1024) != (
                self.cpu_unit_millis, self.mem_unit_bytes):
            raise ValueError(
                "unit scale mismatch between the replay generator and the "
                f"observation bridge: generator=({cpu_unit_millis}m, {mem_unit_mi}Mi) "
                f"bridge=({self.cpu_unit_millis}m, {self.mem_unit_bytes}B)"
            )


@dataclass
class PodState:
    """One replayed task as Kubernetes sees it, plus the annotations we authored."""

    name: str
    cpu_request_millis: int
    mem_request_bytes: int
    work_units: float                 # pltr-ma.io/work-units (PRESCRIBED)
    arrival_step: int                 # pltr-ma.io/arrival-step
    deadline_step: int                # pltr-ma.io/deadline-step
    initial_agent: int                # pltr-ma.io/initial-agent (SYNTHESIZED)
    node_index: int | None = None     # None => still unbound (a simulator "arrival")
    phase: str = "Pending"            # Pending | Running | Succeeded | Failed
    #: Fraction of the prescribed work already done. NOT OBSERVABLE in
    #: Kubernetes; only ever non-None when the workload self-reports. Default
    #: None means "unknown", and the bridge then uses the full work, which is
    #: exactly what cloud_env does for an undecided arrival.
    remaining_work_units: float | None = None
    #: Position in this agent's arrival buffer. The simulator's buffer is a plain
    #: list in INSERTION order, and MIGRATE appends to the destination's TAIL
    #: (cloud_env.py:208), so buffer order is NOT arrival-time order once any
    #: migration has happened. Kubernetes holds no such ordering — the pending set
    #: is unordered from our side — so the replay controller must maintain this
    #: counter itself and bump it to the tail on every retarget. None means the
    #: counter is unavailable, and ClusterSnapshot.arrival_pods then falls back to
    #: (arrival_step, name), which is only faithful before the first migration.
    enqueue_seq: int | None = None
    #: Live-cluster only (None elsewhere). `status.reason` — the kubelet writes
    #: e.g. OutOfcpu / OutOfpods here when it REJECTS a pod after binding, which
    #: is how a refused ACCEPT actually surfaces (MAPPING.md A-1).
    reason: str | None = None
    #: Live-cluster only: container start / finish, unix seconds, from
    #: status.containerStatuses[0].state.{running,terminated}. 1 s granularity.
    started_at: float | None = None
    finished_at: float | None = None
    #: metadata.deletionTimestamp is set: the pod is going away (we deleted it).
    deleting: bool = False
    #: spec.schedulingGates is non-empty: kube-scheduler will not consider the pod
    #: (arm A1, protocol v1 §3.2). Dry-run backends never set it.
    gated: bool = False
    #: Live only (audit):
    #: metadata.uid; containerStatuses[0].state.waiting.reason (ErrImageNeverPull,
    #: ImagePullBackOff, ... -> infrastructure failure); the PodScheduled
    #: condition's lastTransitionTime (when the binding happened), unix s.
    uid: str | None = None
    waiting_reason: str | None = None
    scheduled_at: float | None = None

    @property
    def is_terminal(self) -> bool:
        return self.deleting or self.phase in ("Succeeded", "Failed")


@dataclass
class NodeState:
    """One k3d agent = one cloud_env Server."""

    name: str
    index: int
    cpu_allocatable_millis: int       # from .status.allocatable.cpu (OBSERVED)
    mem_allocatable_bytes: int        # from .status.allocatable.memory (OBSERVED)
    #: Requests already consumed by pods we do NOT own (kube-system, DaemonSets).
    #: Subtracted from allocatable so that a sim utilisation of 0 means "no
    #: replayed task on this node", matching the simulator's empty server.
    foreign_cpu_request_millis: int = 0
    foreign_mem_request_bytes: int = 0
    #: Live only: the node's Ready condition (True / False); None = not reported
    #: (dry-run backends, fakes). False is an infrastructure failure (§6.4).
    ready: bool | None = None

    def sim_cpu_cap(self, scale: UnitScale) -> float:
        return scale.cpu_to_sim(self.cpu_allocatable_millis - self.foreign_cpu_request_millis)

    def sim_mem_cap(self, scale: UnitScale) -> float:
        return scale.mem_to_sim(self.mem_allocatable_bytes - self.foreign_mem_request_bytes)


@dataclass
class ClusterSnapshot:
    """A cluster state at one replay step.

    NOT an atomic snapshot when built from a live cluster. See
    ``ObservationReport.staleness_warning`` and MAPPING.md row G-1.
    """

    step: int
    nodes: list[NodeState]
    pods: list[PodState]
    queue_cap: int = SIM_QUEUE_CAP
    scale: UnitScale = field(default_factory=UnitScale)
    #: Age, in seconds, of the oldest piece of information in this snapshot.
    #: None means unknown (e.g. a hand-built snapshot in a test).
    max_info_age_s: float | None = None

    # -- derived views ----------------------------------------------------- #
    def bound_pods(self, i: int) -> list[PodState]:
        """cloud_env's `srv.running`: accepted, not yet finished."""
        return [p for p in self.pods
                if p.node_index == i and not p.is_terminal]

    def arrival_pods(self, i: int) -> list[PodState]:
        """cloud_env's `srv.arrivals`: undecided tasks, in buffer (FIFO) order.

        Ordering is ours to maintain, and it is not free: Kubernetes' pending set
        is unordered from our point of view, and kube-scheduler's internal
        priority queue is not the simulator's FIFO. The simulator's buffer is
        insertion-ordered and MIGRATE appends to the destination's TAIL
        (cloud_env.py:208), so once any migration has occurred the buffer order
        differs from arrival-time order. We therefore prefer the explicit
        `enqueue_seq` the replay controller maintains, and only fall back to
        (arrival_step, name) when it is absent.
        """
        pend = [p for p in self.pods
                if p.node_index is None and p.initial_agent == i and not p.is_terminal]
        if pend and all(p.enqueue_seq is not None for p in pend):
            return sorted(pend, key=lambda p: (p.enqueue_seq, p.name))
        return sorted(pend, key=lambda p: (p.arrival_step, p.name))


@dataclass
class ObservationReport:
    """Everything the caller must know to interpret an observation honestly."""

    step: int
    synthesized_slots: tuple[int, ...]
    formula_slots: tuple[int, ...]
    staleness_warning: str | None
    clamped: dict[str, int]           # slot name -> how many agents hit the clamp
    refused_regime: dict[str, bool]   # e.g. {"cpu_util>1": False}

    def as_text(self) -> str:
        lines = [f"step {self.step}"]
        lines.append("  slots filled from a SYNTHESIZED source: "
                     + ", ".join(OBS_COMPONENTS[i].name for i in self.synthesized_slots))
        lines.append("  slots reproduced by FORMULA (not measured): "
                     + ", ".join(OBS_COMPONENTS[i].name for i in self.formula_slots))
        if self.clamped:
            lines.append("  clamped: " + ", ".join(f"{k}x{v}" for k, v in self.clamped.items()))
        if self.staleness_warning:
            lines.append("  " + self.staleness_warning)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The mapping itself
# --------------------------------------------------------------------------- #
def build_observation(snap: ClusterSnapshot,
                      max_info_age_budget_s: float = 1.0
                      ) -> tuple[np.ndarray, ObservationReport]:
    """Return ((N, 9) float32 observation, report).

    Numerically identical to ``CloudAllocationEnv._build_obs`` when the snapshot
    describes the same state — verified by ``test_bridge.py::
    test_matches_cloud_env_build_obs`` against the real simulator.
    """
    n = len(snap.nodes)
    obs = np.zeros((n, OBS_DIM), dtype=np.float32)
    clamped: dict[str, int] = {}
    over_one_util = False

    for node in sorted(snap.nodes, key=lambda x: x.index):
        i = node.index
        running = snap.bound_pods(i)
        arrivals = snap.arrival_pods(i)

        # --- O-1 / O-2: utilisation from REQUESTS over OBSERVED allocatable -- #
        cpu_cap = node.sim_cpu_cap(snap.scale)
        mem_cap = node.sim_mem_cap(snap.scale)
        if cpu_cap <= 0 or mem_cap <= 0:
            raise ValueError(
                f"node {node.name}: non-positive simulator capacity "
                f"(cpu_cap={cpu_cap}, mem_cap={mem_cap}). Foreign requests exceed "
                "allocatable; the node cannot host the replay."
            )
        cpu_sim = sum(snap.scale.cpu_to_sim(p.cpu_request_millis) for p in running)
        mem_sim = sum(snap.scale.mem_to_sim(p.mem_request_bytes) for p in running)
        cpu_util = cpu_sim / cpu_cap
        mem_util = mem_sim / mem_cap
        if cpu_util > 1.0 or mem_util > 1.0:
            over_one_util = True
        if cpu_util > UTIL_CLAMP:
            clamped["cpu_util"] = clamped.get("cpu_util", 0) + 1
        if mem_util > UTIL_CLAMP:
            clamped["mem_util"] = clamped.get("mem_util", 0) + 1

        # --- O-3: backlog ---------------------------------------------------- #
        backlog = len(running)
        if backlog > snap.queue_cap:
            clamped["queue_backlog"] = clamped.get("queue_backlog", 0) + 1

        # --- O-4: energy BY FORMULA (cloud_env.py:360-362) ------------------- #
        energy_i = (SIM_P_IDLE + SIM_P_DYN * min(cpu_util, UTIL_CLAMP)) / (
            SIM_P_IDLE + SIM_P_DYN * UTIL_CLAMP)

        # --- O-5..O-8: the head undecided arrival ---------------------------- #
        has_task = 1.0 if arrivals else 0.0
        if arrivals:
            head = arrivals[0]
            t_cpu = snap.scale.cpu_to_sim(head.cpu_request_millis)
            t_mem = snap.scale.mem_to_sim(head.mem_request_bytes)
            remaining = (head.work_units if head.remaining_work_units is None
                         else head.remaining_work_units)
            t_work = remaining / SIM_MAX_WORK
            t_slack = max(0, head.deadline_step - snap.step) / SIM_MAX_SLACK
        else:
            t_cpu = t_mem = t_work = t_slack = 0.0

        obs[i] = [
            min(cpu_util, UTIL_CLAMP) / UTIL_CLAMP,
            min(mem_util, UTIL_CLAMP) / UTIL_CLAMP,
            min(backlog, snap.queue_cap) / snap.queue_cap,
            energy_i,
            t_cpu, t_mem, t_work, t_slack, has_task,
        ]

    stale = None
    if snap.max_info_age_s is not None and snap.max_info_age_s > max_info_age_budget_s:
        stale = (f"STALE: oldest datum is {snap.max_info_age_s:.1f}s old, budget is "
                 f"{max_info_age_budget_s:.1f}s. The simulator's observation is an "
                 "atomic snapshot; this one is not (MAPPING.md G-1).")

    report = ObservationReport(
        step=snap.step,
        synthesized_slots=tuple(c.index for c in OBS_COMPONENTS
                                if c.index in (8,)),
        formula_slots=(3,),
        staleness_warning=stale,
        clamped=clamped,
        refused_regime={"cpu_util>1_or_mem_util>1": over_one_util},
    )
    return obs, report


def global_state(obs: np.ndarray) -> np.ndarray:
    """Flat joint observation for the centralized critic (cloud_env.py:372-374)."""
    return obs.reshape(-1)


def not_observable() -> list[ObsComponent]:
    """The components that cannot be read from Kubernetes. The P4 finding."""
    return [c for c in OBS_COMPONENTS if c.category == "not-observable"]


def approximable() -> list[ObsComponent]:
    return [c for c in OBS_COMPONENTS if c.category == "approximable"]


def directly_observable() -> list[ObsComponent]:
    return [c for c in OBS_COMPONENTS if c.category == "observable"]


# --------------------------------------------------------------------------- #
# Cluster-state constraint costs (cloud_env._costs, cloud_env.py:330-344)
# --------------------------------------------------------------------------- #
def constraint_costs(snap: ClusterSnapshot, sla_step: int) -> np.ndarray:
    """c_1..c_3 from a Kubernetes snapshot, same formulas as cloud_env._costs.

    c_1 (capacity violation) is structurally ~0 in Kubernetes: the API server
    refuses a binding that would exceed allocatable, so max(util) - 1 cannot go
    positive while requests are honest. Reporting "c_1 = 0, constraint satisfied"
    as a transfer result would be circular. This is recorded in MAPPING.md and
    must be stated in the paper.
    """
    caps, ovfs = [], []
    soft = 0.8 * snap.queue_cap
    for node in sorted(snap.nodes, key=lambda x: x.index):
        running = snap.bound_pods(node.index)
        cpu_util = sum(snap.scale.cpu_to_sim(p.cpu_request_millis)
                       for p in running) / node.sim_cpu_cap(snap.scale)
        mem_util = sum(snap.scale.mem_to_sim(p.mem_request_bytes)
                       for p in running) / node.sim_mem_cap(snap.scale)
        caps.append(max(0.0, max(cpu_util, mem_util) - 1.0))
        ovfs.append(max(0.0, len(running) - soft) / snap.queue_cap)
    active = sum(len(snap.bound_pods(nd.index)) + len(snap.arrival_pods(nd.index))
                 for nd in snap.nodes)
    sla_prob = sla_step / max(1, active)
    return np.array([float(np.mean(caps)), float(np.mean(ovfs)), sla_prob], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Optional live-cluster snapshot (guarded; never imported by the tests)
# --------------------------------------------------------------------------- #
def snapshot_from_kubernetes(step: int, namespace: str = "pltr-p4",
                             run_label: str = "run0",
                             agent_label: str = "pltr-ma.io/role=agent",
                             scale: UnitScale | None = None,
                             context: str | None = None,
                             api=None) -> ClusterSnapshot:
    """Build a ClusterSnapshot from a live cluster. Requires the `kubernetes` pkg
    unless a CoreV1Api-compatible `api` object is injected (tests do this).

    NOT an atomic read: nodes and pods are separate API calls, and a pod's
    .status lags the kubelet by up to the node status update period. The wall
    time the reads took is recorded in ``max_info_age_s``; it is a LOWER bound
    on staleness, because the kubelet's own reporting lag is not observable.

    Fixed 2026-10-04 (none of this had ever run):
      * foreign requests are summed over ALL namespaces. The first version only
        looked inside our own namespace, so kube-system pods (CoreDNS lands on an
        agent because the server is tainted) were never subtracted;
      * a pod with metadata.deletionTimestamp is terminal (we deleted it);
      * status.reason and container start/finish times are kept: the first is how
        a kubelet admission rejection shows up, the second is the real
        completion time (MAPPING.md G-10);
      * a pod bound to a node that is not one of the agents is reported, not
        silently treated as unbound.
    """
    if api is None:
        try:
            from kubernetes import client, config  # type: ignore
        except ImportError as exc:  # pragma: no cover - not installed here
            raise RuntimeError(
                "the `kubernetes` python package is not installed. Live snapshots "
                "need it: pip install kubernetes. All dry-run paths work without it."
            ) from exc
        config.load_kube_config(context=context)
        api = client.CoreV1Api()

    import time as _time

    t_before = _time.monotonic()

    node_objs = list(api.list_node(label_selector=agent_label).items)
    node_objs.sort(key=lambda n: n.metadata.name)
    if not node_objs:
        raise RuntimeError(
            f"no node carries the label {agent_label!r}. k3d's options.runtime.labels "
            "are DOCKER labels, not Kubernetes node labels; the cluster config must "
            "use options.k3s.nodeLabels (see p4/cluster/k3d-cluster.yaml).")
    nodes = [
        NodeState(
            name=n.metadata.name,
            index=i,
            cpu_allocatable_millis=_parse_cpu_millis(n.status.allocatable["cpu"]),
            mem_allocatable_bytes=_parse_mem_bytes(n.status.allocatable["memory"]),
            ready=_node_ready(n),
        )
        for i, n in enumerate(node_objs)
    ]
    by_name = {n.name: n.index for n in nodes}

    pods: list[PodState] = []
    for p in api.list_pod_for_all_namespaces().items:
        ann = p.metadata.annotations or {}
        labels = p.metadata.labels or {}
        phase = (p.status.phase if p.status is not None else None) or "Pending"
        deleting = getattr(p.metadata, "deletion_timestamp", None) is not None
        ours = (p.metadata.namespace == namespace
                and labels.get("app") == "pltr-ma-task"
                and labels.get("pltr-ma.io/run") == run_label)
        if not ours:
            # Foreign pod: charge its requests against the node it occupies.
            # Terminal pods hold nothing (kubelet admission ignores them).
            idx = by_name.get(p.spec.node_name)
            if idx is not None and phase not in ("Succeeded", "Failed") and not deleting:
                cpu_m, mem_b = _sum_requests(p)
                nodes[idx].foreign_cpu_request_millis += cpu_m
                nodes[idx].foreign_mem_request_bytes += mem_b
            continue
        cpu_m, mem_b = _sum_requests(p)
        node_name = p.spec.node_name
        node_index = by_name.get(node_name)
        reason = getattr(p.status, "reason", None) if p.status is not None else None
        if node_name and node_index is None:
            phase, reason = "Failed", f"bound-to-non-agent-node:{node_name}"
        started, finished = _container_times(p)
        pods.append(PodState(
            name=p.metadata.name,
            cpu_request_millis=cpu_m,
            mem_request_bytes=mem_b,
            work_units=float(ann.get("pltr-ma.io/work-units", "nan")),
            arrival_step=int(ann.get("pltr-ma.io/arrival-step", "-1")),
            deadline_step=int(ann.get("pltr-ma.io/deadline-step", "-1")),
            initial_agent=int(ann.get("pltr-ma.io/initial-agent", "-1")),
            node_index=node_index,
            phase=phase,
            reason=reason,
            started_at=started,
            finished_at=finished,
            deleting=deleting,
            gated=bool(getattr(p.spec, "scheduling_gates", None)),
            uid=getattr(p.metadata, "uid", None),
            waiting_reason=_waiting_reason(p),
            scheduled_at=_scheduled_at(p),
        ))

    age = _time.monotonic() - t_before
    return ClusterSnapshot(step=step, nodes=nodes, pods=pods,
                           scale=scale or UnitScale(), max_info_age_s=age)


def _ts(v) -> float | None:
    """A kubernetes datetime (or ISO string) -> unix seconds."""
    if v is None:
        return None
    if hasattr(v, "timestamp"):
        return float(v.timestamp())
    import datetime as _dt
    return _dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()


def _node_ready(node) -> bool | None:
    conds = (getattr(node.status, "conditions", None) if node.status is not None else None) or []
    for c in conds:
        if getattr(c, "type", None) == "Ready":
            return str(getattr(c, "status", "")) == "True"
    return None


def _waiting_reason(pod) -> str | None:
    sts = (pod.status.container_statuses if pod.status is not None else None) or []
    st = sts[0].state if sts and sts[0] is not None else None
    w = getattr(st, "waiting", None) if st is not None else None
    return getattr(w, "reason", None) if w is not None else None


def _scheduled_at(pod) -> float | None:
    conds = (getattr(pod.status, "conditions", None) if pod.status is not None else None) or []
    for c in conds:
        if getattr(c, "type", None) == "PodScheduled" and str(getattr(c, "status", "")) == "True":
            return _ts(getattr(c, "last_transition_time", None))
    return None


def _container_times(pod) -> tuple[float | None, float | None]:
    sts = (pod.status.container_statuses if pod.status is not None else None) or []
    if not sts or sts[0].state is None:
        return None, None
    st = sts[0].state
    if getattr(st, "terminated", None) is not None:
        return _ts(st.terminated.started_at), _ts(st.terminated.finished_at)
    if getattr(st, "running", None) is not None:
        return _ts(st.running.started_at), None
    return None, None


def _parse_cpu_millis(v: str) -> int:
    v = str(v)
    return int(v[:-1]) if v.endswith("m") else int(float(v) * 1000)


_MEM_SUFFIX = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
               "K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4}


def _parse_mem_bytes(v: str) -> int:
    v = str(v).strip()
    for suf, mult in _MEM_SUFFIX.items():
        if v.endswith(suf):
            return int(float(v[: -len(suf)]) * mult)
    return int(float(v))


def _sum_requests(pod) -> tuple[int, int]:
    cpu_m = mem_b = 0
    for c in (pod.spec.containers or []):
        req = (c.resources.requests if c.resources else None) or {}
        if "cpu" in req:
            cpu_m += _parse_cpu_millis(req["cpu"])
        if "memory" in req:
            mem_b += _parse_mem_bytes(req["memory"])
    return cpu_m, mem_b


# --------------------------------------------------------------------------- #
def print_mapping_summary() -> None:
    """`python observation.py` prints the categorisation — the P4 headline."""
    for label, items in (("DIRECTLY OBSERVABLE", directly_observable()),
                         ("APPROXIMABLE", approximable()),
                         ("NOT OBSERVABLE", not_observable())):
        print(f"\n== {label} ({len(items)}/{OBS_DIM}) ==")
        for c in items:
            print(f"  O-{c.index + 1} {c.name}")
            print(f"      sim : {c.sim_meaning}")
            print(f"      k8s : {c.k8s_source}")
            print(f"      !   : {c.caveat}")


if __name__ == "__main__":
    print_mapping_summary()
