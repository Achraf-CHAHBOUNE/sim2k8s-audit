#!/usr/bin/env python3
"""action.py — cloud_env actions (accept / defer / migrate) -> Kubernetes operations.

Mirrors the action application block of ``CloudAllocationEnv.step``
(simulator/cloud_env.py:193-209) exactly, and turns each decision into a concrete
Kubernetes operation. Default mode is **dry run**: every intended operation is
logged and returned, and nothing is sent to any cluster. A live cluster is touched
only when a ``LiveExecutor`` is constructed, which requires the ``kubernetes``
package (not installed on this machine as of 2026-09-27).

Mapping, in one paragraph (the full account is in MAPPING.md rows A-1..A-4):

  ACCEPT  -> create a Binding object for the pending pod onto node i. This is
             literally what a scheduler does, and it is the one clean mapping in
             the project. It CAN BE REFUSED by the cluster; the simulator's accept
             never fails.
  DEFER   -> do nothing; the pod stays Pending. Faithful, provided the pod carries
             a non-default schedulerName so kube-scheduler does not bind it.
  MIGRATE -> RE-TARGET an unbound pod to `_least_loaded_other(i)`. NOT an
             eviction. cloud_env.MIGRATE acts on arrivals[0], a task that was
             never accepted and is not running, so no workload moves and the
             operation is free. Eviction of a RUNNING pod is available here as
             EvictOp but is disabled unless explicitly allowed, because the
             simulator never performs it and the trained policy has never chosen
             it. See MAPPING.md A-3.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Iterable, Literal, Protocol, Sequence

import numpy as np

from observation import ClusterSnapshot, PodState  # noqa: F401  (PodState re-exported)

log = logging.getLogger("p4.bridge.action")

# --------------------------------------------------------------------------- #
# Action constants — must match cloud_env.py:98
# --------------------------------------------------------------------------- #
ACCEPT, DEFER, MIGRATE = 0, 1, 2
N_ACTIONS = 3
ACTION_NAMES = {ACCEPT: "accept", DEFER: "defer", MIGRATE: "migrate"}

OpKind = Literal["bind", "noop-defer", "retarget", "evict", "delete-expired", "refused",
                 "release-gate"]

#: Scheduling gate the rate-limited kube-scheduler arm (A1, protocol v1 §3.2) puts
#: on every pod at creation; the controller removes it to make a pod eligible.
RATE_GATE = "pltr-ma.io/rate"
#: Every live replay pod carries this finalizer. A deleted pod then stays readable, with deletionTimestamp set,
#: until the controller has collected its FINAL kubelet status and removes the
#: finalizer. Without it a force deletion erases the evidence that decides MET
#: vs MISSED in the completion/deletion race.
EVIDENCE_FINALIZER = "pltr-ma.io/evidence"


class TransportFailure(RuntimeError):
    """An API call failed without an API answer (timeout, connection reset, TLS):
    an infrastructure failure of the attempt, never a refusal (§6.4)."""


def is_transport_error(exc: BaseException) -> bool:
    if isinstance(exc, TransportFailure):
        return True
    if getattr(exc, "status", None) not in (None, 0):
        return False                      # the API server answered (ApiException)
    mod = type(exc).__module__ or ""
    return (isinstance(exc, (TimeoutError, ConnectionError))
            or mod.startswith(("urllib3", "socket", "ssl")))


def is_not_found(exc: BaseException) -> bool:
    """Only a genuine 404 answer means "absent"; transport or auth failures never do."""
    return getattr(exc, "status", None) == 404


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #
@dataclass
class Op:
    """One intended Kubernetes operation, or one deliberate non-operation."""

    kind: OpKind
    agent: int
    pod: str | None
    target_node: str | None = None
    reason: str = ""
    #: Agent index of the destination of a retarget (cloud_env server index).
    #: Carried explicitly so the live executor never has to parse it out of a
    #: node name, and so the replay controller can update its routing ledger.
    target_agent: int | None = None
    #: True when this Op corresponds to a cluster call. A defer is a real decision
    #: with no call, and must still be logged so the action trace is complete.
    calls_api: bool = False
    #: Populated by an executor.
    applied: bool = False
    refused: bool = False
    refusal_reason: str = ""
    #: Terminal-state evidence read just before an expiry deletion (live only):
    #: phase, container start/finish (unix s). Lets the outcome rule tell a pod
    #: that finished just before its deletion from one that was killed
    #: (protocol v1 §7.1, completion/deletion race).
    evidence: dict | None = None

    def as_line(self) -> str:
        t = f" -> {self.target_node}" if self.target_node else ""
        s = f"[{self.kind}] agent={self.agent} pod={self.pod}{t}"
        if self.reason:
            s += f"  ({self.reason})"
        if self.refused:
            s += f"  REFUSED: {self.refusal_reason}"
        return s

    def as_dict(self) -> dict:
        return {
            "kind": self.kind, "agent": self.agent, "pod": self.pod,
            "target_node": self.target_node, "target_agent": self.target_agent,
            "reason": self.reason,
            "calls_api": self.calls_api, "applied": self.applied,
            "refused": self.refused, "refusal_reason": self.refusal_reason,
            "evidence": self.evidence,
        }


@dataclass
class StepPlan:
    """Everything one policy step implies, before anything is executed."""

    step: int
    actions: tuple[int, ...]
    ops: list[Op] = field(default_factory=list)
    #: Migrations performed this step — cloud_env charges migrate_energy per one
    #: (cloud_env.py:327). Recorded so the energy objective can be reproduced.
    migrate_count: int = 0
    #: Agents whose action was a no-op because their arrival buffer was empty
    #: (cloud_env.py:195-196). The simulator silently ignores the action; we log it,
    #: because a policy that spends most steps acting on nothing is a finding.
    idle_agents: tuple[int, ...] = ()

    def summary(self) -> str:
        lines = [f"step {self.step}: actions="
                 f"{[ACTION_NAMES[a] for a in self.actions]}"]
        lines += ["  " + o.as_line() for o in self.ops]
        if self.idle_agents:
            lines.append(f"  idle agents (no arrival to decide): {list(self.idle_agents)}")
        lines.append(f"  migrate_count={self.migrate_count} "
                     f"(sim energy charge = {self.migrate_count} * 0.05)")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Planning — pure, snapshot in / ops out. No I/O.
# --------------------------------------------------------------------------- #
def least_loaded_other(snap: ClusterSnapshot, i: int) -> int:
    """cloud_env._least_loaded_other (cloud_env.py:277-279), reproduced exactly.

    'Load' is the *count* of bound non-terminal pods, ignoring CPU, memory and
    task size, with ties broken by the lowest index (np.argmin). A real scheduler
    scores nodes on far more than this; reproducing the weak rule is deliberate,
    so that a MIGRATE decision means on the cluster what it meant in training.
    See MAPPING.md A-3.
    """
    loads = [len(snap.bound_pods(nd.index)) if nd.index != i else np.inf
             for nd in sorted(snap.nodes, key=lambda x: x.index)]
    return int(np.argmin(loads))


def _argmin_other(bound: dict[int, int], i: int) -> int:
    """least_loaded_other on a running count map (same tie-break: lowest index)."""
    loads = [bound[k] if k != i else np.inf for k in sorted(bound)]
    return int(np.argmin(loads))


def plan_step(snap: ClusterSnapshot, actions: Sequence[int]) -> StepPlan:
    """Translate one joint action into intended operations.

    Follows cloud_env.py:193-209 statement for statement:
      * an agent with an empty arrival buffer is a no-op;
      * only the HEAD arrival is acted on;
      * MIGRATE moves the task to least_loaded_other's arrival buffer.

    SEQUENTIAL, like the simulator (fixed 2026-10-04). cloud_env applies the N
    actions in a Python loop over servers, mutating state as it goes, so agent i
    sees the effect of agents 0..i-1 in the SAME step:
      * a lower-index ACCEPT raises that server's backlog before a higher-index
        MIGRATE computes `_least_loaded_other` (cloud_env.py:206-207);
      * a task migrated by agent i to an EMPTY buffer j > i becomes j's head and
        is decided by agent j in the same step (cloud_env.py:194-198).
    The first version of this function planned every agent against the
    pre-action snapshot and got both wrong; the controller's end-to-end
    equivalence test (k8s/controller/test_replay.py) caught it. Kubernetes has
    no notion of this intra-step order at all — it is one more piece of
    simulator semantics the harness has to carry.
    """
    acts = tuple(int(a) for a in np.asarray(actions, dtype=int).reshape(len(snap.nodes)))
    for a in acts:
        if a not in (ACCEPT, DEFER, MIGRATE):
            raise ValueError(f"action {a} outside Discrete(3); cloud_env has no reject")

    nodes = {nd.index: nd for nd in snap.nodes}
    plan = StepPlan(step=snap.step, actions=acts)
    idle: list[int] = []
    # Local, mutable copies of the per-agent buffers and backlogs: the state the
    # simulator's loop would see when it reaches agent i.
    buffers = {i: list(snap.arrival_pods(i)) for i in nodes}
    bound = {i: len(snap.bound_pods(i)) for i in nodes}

    for i in sorted(nodes):
        arrivals = buffers[i]
        if not arrivals:
            idle.append(i)
            continue                      # cloud_env.py:195-196
        head = arrivals[0]
        a = acts[i]

        if a == ACCEPT:
            arrivals.pop(0)
            bound[i] += 1
            plan.ops.append(Op(
                kind="bind", agent=i, pod=head.name,
                target_node=nodes[i].name, calls_api=True,
                reason="cloud_env ACCEPT -> Binding; may be refused (MAPPING.md A-1)",
            ))
        elif a == DEFER:
            plan.ops.append(Op(
                kind="noop-defer", agent=i, pod=head.name, calls_api=False,
                reason="pod stays Pending; requires schedulerName != default-scheduler",
            ))
        else:  # MIGRATE
            arrivals.pop(0)
            j = _argmin_other(bound, i)
            buffers[j].append(head)
            plan.ops.append(Op(
                kind="retarget", agent=i, pod=head.name,
                target_node=nodes[j].name, target_agent=j, calls_api=True,
                reason=f"cloud_env MIGRATE -> re-route UNBOUND pod to agent {j}; "
                       "no workload moves, no eviction (MAPPING.md A-3)",
            ))
            plan.migrate_count += 1

    plan.idle_agents = tuple(idle)
    return plan


def plan_expiry(snap: ClusterSnapshot) -> list[Op]:
    """cloud_env._expire_overdue (cloud_env.py:281-300) has NO Kubernetes analogue.

    The simulator drops every task, queued or running, once ``t > deadline_t`` and
    counts it as an SLA violation. Kubernetes will not do this. The replay
    controller must issue these deletions itself, and must issue them identically
    in the kube-scheduler baseline run or the comparison is void (MAPPING.md G-7).
    """
    ops: list[Op] = []
    for nd in sorted(snap.nodes, key=lambda x: x.index):
        for p in list(snap.bound_pods(nd.index)) + list(snap.arrival_pods(nd.index)):
            if snap.step > p.deadline_step:
                ops.append(Op(
                    kind="delete-expired", agent=nd.index, pod=p.name,
                    target_node=nd.name if p.node_index is not None else None,
                    calls_api=True,
                    reason=f"step {snap.step} > deadline_step {p.deadline_step}; "
                           "expiry is OUR rule, not a cluster behaviour",
                ))
    return ops


# --------------------------------------------------------------------------- #
# Executors
# --------------------------------------------------------------------------- #
class Executor(Protocol):
    def apply(self, ops: Iterable[Op]) -> list[Op]: ...


class DryRunExecutor:
    """Logs intended operations. Touches nothing. The default.

    Also maintains the routing bookkeeping that `retarget` implies, so a dry run
    can be driven for a whole episode against a hand-built or simulated snapshot
    and still behave like the simulator.
    """

    def __init__(self, sink=None, jsonl: bool = False) -> None:
        self.sink = sink                  # a writable file-like, or None for logging
        self.jsonl = jsonl
        self.applied: list[Op] = []

    def _emit(self, op: Op) -> None:
        if self.sink is not None:
            self.sink.write((json.dumps(op.as_dict()) if self.jsonl
                             else "DRY-RUN " + op.as_line()) + "\n")
        else:
            log.info("DRY-RUN %s", op.as_line())

    def apply(self, ops: Iterable[Op]) -> list[Op]:
        out = []
        for op in ops:
            op.applied = False            # nothing was applied; be literal
            self._emit(op)
            self.applied.append(op)
            out.append(op)
        return out


class SnapshotMutatingDryRunExecutor(DryRunExecutor):
    """Dry run that also advances a local ClusterSnapshot, for offline loop tests.

    Lets the whole accept/defer/migrate loop be exercised without a cluster: bind
    sets ``node_index``, retarget rewrites ``initial_agent``, delete-expired marks
    the pod Failed. This is a MODEL of the cluster's reaction, not the cluster.
    """

    def __init__(self, snap: ClusterSnapshot, **kw) -> None:
        super().__init__(**kw)
        self.snap = snap

    def apply(self, ops: Iterable[Op]) -> list[Op]:
        by_name = {p.name: p for p in self.snap.pods}
        name_to_index = {nd.name: nd.index for nd in self.snap.nodes}
        out = []
        for op in ops:
            pod = by_name.get(op.pod) if op.pod else None
            if op.kind == "bind" and pod is not None:
                pod.node_index = name_to_index[op.target_node]
                pod.phase = "Running"
                op.applied = True
            elif op.kind == "retarget" and pod is not None:
                pod.initial_agent = name_to_index[op.target_node]
                # cloud_env.py:208 APPENDS to the destination buffer, so a
                # migrated task goes to the TAIL, not to its arrival-time
                # position. Reproduce that, or the head-of-line task after a
                # migration is the wrong one.
                seqs = [p.enqueue_seq for p in self.snap.pods
                        if p.enqueue_seq is not None]
                pod.enqueue_seq = (max(seqs) + 1) if seqs else 0
                op.applied = True
            elif op.kind == "delete-expired" and pod is not None:
                pod.phase = "Failed"
                op.applied = True
            self._emit(op)
            self.applied.append(op)
            out.append(op)
        return out


class LiveExecutor:
    """Sends operations to a real cluster. Requires the `kubernetes` package.

    Not exercised: no cluster has been created in this repo, so every code path
    below is **[UNVERIFIED]** against a real API server.
    """

    def __init__(self, namespace: str = "pltr-p4", allow_eviction: bool = False,
                 context: str | None = None, api=None, client_module=None):
        """`api` / `client_module` let a test inject a fake CoreV1Api and fake
        model classes, so the call logic is exercised offline. In normal use both
        are None and the real `kubernetes` package is loaded."""
        if api is None:
            try:
                from kubernetes import client, config  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "LiveExecutor needs the `kubernetes` package (pip install kubernetes). "
                    "Use DryRunExecutor for everything that does not require a cluster."
                ) from exc
            config.load_kube_config(context=context)
            client_module = client
            api = client.CoreV1Api()
        self._client = client_module
        self.v1 = api
        self.namespace = namespace
        self.allow_eviction = allow_eviction

    def apply(self, ops: Iterable[Op]) -> list[Op]:
        out = []
        for op in ops:
            try:
                if op.kind == "bind":
                    self._bind(op)
                    op.applied = True
                elif op.kind == "noop-defer":
                    op.applied = True          # deferral IS doing nothing
                elif op.kind == "retarget":
                    self._retarget(op)
                    op.applied = True
                elif op.kind == "delete-expired":
                    self._delete(op)
                    op.applied = True
                elif op.kind == "release-gate":
                    self._release_gate(op)
                    op.applied = True
                elif op.kind == "evict":
                    if not self.allow_eviction:
                        raise PermissionError(
                            "eviction is disabled: cloud_env.MIGRATE never moves a "
                            "RUNNING task (it acts on arrivals[0]), so evicting one "
                            "models behaviour the policy was never trained on, and "
                            "the replacement pod restarts work from zero. See "
                            "MAPPING.md A-3. Pass allow_eviction=True only for a "
                            "deliberately separate experiment."
                        )
                    self._evict(op)
                    op.applied = True
            except Exception as exc:              # noqa: BLE001 - log every refusal
                if is_transport_error(exc):
                    raise TransportFailure(f"{op.kind} {op.pod}: {type(exc).__name__}: {exc}") from exc
                op.refused = True
                op.refusal_reason = f"{type(exc).__name__}: {exc}"
                log.warning("REFUSED %s", op.as_line())
            out.append(op)
        return out

    # -- individual calls --------------------------------------------------- #
    def _bind(self, op: Op) -> None:
        """POST .../pods/{name}/binding — exactly what kube-scheduler does.

        A successful POST is NOT an admission. The API server does not check
        resource fit on a Binding (it only refuses a pod that is already bound,
        409). Fit is checked by the KUBELET after the fact: a pod that does not
        fit goes to phase=Failed with status.reason OutOfcpu / OutOfmemory /
        OutOfpods. So most refusals are invisible here and surface one snapshot
        later; k8s/controller/replay.py detects them from the pod status and
        re-queues the task. [TO VERIFY ON FIRST RUN] the exact reason strings.
        """
        body = self._client.V1Binding(
            metadata=self._client.V1ObjectMeta(name=op.pod, namespace=self.namespace),
            target=self._client.V1ObjectReference(
                api_version="v1", kind="Node", name=op.target_node),
        )
        # _preload_content=False: the API returns a Status, which the generated
        # client mis-deserialises as V1Binding.
        self.v1.create_namespaced_pod_binding(
            name=op.pod, namespace=self.namespace, body=body, _preload_content=False)

    def _retarget(self, op: Op) -> None:
        """Re-route an UNBOUND pod. No workload moves; this is bookkeeping only.

        The annotation/label is patched so the routing decision is auditable after
        the run, and so a restarted controller recovers the same routing.

        Fixed 2026-10-04: the first version patched only the LABEL, parsed from
        the node name, and left the `pltr-ma.io/initial-agent` ANNOTATION — the
        field `snapshot_from_kubernetes` reads — untouched. Every re-read of the
        cluster would then have put the pod back in its original buffer, silently
        undoing every MIGRATE. The agent index now travels on the Op.
        """
        if op.target_agent is None:
            raise ValueError(f"retarget of {op.pod} carries no target_agent")
        agent = str(int(op.target_agent))
        patch = {"metadata": {
            "labels": {"pltr-ma.io/initial-agent": agent},
            "annotations": {
                "pltr-ma.io/initial-agent": agent,
                "pltr-ma.io/retargeted-to": op.target_node,
                "pltr-ma.io/retarget-note":
                    "cloud_env MIGRATE on an unbound task; no eviction occurred",
            }}}
        self.v1.patch_namespaced_pod(name=op.pod, namespace=self.namespace, body=patch)

    def _delete(self, op: Op) -> None:
        # A pod can finish between the snapshot that planned its expiry and this
        # call. The simulator's rule is evaluated on the snapshot, so the deletion
        # is still issued; a 404 (already gone) surfaces as `refused` and the
        # controller reconciles it against the next snapshot.
        # Evidence first (protocol v1 §7.1): the pod's state as the API reports it
        # immediately before deletion. A failed read is recorded, never guessed.
        try:
            pod = self.v1.read_namespaced_pod(name=op.pod, namespace=self.namespace)
            from observation import _container_times
            st, fin = _container_times(pod)
            op.evidence = {"phase": getattr(pod.status, "phase", None),
                           "started_at": st, "finished_at": fin}
        except Exception as exc:              # noqa: BLE001
            op.evidence = {"phase": None, "read_error": f"{type(exc).__name__}: {exc}"}
        self.v1.delete_namespaced_pod(
            name=op.pod, namespace=self.namespace, grace_period_seconds=0)

    def release_finalizer(self, name: str) -> bool:
        """Remove the evidence finalizer; True if the pod still existed. A 404 is
        "already gone"; any other failure propagates."""
        try:
            self.v1.patch_namespaced_pod(name=name, namespace=self.namespace,
                                         body=[{"op": "remove", "path": "/metadata/finalizers"}])
            return True
        except Exception as exc:              # noqa: BLE001
            if is_not_found(exc):
                return False
            if is_transport_error(exc):
                raise TransportFailure(f"finalizer {name}: {exc}") from exc
            # 422: no finalizers left on the object (already removed) -> fine
            if getattr(exc, "status", None) == 422:
                return True
            raise

    def _release_gate(self, op: Op) -> None:
        """Remove the A1 scheduling gate: the pod becomes eligible for the
        unmodified kube-scheduler. Kubernetes only allows gates to be REMOVED
        after creation (pod scheduling readiness). JSON patch, so nothing else in
        the spec is touched. [TO VERIFY ON FIRST RUN] on k3s v1.30.4."""
        self.v1.patch_namespaced_pod(
            name=op.pod, namespace=self.namespace,
            body=[{"op": "remove", "path": "/spec/schedulingGates"}])

    def _evict(self, op: Op) -> None:  # pragma: no cover - disabled by default
        body = self._client.V1Eviction(
            metadata=self._client.V1ObjectMeta(name=op.pod, namespace=self.namespace))
        self.v1.create_namespaced_pod_eviction(
            name=op.pod, namespace=self.namespace, body=body)


# --------------------------------------------------------------------------- #
# Convenience: one full step
# --------------------------------------------------------------------------- #
def execute_step(snap: ClusterSnapshot, actions: Sequence[int],
                 executor: Executor | None = None,
                 include_expiry: bool = True) -> StepPlan:
    """Plan and (dry-)execute one step. Order follows cloud_env.step:

      1. apply each agent's action to its head arrival   (cloud_env.py:193-209)
      2. expire overdue tasks                            (cloud_env.py:211-212)

    Processing (step 3 in the simulator) has no counterpart: the pods run by
    themselves. That absence is MAPPING.md G-6.
    """
    executor = executor or DryRunExecutor()
    plan = plan_step(snap, actions)
    if include_expiry:
        plan.ops.extend(plan_expiry(snap))
    executor.apply(plan.ops)
    return plan


# --------------------------------------------------------------------------- #
ACTION_MAPPING_SUMMARY = """\
A-1 accept  (a) clean   Binding to node i. Divergences: the cluster can REFUSE
                        (simulator's accept never fails); binding is irreversible;
                        binding != started (kubelet startup latency unmodelled).
A-2 defer   (a) clean   Do nothing, pod stays Pending. Requires a non-default
                        schedulerName, or kube-scheduler binds it anyway and the
                        experiment is silently invalid.
A-3 migrate (b) partial Re-target an UNBOUND pod, free. NOT eviction: cloud_env
                        migrates arrivals[0], never a running task. Destination is
                        forced by a count-only least-loaded rule, not chosen by the
                        policy. The simulator charges 0.05 energy for an operation
                        that is free here, so expect under-migration on the cluster.
                        Live migration of a RUNNING pod is (c) NOT EXPRESSIBLE:
                        delete+recreate restarts the work from zero, while the
                        simulator preserves `remaining` and `arrival_t`.
A-4 (no reject)  (a)    The absence transfers cleanly: leaving a pod pending
                        forever IS deferral.
expiry      (b) ours    cloud_env._expire_overdue must be re-implemented by our
                        controller and applied identically to the baseline.
"""


if __name__ == "__main__":
    print(ACTION_MAPPING_SUMMARY)
