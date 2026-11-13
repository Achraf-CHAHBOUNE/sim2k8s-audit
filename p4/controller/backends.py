"""backends.py — the three cluster backends the replay controller can drive.

    SimulatorBackend   the P3 simulator, decomposed into the controller's phases.
                       Exact; used to prove the controller adds no behaviour.
    KubeModelBackend   a MODEL of Kubernetes semantics: every bound pod runs
                       concurrently (processor sharing on a host of `host_cores`),
                       the kubelet rejects pods that do not fit AFTER binding,
                       containers take `start_delay` steps to start. Every
                       parameter is an assumption; nothing it outputs is a result.
    LiveBackend        the real cluster, through the `kubernetes` client. Imported
                       lazily; [UNVERIFIED] — no API server has been contacted.

Plus `emulate_kube_scheduler`, a stated approximation of kube-scheduler's default
scoring, used ONLY on the two dry-run backends for the no-op baseline policy.
"""
from __future__ import annotations

import copy
import glob
import os
import subprocess
import time
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from observation import (ClusterSnapshot, NodeState, PodState, UnitScale,
                         snapshot_from_kubernetes)
from action import (Op, RATE_GATE, EVIDENCE_FINALIZER, TransportFailure,  # noqa: F401
                    is_not_found, is_transport_error)
import hashlib
import json

#: Every live API call carries this deadline (protocol v1 §6.4: an API server
#: unreachable for more than 30 s is an infrastructure failure). Stricter than
#: the rule: ANY transport failure fails the attempt, it is never retried inside.
API_TIMEOUT_S = 30.0
#: Bounded end-of-episode collection of final status for deleted pods (§7.1).
COLLECT_TIMEOUT_S = 30.0
#: Container waiting reasons that mean the workload cannot run (§6.4).
INFRA_WAITING_REASONS = ("ErrImageNeverPull", "ImagePullBackOff", "ErrImagePull",
                         "InvalidImageName", "CreateContainerConfigError",
                         "CreateContainerError", "RunContainerError")


class _Deadline:
    """Wrap a real CoreV1Api so that every call carries `_request_timeout`."""

    def __init__(self, api, timeout: float):
        self._api, self._timeout = api, float(timeout)

    def __getattr__(self, name):
        f = getattr(self._api, name)
        if not callable(f):
            return f

        def call(*a, **kw):
            kw.setdefault("_request_timeout", self._timeout)
            return f(*a, **kw)
        return call


def manifest_digest(body: dict) -> str:
    """Pod spec digest for workload identity (protocol v1 §5.3): the spec after
    unit scaling WITHOUT the A1 gate (excluded by rule) and without
    schedulerName (it names the arm's scheduler, not the workload), plus the
    workload annotations. Names, run labels and timestamps are excluded."""
    spec = {k: v for k, v in body["spec"].items() if k not in ("schedulingGates", "schedulerName")}
    ann = body["metadata"].get("annotations", {})
    keep = {k: ann[k] for k in sorted(ann) if k in ("pltr-ma.io/work-units", "pltr-ma.io/arrival-step",
                                                   "pltr-ma.io/deadline-step",
                                                   "pltr-ma.io/initial-agent")}
    raw = json.dumps({"spec": spec, "annotations": keep}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()

MI = 1024 * 1024


# --------------------------------------------------------------------------- #
# kube-scheduler emulation (dry-run backends only)
# --------------------------------------------------------------------------- #
def emulate_kube_scheduler(snap: ClusterSnapshot, max_pods: int | None = None,
                           eligible: set | None = None) -> list[Op]:
    """Bind every pending pod that fits, in queue order, to its best-scoring node.

    An APPROXIMATION of the default profile, stated so it can be checked:
      * filter  NodeResourcesFit (requests + pod <= allocatable) and the pod-count
                limit (kubelet max-pods; here snap.queue_cap = 15, as configured
                in k3d-cluster.yaml);
      * score   NodeResourcesFit/LeastAllocated (mean free fraction of cpu, mem)
                + NodeResourcesBalancedAllocation (1 - |cpu_frac - mem_frac|),
                equal weights; the other default plugins (affinity, spread,
                image locality, taints) are constant on this cluster;
      * order   activeQ FIFO by creation (arrival step, then admission order);
      * ties    lowest node index. The real scheduler breaks ties at RANDOM.
    The live run uses the real kube-scheduler, never this function.
    `eligible` (arm A1): only these pending pods are considered -- the emulated
    counterpart of scheduling gates. None = every pending pod (arm A0).
    """
    max_pods = snap.queue_cap if max_pods is None else max_pods
    sc = snap.scale
    nodes = sorted(snap.nodes, key=lambda x: x.index)
    used = {}
    for nd in nodes:
        run = snap.bound_pods(nd.index)
        used[nd.index] = [sum(p.cpu_request_millis for p in run),
                          sum(p.mem_request_bytes for p in run), len(run)]
    pending = sorted((p for p in snap.pods if p.node_index is None and not p.is_terminal
                      and (eligible is None or p.name in eligible)),
                     key=lambda p: (p.arrival_step,
                                    p.enqueue_seq if p.enqueue_seq is not None else 0, p.name))
    ops: list[Op] = []
    for p in pending:
        best, best_s = None, -np.inf
        for nd in nodes:
            ca = nd.cpu_allocatable_millis - nd.foreign_cpu_request_millis
            ma = nd.mem_allocatable_bytes - nd.foreign_mem_request_bytes
            u = used[nd.index]
            if (u[0] + p.cpu_request_millis > ca or u[1] + p.mem_request_bytes > ma
                    or u[2] + 1 > max_pods):
                continue
            fc = (u[0] + p.cpu_request_millis) / ca
            fm = (u[1] + p.mem_request_bytes) / ma
            s = ((1 - fc) + (1 - fm)) / 2 * 100 + (1 - abs(fc - fm)) * 100
            if s > best_s + 1e-12:
                best, best_s = nd, s
        if best is None:
            continue                      # stays Pending (Unschedulable)
        u = used[best.index]
        u[0] += p.cpu_request_millis
        u[1] += p.mem_request_bytes
        u[2] += 1
        ops.append(Op(kind="bind", agent=-1, pod=p.name, target_node=best.name,
                      calls_api=True,
                      reason="EMULATED kube-scheduler (LeastAllocated+Balanced); "
                             "a model, not the real scheduler"))
    return ops


# --------------------------------------------------------------------------- #
# Simulator backend — cloud_env.step decomposed into the controller's phases
# --------------------------------------------------------------------------- #
class SimulatorBackend:
    """The P3 simulator standing in for the cluster.

    Each controller phase calls the SAME env code that env.step would:
      apply(bind/retarget)  = cloud_env.py:200-209 (on named tasks; ACCEPT and
                              MIGRATE must act on the agent's head — asserted)
      apply(delete-expired) = removal of exactly the tasks _expire_overdue drops
      advance               = env._process()                 (cloud_env.py:302)
      admit                 = env._spawn_arrivals()          (cloud_env.py:249)
    Units: 1 "milli" = 1 simulator unit, so the bridge arithmetic is exact.
    `expose_remaining=False` hides `remaining` (as Kubernetes does) to test the
    controller's remaining-work belief.
    """

    kind = "sim"
    emulates_scheduler = True
    calibrated = None
    #: cloud_env drops a finished task from `running`; disappearance IS completion
    #: here, and only here (protocol v1 7.1: elsewhere completion needs evidence).
    completion_by_vanish = True

    def __init__(self, template, run_id: str = "sim", expose_remaining: bool = True):
        self.env = copy.deepcopy(template)
        self.scale = UnitScale(cpu_unit_millis=1, mem_unit_bytes=1)
        self.expose_remaining = expose_remaining
        self.names: dict[int, str] = {}
        self.by_name: dict[str, object] = {}
        # Every task ever named is kept alive for the episode: `names` is keyed by
        # id(), and CPython reuses the id of a freed (completed/expired) task.
        self._alive: list = []

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self, schedule, native: bool) -> None:
        self.env.reset(seed=schedule.seed)
        self.names.clear()
        self.by_name.clear()
        self._alive.clear()

    def admit(self, tasks, step, native, names, agents, gated: bool = False) -> None:
        env = self.env
        if any(n != t.name for n, t in zip(names, tasks)):
            raise RuntimeError("the simulator never refuses an ACCEPT, so nothing "
                               "can be re-queued on this backend")
        if step > 0:
            env.t = step
            env._spawn_arrivals()
        fresh = [(i, task) for i, srv in enumerate(env.servers)
                 for task in srv.arrivals if id(task) not in self.names]
        want = sorted(zip(agents, names, tasks), key=lambda x: (x[0], x[2].index))
        if len(fresh) != len(want):
            raise RuntimeError(f"step {step}: env spawned {len(fresh)} tasks, "
                               f"schedule says {len(want)}")
        for (i, task), (agent, name, st) in zip(fresh, want):
            if (i != agent or task.cpu != st.cpu or task.mem != st.mem
                    or task.work != st.work or task.deadline_t != st.deadline_step):
                raise RuntimeError(f"step {step}: schedule and simulator disagree on {name}")
            self.names[id(task)] = name
            self.by_name[name] = task
            self._alive.append(task)

    # -- state -------------------------------------------------------------- #
    def snapshot(self, step: int) -> ClusterSnapshot:
        env = self.env
        nodes = [NodeState(name=f"sim-agent-{i}", index=i,
                           cpu_allocatable_millis=s.cpu_cap,
                           mem_allocatable_bytes=s.mem_cap)
                 for i, s in enumerate(env.servers)]
        pods = []
        for i, s in enumerate(env.servers):
            for t in s.running:
                pods.append(self._pod(t, i, i, "Running"))
            for t in s.arrivals:
                pods.append(self._pod(t, i, None, "Pending"))
        return ClusterSnapshot(step=step, nodes=nodes, pods=pods,
                               queue_cap=env.queue_cap, scale=self.scale)

    def _pod(self, t, agent, node, phase) -> PodState:
        return PodState(name=self.names[id(t)], cpu_request_millis=t.cpu,
                        mem_request_bytes=t.mem, work_units=t.work,
                        arrival_step=t.arrival_t, deadline_step=t.deadline_t,
                        initial_agent=agent, node_index=node, phase=phase,
                        remaining_work_units=(t.remaining if self.expose_remaining
                                              or node is None else None))

    def _where(self, name):
        task = self.by_name[name]
        for i, s in enumerate(self.env.servers):
            for lst in (s.arrivals, s.running):
                for k, x in enumerate(lst):
                    if x is task:
                        return i, lst, k, task
        raise KeyError(name)

    def apply(self, ops) -> list[Op]:
        servers = self.env.servers
        out = []
        for op in ops:
            if op.kind == "noop-defer":
                op.applied = True
            elif op.kind == "bind":
                i, lst, k, task = self._where(op.pod)
                if op.agent is not None and op.agent >= 0:
                    assert lst is servers[op.agent].arrivals and k == 0, \
                        f"ACCEPT of {op.pod} is not agent {op.agent}'s head"
                lst.pop(k)
                node = int(op.target_node.rsplit("-", 1)[-1])
                servers[node].running.append(task)
                op.applied = True
            elif op.kind == "retarget":
                i, lst, k, task = self._where(op.pod)
                assert lst is servers[op.agent].arrivals and k == 0, \
                    f"MIGRATE of {op.pod} is not agent {op.agent}'s head"
                lst.pop(k)
                servers[op.target_agent].arrivals.append(task)
                op.applied = True
            elif op.kind == "delete-expired":
                i, lst, k, task = self._where(op.pod)
                lst.pop(k)
                del self.by_name[op.pod]
                op.applied = True
            elif op.kind == "release-gate":
                op.applied = True          # the controller's eligible set is the gate
            else:
                raise ValueError(op.kind)
            out.append(op)
        return out

    def advance(self, step: int) -> dict:
        self.env.t = step
        before = {id(t) for s in self.env.servers for t in s.running}
        self.env._process()
        after = {id(t) for s in self.env.servers for t in s.running}
        for tid in before - after:
            self.by_name.pop(self.names[tid], None)
        return {}

    def gc(self, names) -> None:
        return None

    def cpu_seconds(self):
        return None, "none (simulator backend)"

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# Kubernetes-semantics MODEL backend
# --------------------------------------------------------------------------- #
class KubeModelBackend:
    """A model of what the live cluster will do differently from the simulator.

    ASSUMPTIONS (each is a parameter, none is measured):
      * fleet allocatable = cloud_env caps x (1000m, 512Mi) — k3d-cluster.yaml's aim;
      * every bound, started pod runs CONCURRENTLY; each is single-threaded and
        progresses `work_rate` work units per step when it has a core, and
        min(1, host_cores / n_running) of that when the host is oversubscribed
        (MAPPING.md G-6: no head-of-line service; G-3: no per-node speed);
      * a Binding is accepted by the API; the kubelet then rejects a pod that
        does not fit (requests or max-pods) -> phase Failed, reason OutOf*;
      * a bound pod starts `start_delay` steps after binding (image cached).
    """

    kind = "kube-model"
    emulates_scheduler = True
    calibrated = None
    completion_by_vanish = False

    def __init__(self, template, run_id: str = "model", host_cores: float = 12.0,
                 max_pods: int = 15, start_delay: int = 1, work_rate: float = 1.0,
                 step_seconds: float = 1.0):
        self.scale = UnitScale()
        self.caps = [(float(c), float(m)) for c, m, _ in template._fleet_spec]
        self.queue_cap = template.queue_cap
        self.host_cores = float(host_cores)
        self.max_pods = int(max_pods)
        self.start_delay = int(start_delay)
        self.work_rate = float(work_rate)
        self.step_seconds = float(step_seconds)
        self.pods: dict[str, PodState] = {}
        self.meta: dict[str, dict] = {}
        self.core_seconds = 0.0
        self._step = 0

    def _nodes(self):
        return [NodeState(name=f"k3d-pltr-p4-agent-{i}", index=i,
                          cpu_allocatable_millis=int(round(c * self.scale.cpu_unit_millis)),
                          mem_allocatable_bytes=int(round(m * self.scale.mem_unit_bytes)))
                for i, (c, m) in enumerate(self.caps)]

    def reset(self, schedule, native: bool) -> None:
        self.pods.clear()
        self.meta.clear()
        self.core_seconds = 0.0

    def admit(self, tasks, step, native, names, agents, gated: bool = False) -> None:
        self._step = step
        for task, name, agent in zip(tasks, names, agents):
            self.pods[name] = PodState(
                name=name,
                cpu_request_millis=int(round(task.cpu * 1000)),
                mem_request_bytes=int(round(task.mem * 512)) * MI,   # trace_to_pods rounding
                work_units=task.work, arrival_step=task.step,
                deadline_step=task.deadline_step, initial_agent=agent, gated=gated)
            self.meta[name] = {"progress": 0.0, "start_step": None}

    def snapshot(self, step: int) -> ClusterSnapshot:
        self._step = step
        return ClusterSnapshot(step=step, nodes=self._nodes(),
                               pods=[copy.copy(p) for p in self.pods.values()],
                               queue_cap=self.queue_cap, scale=self.scale)

    def _fits(self, node: NodeState, pod: PodState) -> str | None:
        live = [p for p in self.pods.values()
                if p.node_index == node.index and not p.is_terminal]
        if len(live) + 1 > self.max_pods:
            return "OutOfpods"
        if sum(p.cpu_request_millis for p in live) + pod.cpu_request_millis > node.cpu_allocatable_millis:
            return "OutOfcpu"
        if sum(p.mem_request_bytes for p in live) + pod.mem_request_bytes > node.mem_allocatable_bytes:
            return "OutOfmemory"
        return None

    def apply(self, ops) -> list[Op]:
        nodes = {n.name: n for n in self._nodes()}
        out = []
        for op in ops:
            pod = self.pods.get(op.pod) if op.pod else None
            if op.kind == "noop-defer":
                op.applied = True
            elif op.kind == "bind":
                if pod is None or pod.node_index is not None:
                    op.refused, op.refusal_reason = True, "409 Conflict: already bound or gone"
                else:
                    nd = nodes[op.target_node]
                    why = self._fits(nd, pod)      # evaluated BEFORE the pod is placed
                    pod.node_index = nd.index
                    op.applied = True              # the API accepts the Binding ...
                    if why:                        # ... and the kubelet rejects the pod
                        pod.phase, pod.reason = "Failed", why
                    else:
                        self.meta[pod.name]["start_step"] = self._step + self.start_delay
            elif op.kind == "retarget":
                if pod is not None:
                    pod.initial_agent = int(op.target_agent)
                    op.applied = True
                else:
                    op.refused, op.refusal_reason = True, "404 Not Found"
            elif op.kind == "delete-expired":
                if pod is not None:
                    op.evidence = {"phase": pod.phase, "started_at": pod.started_at,
                                   "finished_at": pod.finished_at}
                    del self.pods[op.pod]
                    op.applied = True
                else:
                    op.refused, op.refusal_reason = True, "404 Not Found"
            elif op.kind == "release-gate":
                if pod is not None:
                    pod.gated = False
                    op.applied = True
                else:
                    op.refused, op.refusal_reason = True, "404 Not Found"
            else:
                raise ValueError(op.kind)
            out.append(op)
        return out

    def advance(self, step: int) -> dict:
        running = []
        for p in self.pods.values():
            m = self.meta[p.name]
            if (p.node_index is not None and not p.is_terminal
                    and m["start_step"] is not None and m["start_step"] <= step):
                p.phase = "Running"
                if p.started_at is None:
                    p.started_at = float(step) * self.step_seconds
                running.append(p)
        n = len(running)
        share = min(1.0, self.host_cores / n) if n else 0.0
        self.core_seconds += min(self.host_cores, n) * self.step_seconds
        for p in running:
            m = self.meta[p.name]
            m["progress"] += self.work_rate * share
            if m["progress"] >= p.work_units - 1e-9:
                p.phase = "Succeeded"
                p.finished_at = float(step + 1) * self.step_seconds
        return {"model_running": n, "model_share": share}

    def gc(self, names) -> None:
        for n in names:
            self.pods.pop(n, None)

    def cpu_seconds(self):
        return self.core_seconds, "MODEL core-seconds (KubeModelBackend assumption, not a measurement)"

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# CPU-seconds on the live host
# --------------------------------------------------------------------------- #
class NodeCgroupCpuProbe:
    """CPU-seconds consumed by the k3d node containers, from cgroup v2.

    On a Linux Docker host each k3d node is a container with its own cgroup, so
    usage_usec in cpu.stat is a PER-NODE CPU-time counter that includes every
    pod on that node plus that node's kubelet/containerd. It is CPU time, not
    energy. Paths are [TO VERIFY ON FIRST RUN] on the P4 host (Ubuntu 24.04,
    cgroup v2, systemd driver expected).

    Scope rule (protocol v1 §7.4): with `require_all=n` this
    probe is AGENT-ONLY and fail-closed — if fewer than n agent cgroups are found
    it reports itself "unavailable (partial coverage)" and returns None. It never
    falls back to host-wide time, which has a different accounting boundary;
    host-wide CPU time and package energy are separate outcomes
    (energy/rapl_log.py HostEnergyProbe). `require_all=None` keeps the old
    behaviour (host-wide psutil fallback) for dry runs and old tests only.
    """

    PATTERNS = ("/sys/fs/cgroup/system.slice/docker-{id}.scope/cpu.stat",
                "/sys/fs/cgroup/docker/{id}/cpu.stat")

    def __init__(self, cluster: str = "pltr-p4", runner=None, reader=None,
                 require_all: int | None = None):
        self.cluster = cluster
        self.runner = runner or self._run
        self.reader = reader or (lambda p: Path(p).read_text())
        self.paths: dict[str, str] = {}
        self.source = "unavailable"
        self.require_all = require_all
        try:
            out = self.runner(["docker", "ps", "--filter", f"name=k3d-{cluster}-agent-",
                               "--format", "{{.Names}} {{.ID}}", "--no-trunc"])
            for line in out.splitlines():
                name, cid = line.split()
                for pat in self.PATTERNS:
                    p = pat.format(id=cid)
                    try:
                        self.reader(p)
                        self.paths[name] = p
                        break
                    except OSError:
                        continue
            if self.paths:
                self.source = "cgroup-v2 usage_usec of k3d agent containers (CPU time, NOT energy)"
        except Exception:   # noqa: BLE001
            self.paths = {}
        if self.require_all is not None:
            if len(self.paths) != self.require_all:
                self.source = (f"unavailable (partial coverage: {len(self.paths)} of "
                               f"{self.require_all} agent cgroups)")
                self.paths = {}
            return
        if not self.paths:
            try:
                import psutil  # noqa: F401
                self.source = "psutil host-wide busy time (CPU time, NOT energy; includes everything on the host)"
            except ImportError:
                self.source = "unavailable"

    @staticmethod
    def _run(cmd):
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout

    def per_node(self) -> dict[str, float]:
        out = {}
        for name, p in sorted(self.paths.items()):
            for line in self.reader(p).splitlines():
                k, _, v = line.partition(" ")
                if k == "usage_usec":
                    out[name] = int(v) / 1e6
        return out

    def total(self) -> float | None:
        if self.paths:
            return float(sum(self.per_node().values()))
        if self.source.startswith("psutil"):
            import psutil
            t = psutil.cpu_times()
            return float(t.user + t.system)
        return None


# --------------------------------------------------------------------------- #
# Live backend — [UNVERIFIED]: never run against an API server
# --------------------------------------------------------------------------- #
class LiveBackend:
    kind = "live"
    emulates_scheduler = False
    completion_by_vanish = False
    #: deleted pods stay readable (finalizer) until their final status is collected
    retains_evidence = True

    def __init__(self, template, run_id: str, context: str = "k3d-pltr-p4",
                 namespace: str = "pltr-p4", step_seconds: float = 1.0,
                 iters_per_work_unit: int | None = None, api=None, client_module=None,
                 cpu_probe=None, clock=time.monotonic, sleep=time.sleep,
                 cluster: str = "pltr-p4", image: str | None = None,
                 image_pull_policy: str | None = None, wall=time.time):
        from action import LiveExecutor
        import trace_to_pods as ttp

        if api is None:
            try:
                from kubernetes import client, config  # type: ignore
            except ImportError as exc:
                raise RuntimeError("--backend live needs the `kubernetes` package "
                                   "(pip install kubernetes); dry runs do not") from exc
            contexts, active = config.list_kube_config_contexts()
            names = {c["name"] for c in contexts}
            if context not in names:
                raise RuntimeError(f"kube context {context!r} not found ({sorted(names)}); "
                                   "refusing to guess which cluster to drive")
            config.load_kube_config(context=context)
            api, client_module = _Deadline(client.CoreV1Api(), API_TIMEOUT_S), client
        self.api = api
        self.client = client_module
        self.namespace = namespace
        self.run_id = run_id
        self.step_seconds = float(step_seconds)
        self.executor = LiveExecutor(namespace=namespace, api=api, client_module=client_module)
        self.scale = UnitScale()
        self.ttp = ttp
        self.cfg = ttp.ScaleConfig(
            step_seconds=self.step_seconds, namespace=namespace, run_id=run_id,
            n_agents=template.N, name_prefix=run_id,
            iters_per_work_unit=(iters_per_work_unit or ttp.DEFAULT_ITERS_PER_WORK_UNIT),
            calibrated=iters_per_work_unit is not None,
            **({"image": image} if image else {}),
            image_pull_policy=image_pull_policy)
        self.scale.assert_consistent_with(self.cfg.cpu_unit_millis, self.cfg.mem_unit_mi)
        self.calibrated = self.cfg.calibrated
        self.cpu_probe = cpu_probe if cpu_probe is not None else NodeCgroupCpuProbe(cluster)
        self.clock, self.sleep, self.wall = clock, sleep, wall
        self.t0 = None
        self.t0_unix = None
        self.create_errors = 0
        #: per pod: (planned admission wall time, actual create time), unix s
        self.admit_times: dict[str, tuple] = {}
        #: per pod: manifest digest (protocol v1 §5.3)
        self.manifests: dict[str, str] = {}

    def _label_selector(self) -> str:
        return f"app=pltr-ma-task,pltr-ma.io/run={self.run_id}"

    def reset(self, schedule, native: bool) -> None:
        try:
            self.api.read_namespace(self.namespace)
        except Exception as exc:   # noqa: BLE001
            if not is_not_found(exc):           # only a genuine 404 means "absent"
                raise
            self.api.create_namespace({"apiVersion": "v1", "kind": "Namespace",
                                       "metadata": {"name": self.namespace}})
        left = self.api.list_namespaced_pod(self.namespace,
                                            label_selector=self._label_selector()).items
        for p in left:          # leftovers of an aborted attempt with the SAME run id
            self._discard(p.metadata.name)
        self.admit_times.clear()
        self.manifests.clear()
        self.t0 = self.clock()
        self.t0_unix = self.wall()      # deadline_wall = t0_unix + d * step_seconds

    def admit(self, tasks, step, native, names, agents, gated: bool = False) -> None:
        cfg = copy.copy(self.cfg)
        cfg.scheduler_name = "default-scheduler" if native else "pltr-ma-scheduler"
        for task, name, agent in zip(tasks, names, agents):
            row = self.ttp.TraceRow(arrival_step=task.step, cpu=task.cpu, mem=task.mem,
                                    work=task.work, slack=task.slack)
            body = self.ttp.row_to_pod(row, task.index, cfg, agent)
            body["metadata"]["name"] = name
            body["metadata"]["finalizers"] = [EVIDENCE_FINALIZER]
            self.manifests[name] = manifest_digest(body)
            now = self.wall()
            body["metadata"]["annotations"]["pltr-ma.io/admitted-unix-ms"] = str(int(now * 1000))
            if gated:
                body["spec"]["schedulingGates"] = [{"name": RATE_GATE}]
            planned = (self.t0_unix + task.step * self.step_seconds) if self.t0_unix else None
            self.admit_times[name] = (planned, now)
            try:
                self.api.create_namespaced_pod(self.namespace, body)
            except Exception as exc:   # noqa: BLE001
                self.create_errors += 1
                if is_transport_error(exc):
                    raise TransportFailure(f"create {name}: {exc}") from exc
                raise

    def snapshot(self, step: int) -> ClusterSnapshot:
        return snapshot_from_kubernetes(step, namespace=self.namespace, run_label=self.run_id,
                                        scale=self.scale, api=self.api)

    def apply(self, ops) -> list[Op]:
        return self.executor.apply(ops)

    def advance(self, step: int) -> dict:
        """Wait for the end of step `step` (wall t0 + step*dt). Timeline per tick
        (protocol v1 10): the policy acts at the START of step k, right after
        T_{k-1}; service runs until T_k; then measure, admit step k's arrivals and
        observe for the next decision. `tick_overrun_s` > 0 means the control work
        of this tick did not fit in the step."""
        target = self.t0 + step * self.step_seconds
        now = self.clock()
        if now < target:
            self.sleep(target - now)
        woke = self.clock()                 # measured AFTER the sleep
        overrun = max(0.0, now - target)
        out = {"tick_overrun_s": overrun, "planned_end_s": step * self.step_seconds,
               "woke_s": woke - self.t0, "sleep_overshoot_s": max(0.0, woke - max(now, target))}
        if getattr(self.cpu_probe, "paths", None):
            out["node_cpu_seconds"] = self.cpu_probe.per_node()
        return out

    def _delete(self, name: str) -> None:
        try:
            self.api.delete_namespaced_pod(name=name, namespace=self.namespace,
                                           grace_period_seconds=0)
        except Exception as exc:   # noqa: BLE001
            if is_not_found(exc):
                return
            if is_transport_error(exc):
                raise TransportFailure(f"delete {name}: {exc}") from exc
            raise

    def _discard(self, name: str) -> None:
        """Remove a pod whose evidence is no longer needed: finalizer, then delete."""
        if self.executor.release_finalizer(name):
            self._delete(name)

    def gc(self, names) -> None:
        for n in names:
            self._discard(n)

    def read_final(self, name: str) -> dict | None:
        """Final kubelet status of a DELETED (deleting) pod, or None while the
        kubelet has not yet written a terminal state. The finalizer keeps the
        object readable; it is removed here once the status is final. A pod that
        is already gone returns {"lost": ...} (evidence not recoverable)."""
        from observation import _container_times
        try:
            pod = self.api.read_namespaced_pod(name=name, namespace=self.namespace)
        except Exception as exc:   # noqa: BLE001
            if is_not_found(exc):
                return {"phase": None, "final": True, "lost": "object gone before final status"}
            if is_transport_error(exc):
                raise TransportFailure(f"read {name}: {exc}") from exc
            raise
        phase = getattr(pod.status, "phase", None) if pod.status is not None else None
        st, fin = _container_times(pod)
        if phase not in ("Succeeded", "Failed") and fin is None:
            return None
        ev = {"phase": phase, "started_at": st, "finished_at": fin, "final": True,
              "uid": getattr(pod.metadata, "uid", None), "source": "retained-object",
              "read_unix": self.wall()}
        self._discard(name)
        return ev

    def collect(self, names, deadline_s: float = 0.0, poll_s: float = 0.5) -> dict:
        """Final evidence for deleted pods. deadline_s=0: one pass. >0: poll
        until every pod is final or the deadline passes (bounded collection)."""
        pending, out = list(names), {}
        t_end = self.clock() + deadline_s
        while True:
            for n in list(pending):
                ev = self.read_final(n)
                if ev is not None:
                    out[n] = ev
                    pending.remove(n)
            if not pending or self.clock() >= t_end:
                return out
            self.sleep(poll_s)

    def cpu_seconds(self):
        return self.cpu_probe.total(), self.cpu_probe.source

    def close(self) -> None:
        try:
            for p in self.api.list_namespaced_pod(
                    self.namespace, label_selector=self._label_selector()).items:
                self._discard(p.metadata.name)
        except Exception:   # noqa: BLE001 - the namespace reset re-checks (§6.4)
            pass


def make_backend(kind: str, template, run_id: str, **kw):
    if kind == "sim":
        return SimulatorBackend(template, run_id=run_id, **kw)
    if kind == "kube-model":
        return KubeModelBackend(template, run_id=run_id, **kw)
    if kind == "live":
        return LiveBackend(template, run_id=run_id, **kw)
    raise ValueError(kind)
