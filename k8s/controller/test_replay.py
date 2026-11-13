"""Tests for the replay controller. No cluster, no Docker, no network.

The load-bearing tests are the `test_sim_backend_reproduces_env_step_*` family:
the controller, driving the P3 simulator through the bridges, must produce the
SAME vector return, constraint costs, completions and SLA count as the plain
`env.reset(seed); env.step(...)` loop — bit for bit (np.array_equal, not
allclose) — for every policy type the paper compares. If they pass, the tick
loop, the bridges, the ledger and the shadow env add no behaviour of their own,
and every difference measured on the live cluster is the cluster's.

Run:  python -m pytest k8s/controller -q
"""
from __future__ import annotations

import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

import _paths  # noqa: F401
import backends as B
import replay as R
from action import ACCEPT, DEFER, MIGRATE

HORIZON = 150
SEEDS = (30000, 30001)


@pytest.fixture(scope="module")
def factory():
    return R.trace_env_factory(R.DEFAULT_TRACE, HORIZON)


class RandomPolicy:
    """Exercises MIGRATE heavily (and so the intra-step ordering)."""
    name = "random"

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, cond=None):
        return self.rng.integers(0, 3, size=np.asarray(obs).shape[0]), None


class AlwaysMigrate:
    name = "always-migrate"

    def act(self, obs, cond=None):
        return np.full(np.asarray(obs).shape[0], MIGRATE), None


def heuristic(name, param, env):
    import heuristics
    return heuristics.make_policy(name, env, param)


POLICIES = [
    ("heur_local", 1.0), ("heur_local", 5.0), ("heur_local_urgent", 1.0),
    ("heur_leastloaded", 1.0), ("heur_leastloaded", 2.0),
    ("heur_dpp", 10.0), ("random", 0), ("always-migrate", 0),
]


def build(name, param, env):
    if name == "random":
        return RandomPolicy(7)
    if name == "always-migrate":
        return AlwaysMigrate()
    return heuristic(name, param, env)


def reference(factory, name, param, seed, cond):
    """The plain simulator loop (train.eval_policy's loop, one episode)."""
    env = factory()
    pol = build(name, param, env)
    obs, _ = env.reset(seed=seed)
    rv, cs = np.zeros(3), np.zeros(3)
    for _ in range(HORIZON):
        a, _ = pol.act(obs, cond)
        obs, r, term, trunc, info = env.step(a)
        rv += r
        cs += info["cost"]
        if term or trunc:
            break
    return rv, cs, env.episode_summary()


def controlled(factory, name, param, seed, cond, expose_remaining=True,
               backend="sim", sink=None, **bkw):
    sched = R.ArrivalSchedule(factory, seed, HORIZON, prefix=f"e{seed}")
    tmpl = factory()
    tmpl.trace.offset = sched.offset
    pol = build(name, param, tmpl)
    if backend == "sim":
        be = B.SimulatorBackend(tmpl, expose_remaining=expose_remaining)
    else:
        be = B.KubeModelBackend(tmpl, **bkw)
    ctl = R.ReplayController(be, pol, sched, cond=cond, template_env=tmpl, sink=sink)
    return ctl.run(), ctl


UNI = np.full(3, 1 / 3, dtype=np.float32)


# --------------------------------------------------------------------------- #
# The schedule is the simulator's own arrival process
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
def test_schedule_matches_env_arrivals_and_rng(factory, seed):
    sched = R.ArrivalSchedule(factory, seed, HORIZON)
    env = factory()
    env.reset(seed=seed)
    assert env.rng.bit_generator.state == sched.rng_after(0).bit_generator.state
    assert sched.offset == env.trace.offset
    n0 = sum(len(s.arrivals) for s in env.servers)
    assert n0 == len(sched.at(0))
    for t in range(1, HORIZON + 1):
        alive = [x for s in env.servers for x in s.arrivals]   # keep ids unique
        before = {id(x) for x in alive}
        env.step([DEFER] * env.N)
        new = [(i, x) for i, s in enumerate(env.servers) for x in s.arrivals
               if id(x) not in before]
        got = sorted((st.agent, st.cpu, st.mem, st.work, st.deadline_step) for st in sched.at(t))
        want = sorted((i, x.cpu, x.mem, x.work, x.deadline_t) for i, x in new)
        assert got == want, t
        assert env.rng.bit_generator.state == sched.rng_after(t).bit_generator.state


# --------------------------------------------------------------------------- #
# THE load-bearing tests: controller + sim backend == env.step, bit for bit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name,param", POLICIES)
def test_sim_backend_reproduces_env_step(factory, name, param, seed):
    rv, cs, summ = reference(factory, name, param, seed, UNI)
    res, _ = controlled(factory, name, param, seed, UNI)
    s = res.summary
    assert np.array_equal(np.array(s["return_vec"]), rv), (s["return_vec"], rv)
    assert np.array_equal(np.array(s["mean_cost"]) * HORIZON, cs) or \
        np.array_equal(np.array(s["mean_cost"]), cs / HORIZON)
    assert s["completed_tasks"] == summ["completed_tasks"]
    assert s["sla_violations"] == summ["sla_violations"]
    assert s["mean_latency_steps"] == summ["mean_latency"]
    assert s["mean_backlog"] == summ["mean_backlog"]
    assert s["refused_bindings"] == 0


@pytest.mark.parametrize("name,param", [("heur_dpp", 10.0), ("heur_leastloaded", 1.0),
                                        ("random", 0)])
def test_remaining_work_belief_reproduces_hidden_state(factory, name, param):
    """With `remaining` hidden (as on Kubernetes), the ledger's sim-FIFO belief
    must recover it exactly — so DPP's shadow env is the true env state."""
    rv, _, _ = reference(factory, name, param, SEEDS[0], UNI)
    res, _ = controlled(factory, name, param, SEEDS[0], UNI, expose_remaining=False)
    assert np.array_equal(np.array(res.summary["return_vec"]), rv)


def test_the_migrate_heavy_runs_really_migrate(factory):
    res, _ = controlled(factory, "always-migrate", 0, SEEDS[0], UNI)
    assert res.summary["migrations"] > 100


def test_sequential_plan_semantics_are_needed():
    """A migration into an EMPTY higher-index buffer is decided again by that
    agent in the same step (cloud_env.py:194-209). The pre-2026-10-04 plan_step
    ignored this; this pins the fix."""
    from observation import ClusterSnapshot, NodeState, PodState, UnitScale
    from action import plan_step
    sc = UnitScale(1, 1)
    nodes = [NodeState(f"n-{i}", i, 1.0, 1.0) for i in range(3)]
    pods = [PodState("a", 0.1, 0.1, 1.0, 0, 20, 0, enqueue_seq=0),
            PodState("r1", 0.1, 0.1, 1.0, 0, 20, 2, node_index=2, phase="Running"),
            PodState("r0", 0.1, 0.1, 1.0, 0, 20, 0, node_index=0, phase="Running")]
    snap = ClusterSnapshot(step=1, nodes=nodes, pods=pods, scale=sc)
    plan = plan_step(snap, [MIGRATE, ACCEPT, ACCEPT])
    kinds = [(o.kind, o.agent, o.pod, o.target_agent) for o in plan.ops]
    # agent 0 migrates "a" to agent 1 (backlog 0); agent 1 then ACCEPTs it.
    assert kinds == [("retarget", 0, "a", 1, ), ("bind", 1, "a", None)]
    # and a lower-index ACCEPT changes the least-loaded target of a later MIGRATE
    pods2 = [PodState("x", 0.1, 0.1, 1.0, 0, 20, 0, enqueue_seq=0),
             PodState("y", 0.1, 0.1, 1.0, 0, 20, 2, enqueue_seq=1),
             PodState("r", 0.1, 0.1, 1.0, 0, 20, 1, node_index=1, phase="Running")]
    snap2 = ClusterSnapshot(step=1, nodes=nodes, pods=pods2, scale=sc)
    plan2 = plan_step(snap2, [ACCEPT, DEFER, MIGRATE])
    # loads after agent 0's ACCEPT: [1, 1, inf] -> argmin = 0
    assert plan2.ops[-1].target_agent == 0


# --------------------------------------------------------------------------- #
# Default scheduler (no-op policy) — on both dry-run backends
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", ["sim", "kube-model"])
def test_default_scheduler_policy_runs_and_never_overcommits(factory, backend):
    sched = R.ArrivalSchedule(factory, SEEDS[0], HORIZON, prefix="d")
    tmpl = factory()
    be = B.make_backend(backend, tmpl, run_id="d")
    res = R.ReplayController(be, R.DefaultSchedulerPolicy(), sched,
                             template_env=tmpl).run()
    s = res.summary
    assert s["binds"] > 0 and s["completed_tasks"] > 0
    assert s["mean_cost"][0] == 0.0          # c1 structurally 0 (MAPPING G-9)
    assert s["refused_bindings"] == 0        # the emulated scheduler filters first
    for tk in res.ticks:
        assert tk["actions"] == ["defer"] * 3 and tk["native"]
        assert all(o["agent"] == -1 for o in tk["ops"] if o["kind"] == "bind")


def test_emulated_scheduler_respects_allocatable_and_max_pods():
    from observation import ClusterSnapshot, NodeState, PodState, UnitScale
    sc = UnitScale()
    nodes = [NodeState("n-0", 0, 1000, 512 * B.MI), NodeState("n-1", 1, 2000, 1024 * B.MI)]
    pods = [PodState(f"p{k}", 400, 100 * B.MI, 1.0, k, k + 20, 0) for k in range(10)]
    snap = ClusterSnapshot(step=1, nodes=nodes, pods=pods, scale=sc, queue_cap=15)
    ops = B.emulate_kube_scheduler(snap)
    per = {"n-0": 0, "n-1": 0}
    for o in ops:
        per[o.target_node] += 1
    assert per == {"n-0": 2, "n-1": 5}                 # 2x400<=1000, 5x400<=2000
    assert B.emulate_kube_scheduler(snap, max_pods=1).__len__() == 2


# --------------------------------------------------------------------------- #
# Kubernetes-semantics model: refusals, requeues, concurrency
# --------------------------------------------------------------------------- #
def test_kube_model_kubelet_refusals_are_detected_and_requeued(factory):
    # tau=5 accepts almost everything -> the kubelet must reject some pods
    res, ctl = controlled(factory, "heur_local", 5.0, SEEDS[0], UNI, backend="kube-model")
    s = res.summary
    assert s["refused_kubelet"] > 0
    assert s["requeues"] == s["refused_kubelet"]
    assert s["mean_cost"][0] == 0.0
    recs = ctl.ledger.recs.values()
    req = [r for r in recs if r.requeued_from]
    assert req and all(r.name.startswith(r.task.name + "-r") for r in req)
    # every task is accounted for exactly once at the end
    final = [r for r in recs if r.state in ("pending", "bound", "completed", "expired", "failed")]
    assert len(final) == ctl.schedule.n_tasks
    assert s["completed_tasks"] + s["sla_violations"] + s["unfinished_at_end"] + \
        s["failed_other"] == ctl.schedule.n_tasks


@pytest.mark.parametrize("name,param", [("heur_leastloaded", 1.0), ("heur_dpp", 10.0),
                                        ("heur_local", 1.0), ("random", 0)])
def test_kube_model_runs_every_policy_type(factory, name, param):
    out = io.StringIO()
    res, _ = controlled(factory, name, param, SEEDS[0], UNI, backend="kube-model", sink=out)
    lines = [json.loads(x) for x in out.getvalue().splitlines()]
    assert lines[0]["type"] == "header" and lines[-1]["type"] == "summary"
    assert sum(1 for x in lines if x["type"] == "tick") == HORIZON
    assert res.summary["cpu_seconds"] is not None
    assert "MODEL" in res.summary["cpu_seconds_source"]
    assert lines[0]["info_class"] == R.INFO_CLASS.get(name, "local-obs")


def test_kube_model_serves_concurrently_unlike_the_simulator(factory):
    """MAPPING G-6 in one assertion: the same policy, same episode, completes
    more work under concurrent service than under one-head-per-server."""
    sim, _ = controlled(factory, "heur_local", 1.0, SEEDS[0], UNI)
    mod, _ = controlled(factory, "heur_local", 1.0, SEEDS[0], UNI, backend="kube-model",
                        start_delay=0)
    assert mod.summary["completed_tasks"] > sim.summary["completed_tasks"]


# --------------------------------------------------------------------------- #
# Learned policies (torch) through the controller
# --------------------------------------------------------------------------- #
def test_loaded_actor_policy_reproduces_env_step(factory, tmp_path):
    torch = pytest.importorskip("torch")
    import learned
    from mappo import Actor
    actors = []
    torch.manual_seed(3)
    for _ in range(3):
        actors.append(Actor(9, 3, 64, cond_dim=3))
    path = tmp_path / "w.pt"
    learned.save_actors(actors, path, meta={"variant": "test-random-init"}, cond_dim=3)
    for greedy in (True, False):
        pol = learned.LoadedActorPolicy.load(path, greedy=greedy)
        env = factory()
        obs, _ = env.reset(seed=SEEDS[0])
        torch.manual_seed(11)
        rv = np.zeros(3)
        for _ in range(HORIZON):
            a, _ = pol.act(obs, UNI)
            obs, r, *_ = env.step(a)
            rv += r
        sched = R.ArrivalSchedule(factory, SEEDS[0], HORIZON)
        tmpl = factory()
        pol2 = learned.LoadedActorPolicy.load(path, greedy=greedy)
        torch.manual_seed(11)
        res = R.ReplayController(B.SimulatorBackend(tmpl), pol2, sched, cond=UNI,
                                 template_env=tmpl).run()
        assert np.array_equal(np.array(res.summary["return_vec"]), rv)


# --------------------------------------------------------------------------- #
# Live path against a FAKE API (logic only; the real API is [UNVERIFIED])
# --------------------------------------------------------------------------- #
class _Obj(SimpleNamespace):
    pass


class FakeClient:
    """Stand-in for kubernetes.client's model classes."""
    class V1Binding(_Obj):
        pass

    class V1ObjectMeta(_Obj):
        pass

    class V1ObjectReference(_Obj):
        pass


class NotFound(RuntimeError):
    """A genuine API 404 answer (kubernetes ApiException carries .status)."""
    status = 404


class FakeAPI:
    """A minimal in-memory CoreV1Api with a toy kubelet: a bound pod runs for
    `run_ticks` pod listings, then Succeeds; a pod that does not fit is Failed
    OutOfcpu (kubelet admission). Finalizers behave as in Kubernetes: a delete
    of a pod with finalizers only sets deletionTimestamp; the kubelet then kills
    a running container (phase Failed); the object goes when the last
    finalizer is removed."""

    def __init__(self, n_nodes=3, run_ticks=3):
        self.nodes = [_Obj(metadata=_Obj(name=f"k3d-pltr-p4-agent-{i}"),
                           status=_Obj(allocatable={"cpu": f"{1000 + 500 * i}m",
                                                    "memory": f"{(512 + 256 * i) * 1024}Ki",
                                                    "pods": "15"}))
                      for i in range(n_nodes)]
        self.pods: dict[str, _Obj] = {}
        self.calls = []
        self.run_ticks = run_ticks
        self.namespaces = set()
        # a kube-system pod on agent-1 that must be subtracted (CoreDNS-like)
        self.pods["coredns"] = self._mk({"metadata": {"name": "coredns", "namespace": "kube-system",
                                                      "labels": {}, "annotations": {}},
                                         "spec": {"containers": [{"resources": {"requests": {
                                             "cpu": "100m", "memory": "70Mi"}}}]}},
                                        node="k3d-pltr-p4-agent-1", phase="Running")

    def _mk(self, body, node=None, phase="Pending"):
        md = body["metadata"]
        cs = [_Obj(resources=_Obj(requests=c.get("resources", {}).get("requests", {})))
              for c in body["spec"]["containers"]]
        return _Obj(metadata=_Obj(name=md["name"], namespace=md.get("namespace"),
                                  labels=dict(md.get("labels", {})),
                                  annotations=dict(md.get("annotations", {})),
                                  finalizers=list(md.get("finalizers", []) or []),
                                  uid=f"uid-{md['name']}",
                                  deletion_timestamp=None),
                    spec=_Obj(node_name=node, containers=cs,
                              scheduler_name=body["spec"].get("schedulerName")),
                    status=_Obj(phase=phase, reason=None, container_statuses=None),
                    _age=0)

    # -- API surface used by the code ------------------------------------- #
    def read_namespace(self, ns):
        if ns not in self.namespaces:
            raise NotFound("404")

    def create_namespace(self, body):
        self.namespaces.add(body["metadata"]["name"])

    def list_node(self, label_selector=None):
        assert label_selector == "pltr-ma.io/role=agent"
        return _Obj(items=list(self.nodes))

    def list_pod_for_all_namespaces(self):
        self._kubelet()
        return _Obj(items=list(self.pods.values()))

    def list_namespaced_pod(self, ns, label_selector=None):
        want = dict(kv.split("=") for kv in label_selector.split(","))
        return _Obj(items=[p for p in self.pods.values() if p.metadata.namespace == ns
                           and all(p.metadata.labels.get(k) == v for k, v in want.items())])

    def create_namespaced_pod(self, ns, body):
        self.calls.append(("create", body["metadata"]["name"], body["spec"]["schedulerName"]))
        self.pods[body["metadata"]["name"]] = self._mk(body)

    def create_namespaced_pod_binding(self, name, namespace, body, _preload_content=True):
        p = self.pods[name]
        if p.spec.node_name:
            raise RuntimeError("409 Conflict")
        p.spec.node_name = body.target.name
        self.calls.append(("bind", name, body.target.name))
        node = next(n for n in self.nodes if n.metadata.name == body.target.name)
        alloc = int(node.status.allocatable["cpu"][:-1])
        used = sum(int(q.spec.containers[0].resources.requests["cpu"][:-1])
                   for q in self.pods.values()
                   if q is not p and q.spec.node_name == body.target.name
                   and q.status.phase in ("Pending", "Running"))
        if used + int(p.spec.containers[0].resources.requests["cpu"][:-1]) > alloc:
            p.status.phase, p.status.reason = "Failed", "OutOfcpu"

    def read_namespaced_pod(self, name, namespace, **kw):
        if name not in self.pods:
            raise NotFound("404")
        return self.pods[name]

    def patch_namespaced_pod(self, name, namespace, body):
        if name not in self.pods:
            raise NotFound("404")
        p = self.pods[name]
        if isinstance(body, list) and body == [{"op": "remove", "path": "/metadata/finalizers"}]:
            p.metadata.finalizers = []
            self.calls.append(("unfinalize", name, None))
            if p.metadata.deletion_timestamp is not None:
                del self.pods[name]
            return
        p.metadata.labels.update(body["metadata"].get("labels", {}))
        p.metadata.annotations.update(body["metadata"].get("annotations", {}))
        self.calls.append(("patch", name, body["metadata"]["annotations"]["pltr-ma.io/initial-agent"]))

    def delete_namespaced_pod(self, name, namespace, grace_period_seconds=None):
        if name not in self.pods:
            raise NotFound("404")
        p = self.pods[name]
        self.calls.append(("delete", name, None))
        if p.metadata.finalizers:
            p.metadata.deletion_timestamp = p.metadata.deletion_timestamp or "now"
            return
        del self.pods[name]

    def _kubelet(self):
        for p in self.pods.values():
            if p.metadata.namespace != "pltr-p4":
                continue
            if p.metadata.deletion_timestamp is not None:
                if p.status.phase in ("Pending", "Running"):     # kill: terminal phase
                    p.status.phase, p.status.reason = "Failed", None
                    p.status.container_statuses = [_Obj(state=_Obj(running=None, terminated=_Obj(
                        started_at="2026-10-04T10:00:00Z", finished_at="2026-10-04T10:00:09Z")))]
                continue
            if not p.spec.node_name:
                continue
            if p.status.phase == "Pending":
                p.status.phase = "Running"
                p.status.container_statuses = [_Obj(state=_Obj(
                    running=_Obj(started_at="2026-10-04T10:00:00Z"), terminated=None))]
            elif p.status.phase == "Running":
                p._age += 1
                if p._age >= self.run_ticks:
                    p.status.phase = "Succeeded"
                    p.status.container_statuses = [_Obj(state=_Obj(running=None, terminated=_Obj(
                        started_at="2026-10-04T10:00:00Z", finished_at="2026-10-04T10:00:04Z")))]


class _NoCpu:
    paths = {}
    source = "none (test)"

    def total(self):
        return None


def _live(factory, policy_name, param, horizon=25, run_ticks=3):
    api = FakeAPI(run_ticks=run_ticks)
    sched = R.ArrivalSchedule(factory, SEEDS[0], horizon, prefix="e1-live")
    tmpl = factory()
    clock = iter(np.arange(0, 10_000, 0.25)).__next__
    be = B.LiveBackend(tmpl, run_id="e1-live", api=api, client_module=FakeClient,
                       cpu_probe=_NoCpu(), clock=clock, sleep=lambda s: None)
    pol = R.DefaultSchedulerPolicy() if policy_name == "default" else build(policy_name, param, tmpl)
    ctl = R.ReplayController(be, pol, sched, cond=UNI, template_env=tmpl, horizon=horizon)
    res = ctl.run()
    be.close()
    return api, res, ctl


def test_live_backend_end_to_end_against_a_fake_api(factory):
    api, res, ctl = _live(factory, "heur_leastloaded", 1.0)
    s = res.summary
    creates = [c for c in api.calls if c[0] == "create"]
    assert creates and all(c[2] == "pltr-ma-scheduler" for c in creates)
    assert any(c[0] == "bind" for c in api.calls)
    assert s["completed_tasks"] > 0
    assert s["completion_seconds_n"] > 0 and s["completion_seconds_mean"] == 4.0
    assert s["mean_cost"][0] == 0.0
    # cleanup: no pod of this run remains, the kube-system pod is untouched
    assert set(api.pods) == {"coredns"}


def test_live_retarget_updates_the_annotation_the_snapshot_reads(factory):
    api, res, ctl = _live(factory, "always-migrate", 0, horizon=6)
    patches = [c for c in api.calls if c[0] == "patch"]
    assert patches
    # the ledger and the annotation agree on the final routing of every patched pod
    for _, name, agent in patches:
        r = ctl.ledger.recs[name]
        if r.state == "pending":
            assert str(r.agent) == agent or any(
                p[1] == name and p[2] == str(r.agent) for p in patches)


def test_live_snapshot_subtracts_foreign_pods_in_other_namespaces():
    from observation import snapshot_from_kubernetes
    api = FakeAPI()
    snap = snapshot_from_kubernetes(0, namespace="pltr-p4", run_label="x", api=api)
    assert [n.foreign_cpu_request_millis for n in snap.nodes] == [0, 100, 0]
    assert snap.nodes[1].foreign_mem_request_bytes == 70 * B.MI
    assert snap.nodes[2].mem_allocatable_bytes == 1024 * B.MI


def test_live_default_scheduler_creates_default_scheduler_pods_and_never_binds(factory):
    api, res, _ = _live(factory, "default", 0, horizon=5)
    creates = [c for c in api.calls if c[0] == "create"]
    assert creates and all(c[2] == "default-scheduler" for c in creates)
    assert not any(c[0] == "bind" for c in api.calls)


def test_live_kubelet_rejection_is_counted_and_requeued(factory):
    api, res, ctl = _live(factory, "heur_local", 5.0, horizon=40, run_ticks=30)
    s = res.summary
    assert s["refused_kubelet"] > 0 and s["requeues"] == s["refused_kubelet"]


def test_live_backend_without_client_fails_loudly(factory):
    try:
        import kubernetes  # noqa: F401
        pytest.skip("kubernetes is installed here")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="kubernetes"):
        B.LiveBackend(factory(), run_id="x")


def test_cli_dry_run_writes_jsonl(tmp_path):
    out = tmp_path / "run.jsonl"
    rc = R.main(["--backend", "kube-model", "--policy", "heur_local:1.0",
                 "--seeds", "30000", "--horizon", "20", "--out", str(out), "--no-obs"])
    assert rc == 0
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [x["type"] for x in lines].count("tick") == 20
    assert lines[-1]["p3_code_sha256"]["cloud_env.py"]


def test_cli_refuses_uncalibrated_live_run():
    assert R.main(["--backend", "live", "--policy", "default", "--seeds", "1"]) == 2
