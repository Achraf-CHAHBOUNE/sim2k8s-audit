"""Tests for the observation and action bridges. No cluster, no Docker, no network.

The load-bearing test is `test_matches_cloud_env_build_obs`: it constructs a real
`CloudAllocationEnv`, translates its internal state into a `ClusterSnapshot`, and
asserts the bridge reproduces `_build_obs()` exactly. If that passes, the bridge's
arithmetic is the simulator's arithmetic.

Run:  python -m pytest p4/bridge/test_bridge.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "p4" / "controller"))
import _paths  # noqa: E402  (P3 tree, same rule everywhere)

import action as act                      # noqa: E402
import observation as obsmod              # noqa: E402
from observation import (                  # noqa: E402
    ClusterSnapshot, NodeState, PodState, UnitScale, build_observation,
)

MI = 1024 * 1024


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_nodes(caps=(1.0, 1.5, 2.0), scale: UnitScale | None = None) -> list[NodeState]:
    scale = scale or UnitScale()
    return [
        NodeState(
            name=f"k3d-pltr-p4-agent-{i}",
            index=i,
            cpu_allocatable_millis=int(c * scale.cpu_unit_millis),
            mem_allocatable_bytes=int(c * scale.mem_unit_bytes),
        )
        for i, c in enumerate(caps)
    ]


def make_pod(name: str, cpu: float, mem: float, work: float, arrival: int,
             slack: int, agent: int, node: int | None = None,
             phase: str = "Pending", scale: UnitScale | None = None) -> PodState:
    scale = scale or UnitScale()
    return PodState(
        name=name,
        cpu_request_millis=int(round(cpu * scale.cpu_unit_millis)),
        mem_request_bytes=int(round(mem * scale.mem_unit_bytes)),
        work_units=work,
        arrival_step=arrival,
        deadline_step=arrival + slack,
        initial_agent=agent,
        node_index=node,
        phase=phase,
    )


# --------------------------------------------------------------------------- #
# The table in MAPPING.md and the table in code must agree
# --------------------------------------------------------------------------- #
def test_obs_components_cover_all_nine_slots():
    assert len(obsmod.OBS_COMPONENTS) == obsmod.OBS_DIM == 9
    assert [c.index for c in obsmod.OBS_COMPONENTS] == list(range(9))
    assert all(c.category in ("observable", "approximable", "not-observable")
               for c in obsmod.OBS_COMPONENTS)


def test_component_names_match_cloud_env_docstring_order():
    """cloud_env.py:17-18 documents the slot order; drifting from it is a bug."""
    expected = ["cpu_util", "mem_util", "queue_backlog", "energy", "task_cpu",
                "task_mem", "task_proc", "task_deadline_slack", "has_task"]
    assert [c.name for c in obsmod.OBS_COMPONENTS] == expected

    src = (_paths.P3_CODE / "cloud_env.py").read_text(encoding="utf-8")
    for name in expected:
        assert name in src, f"{name} is not in cloud_env.py — the mapping is stale"


def test_mapping_md_documents_every_component():
    """MAPPING.md is the deliverable; code and doc must not drift."""
    md = (HERE / "MAPPING.md").read_text(encoding="utf-8")
    for c in obsmod.OBS_COMPONENTS:
        assert f"O-{c.index + 1}" in md, f"MAPPING.md has no row O-{c.index + 1}"
        assert f"`{c.name}`" in md, f"MAPPING.md never names `{c.name}`"


def test_mapping_md_documents_every_action():
    md = (HERE / "MAPPING.md").read_text(encoding="utf-8")
    for row, name in (("A-1", "ACCEPT"), ("A-2", "DEFER"), ("A-3", "MIGRATE")):
        assert row in md and name in md


def test_mapping_md_and_code_agree_on_the_not_observable_set():
    """The (c) list is the primary result; it must be identical in both places."""
    md = (HERE / "MAPPING.md").read_text(encoding="utf-8")
    not_obs = {c.name for c in obsmod.not_observable()}
    assert not_obs == {"energy", "has_task"}
    # The scoreboard line in MAPPING.md must name exactly these.
    scoreboard = md.split("## 0. Scoreboard", 1)[1].split("##", 1)[0]
    for name in not_obs:
        assert name in scoreboard
    assert "2 / 9" in scoreboard


def test_categorisation_counts():
    assert len(obsmod.directly_observable()) == 2
    assert len(obsmod.approximable()) == 5
    assert len(obsmod.not_observable()) == 2


# --------------------------------------------------------------------------- #
# THE load-bearing test: numerical equivalence with the simulator
# --------------------------------------------------------------------------- #
def _snapshot_from_env(env, scale: UnitScale) -> ClusterSnapshot:
    """Translate a live CloudAllocationEnv's internal state into a ClusterSnapshot.

    This is the inverse of the bridge and exists only for this test: it is how we
    check that the Kubernetes-shaped path computes what the simulator computes.
    """
    nodes = [
        NodeState(
            name=f"agent-{i}",
            index=i,
            cpu_allocatable_millis=int(round(s.cpu_cap * scale.cpu_unit_millis)),
            mem_allocatable_bytes=int(round(s.mem_cap * scale.mem_unit_bytes)),
        )
        for i, s in enumerate(env.servers)
    ]
    pods: list[PodState] = []
    seq = 0
    for i, s in enumerate(env.servers):
        for k, t in enumerate(s.running):
            pods.append(PodState(
                name=f"run-{i}-{k}",
                cpu_request_millis=t.cpu * scale.cpu_unit_millis,
                mem_request_bytes=t.mem * scale.mem_unit_bytes,
                work_units=t.work, arrival_step=t.arrival_t,
                deadline_step=t.deadline_t, initial_agent=i,
                node_index=i, phase="Running",
                remaining_work_units=t.remaining,
            ))
        for k, t in enumerate(s.arrivals):
            # enqueue_seq carries the simulator's INSERTION order. It cannot be
            # derived from arrival_t: MIGRATE appends to the destination's tail
            # (cloud_env.py:208), so after a migration the buffer is not in
            # arrival-time order. This is the state Kubernetes does not hold and
            # the replay controller must maintain (MAPPING.md O-9).
            pods.append(PodState(
                name=f"arr-{i}-{k:04d}",
                cpu_request_millis=t.cpu * scale.cpu_unit_millis,
                mem_request_bytes=t.mem * scale.mem_unit_bytes,
                work_units=t.work, arrival_step=t.arrival_t,
                deadline_step=t.deadline_t, initial_agent=i,
                node_index=None, phase="Pending",
                remaining_work_units=t.remaining,
                enqueue_seq=seq,
            ))
            seq += 1
    return ClusterSnapshot(step=env.t, nodes=nodes, pods=pods,
                           queue_cap=env.queue_cap, scale=scale)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_cloud_env_build_obs(seed: int):
    cloud_env = pytest.importorskip("cloud_env")
    scale = UnitScale()
    env = cloud_env.CloudAllocationEnv(n_servers=3, arrival_rate=1.5,
                                       horizon=60, seed=seed)
    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(40):
        want = env._build_obs()
        snap = _snapshot_from_env(env, scale)
        got, report = build_observation(snap)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
        # the flags must be truthful at every step, not just on average
        assert report.formula_slots == (3,)
        assert report.synthesized_slots == (8,)
        checked += 1
        _, _, _, trunc, _ = env.step(rng.integers(0, 3, size=3))
        if trunc:
            break
    assert checked >= 20   # the loop really ran


@pytest.mark.parametrize("seed", [0, 7])
def test_matches_cloud_env_global_state(seed: int):
    cloud_env = pytest.importorskip("cloud_env")
    env = cloud_env.CloudAllocationEnv(n_servers=3, arrival_rate=2.0, seed=seed)
    for _ in range(10):
        snap = _snapshot_from_env(env, UnitScale())
        got, _ = build_observation(snap)
        np.testing.assert_allclose(obsmod.global_state(got), env.global_state(),
                                   rtol=1e-5, atol=1e-6)
        env.step([0, 0, 1])


@pytest.mark.parametrize("seed", [0, 5])
def test_matches_cloud_env_constraint_costs(seed: int):
    """c_1..c_3 must be the same formulas (cloud_env._costs)."""
    cloud_env = pytest.importorskip("cloud_env")
    env = cloud_env.CloudAllocationEnv(n_servers=3, arrival_rate=2.0, seed=seed)
    for _ in range(10):
        want = env._costs(sla_step=2)
        snap = _snapshot_from_env(env, UnitScale())
        got = obsmod.constraint_costs(snap, sla_step=2)
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9)
        env.step([0, 1, 2])


# --------------------------------------------------------------------------- #
# Hand-computed observation values
# --------------------------------------------------------------------------- #
def test_hand_computed_observation():
    scale = UnitScale()
    nodes = make_nodes((1.0, 1.5, 2.0), scale)
    pods = [
        # agent 0: one running task (0.25 cpu, 0.25 mem) and one pending arrival
        make_pod("run-a", 0.25, 0.25, 4.0, 0, 20, agent=0, node=0, phase="Running"),
        make_pod("arr-a", 0.4, 0.2, 2.5, 5, 30, agent=0, node=None),
        # agent 1: nothing at all
        # agent 2: two running, no arrival
        make_pod("run-b", 0.5, 0.5, 1.0, 0, 25, agent=2, node=2, phase="Running"),
        make_pod("run-c", 0.5, 0.5, 1.0, 1, 25, agent=2, node=2, phase="Running"),
    ]
    snap = ClusterSnapshot(step=10, nodes=nodes, pods=pods, scale=scale)
    obs, report = build_observation(snap)

    # agent 0: cpu_util = 0.25/1.0 = 0.25 -> slot0 = 0.125
    assert obs[0][0] == pytest.approx(0.125)
    assert obs[0][1] == pytest.approx(0.125)
    assert obs[0][2] == pytest.approx(1 / 15)
    assert obs[0][3] == pytest.approx((0.1 + 0.9 * 0.25) / (0.1 + 0.9 * 2.0))
    assert obs[0][4] == pytest.approx(0.4)          # head arrival cpu
    assert obs[0][5] == pytest.approx(0.2)
    assert obs[0][6] == pytest.approx(2.5 / 5.0)    # remaining/max_work
    assert obs[0][7] == pytest.approx((35 - 10) / 30.0)   # deadline 5+30=35, t=10
    assert obs[0][8] == 1.0

    # agent 1: entirely empty -> util 0, energy = p_idle/(p_idle+2*p_dyn)
    assert obs[1][0] == 0.0 and obs[1][2] == 0.0 and obs[1][8] == 0.0
    assert obs[1][3] == pytest.approx(0.1 / 1.9)
    assert list(obs[1][4:8]) == [0.0, 0.0, 0.0, 0.0]

    # agent 2: cpu_util = 1.0/2.0 = 0.5 -> slot0 = 0.25; backlog 2; no arrival
    assert obs[2][0] == pytest.approx(0.25)
    assert obs[2][2] == pytest.approx(2 / 15)
    assert obs[2][8] == 0.0

    assert report.refused_regime["cpu_util>1_or_mem_util>1"] is False


def test_terminal_pods_leave_the_backlog():
    scale = UnitScale()
    nodes = make_nodes(scale=scale)
    pods = [
        make_pod("done", 0.5, 0.5, 1.0, 0, 20, agent=0, node=0, phase="Succeeded"),
        make_pod("failed", 0.5, 0.5, 1.0, 0, 20, agent=0, node=0, phase="Failed"),
        make_pod("live", 0.2, 0.2, 1.0, 0, 20, agent=0, node=0, phase="Running"),
    ]
    snap = ClusterSnapshot(step=1, nodes=nodes, pods=pods, scale=scale)
    obs, _ = build_observation(snap)
    assert obs[0][2] == pytest.approx(1 / 15)
    assert obs[0][0] == pytest.approx(0.1)          # 0.2/1.0 -> /2


def test_arrival_head_is_fifo_by_arrival_step():
    scale = UnitScale()
    nodes = make_nodes(scale=scale)
    pods = [
        make_pod("later", 0.5, 0.5, 5.0, 9, 10, agent=0),
        make_pod("earlier", 0.1, 0.1, 1.0, 2, 10, agent=0),
    ]
    snap = ClusterSnapshot(step=3, nodes=nodes, pods=pods, scale=scale)
    obs, _ = build_observation(snap)
    assert obs[0][4] == pytest.approx(0.1)   # the earlier arrival is the head


def test_enqueue_seq_overrides_arrival_time_for_the_head():
    """After a migration the buffer is NOT in arrival-time order (MAPPING.md O-9)."""
    scale = UnitScale()
    nodes = make_nodes(scale=scale)
    early = make_pod("migrated-in", 0.5, 0.5, 5.0, 1, 10, agent=0)
    late = make_pod("native", 0.1, 0.1, 1.0, 8, 10, agent=0)
    early.enqueue_seq = 99      # appended to the tail by a MIGRATE
    late.enqueue_seq = 4
    snap = ClusterSnapshot(step=9, nodes=nodes, pods=[early, late], scale=scale)
    obs, _ = build_observation(snap)
    assert obs[0][4] == pytest.approx(0.1)      # `native` is the head, not `migrated-in`


def test_retarget_moves_the_pod_to_the_destination_tail():
    snap = _three_agent_snap()
    for k, p in enumerate(snap.pods):
        p.enqueue_seq = k
    ex = act.SnapshotMutatingDryRunExecutor(snap)
    act.execute_step(snap, [act.MIGRATE, act.DEFER, act.DEFER], executor=ex)
    moved = next(p for p in snap.pods if p.name == "arr-0")
    assert moved.initial_agent == 2
    assert moved.enqueue_seq == max(p.enqueue_seq for p in snap.pods)


def test_clamping_and_over_one_flag_are_reported():
    """The simulator can reach util > 1; Kubernetes cannot (MAPPING.md O-1/G-9)."""
    scale = UnitScale()
    nodes = make_nodes((1.0,), scale)
    pods = [make_pod(f"p{i}", 0.5, 0.5, 1.0, 0, 30, agent=0, node=0,
                     phase="Running", scale=scale) for i in range(6)]
    snap = ClusterSnapshot(step=1, nodes=nodes, pods=pods, scale=scale)
    obs, report = build_observation(snap)
    assert obs[0][0] == pytest.approx(1.0)                 # min(3.0, 2.0)/2
    assert report.clamped.get("cpu_util") == 1
    assert report.refused_regime["cpu_util>1_or_mem_util>1"] is True


def test_staleness_warning():
    snap = ClusterSnapshot(step=1, nodes=make_nodes(), pods=[], max_info_age_s=12.0)
    _, report = build_observation(snap, max_info_age_budget_s=1.0)
    assert report.staleness_warning is not None
    assert "STALE" in report.staleness_warning
    assert "G-1" in report.staleness_warning


def test_foreign_requests_reduce_capacity():
    scale = UnitScale()
    nodes = make_nodes((1.0,), scale)
    nodes[0].foreign_cpu_request_millis = 500      # kube-system pods
    pods = [make_pod("p", 0.25, 0.1, 1.0, 0, 20, agent=0, node=0, phase="Running")]
    snap = ClusterSnapshot(step=1, nodes=nodes, pods=pods, scale=scale)
    obs, _ = build_observation(snap)
    assert obs[0][0] == pytest.approx(0.25 / 0.5 / 2)      # cap shrank to 0.5


def test_non_positive_capacity_raises_instead_of_guessing():
    nodes = make_nodes((1.0,))
    nodes[0].foreign_cpu_request_millis = 5000
    snap = ClusterSnapshot(step=0, nodes=nodes, pods=[])
    with pytest.raises(ValueError, match="non-positive simulator capacity"):
        build_observation(snap)


def test_unit_scale_mismatch_is_caught():
    scale = UnitScale()
    scale.assert_consistent_with(1000, 512)          # ok
    with pytest.raises(ValueError, match="unit scale mismatch"):
        scale.assert_consistent_with(2000, 512)


# --------------------------------------------------------------------------- #
# Action bridge
# --------------------------------------------------------------------------- #
def _three_agent_snap(step: int = 5) -> ClusterSnapshot:
    scale = UnitScale()
    nodes = make_nodes((1.0, 1.5, 2.0), scale)
    pods = [
        make_pod("arr-0", 0.2, 0.2, 2.0, 1, 20, agent=0),
        make_pod("arr-1", 0.3, 0.2, 2.0, 1, 20, agent=1),
        make_pod("arr-2", 0.1, 0.1, 1.0, 2, 20, agent=2),
        make_pod("run-1", 0.3, 0.3, 2.0, 0, 30, agent=1, node=1, phase="Running"),
        make_pod("run-1b", 0.3, 0.3, 2.0, 0, 30, agent=1, node=1, phase="Running"),
    ]
    return ClusterSnapshot(step=step, nodes=nodes, pods=pods, scale=scale)


def test_accept_plans_a_binding_to_its_own_node():
    snap = _three_agent_snap()
    plan = act.plan_step(snap, [act.ACCEPT, act.DEFER, act.DEFER])
    binds = [o for o in plan.ops if o.kind == "bind"]
    assert len(binds) == 1
    assert binds[0].pod == "arr-0"
    assert binds[0].target_node == "k3d-pltr-p4-agent-0"
    assert binds[0].calls_api is True


def test_defer_plans_no_api_call_but_is_still_logged():
    snap = _three_agent_snap()
    plan = act.plan_step(snap, [act.DEFER, act.DEFER, act.DEFER])
    assert len(plan.ops) == 3
    assert all(o.kind == "noop-defer" and o.calls_api is False for o in plan.ops)
    assert plan.migrate_count == 0


def test_migrate_retargets_and_is_not_an_eviction():
    """MAPPING.md A-3: cloud_env.MIGRATE acts on an UNBOUND task."""
    snap = _three_agent_snap()
    plan = act.plan_step(snap, [act.MIGRATE, act.DEFER, act.DEFER])
    ops = [o for o in plan.ops if o.kind == "retarget"]
    assert len(ops) == 1 and plan.migrate_count == 1
    assert not any(o.kind == "evict" for o in plan.ops)
    assert "no eviction" in ops[0].reason
    # agent 1 has 2 bound pods, agent 2 has none -> destination is agent 2
    assert ops[0].target_node == "k3d-pltr-p4-agent-2"


def test_least_loaded_other_matches_cloud_env_rule():
    """cloud_env._least_loaded_other: min backlog COUNT, self excluded, ties -> lowest index."""
    snap = _three_agent_snap()
    assert act.least_loaded_other(snap, 0) == 2      # agent1 has 2 bound, agent2 has 0
    assert act.least_loaded_other(snap, 1) == 0      # agent0 and agent2 both 0 -> 0
    assert act.least_loaded_other(snap, 2) == 0


def test_least_loaded_other_agrees_with_the_simulator_on_random_states():
    cloud_env = pytest.importorskip("cloud_env")
    env = cloud_env.CloudAllocationEnv(n_servers=3, arrival_rate=2.0, seed=3)
    rng = np.random.default_rng(3)
    for _ in range(25):
        snap = _snapshot_from_env(env, UnitScale())
        for i in range(3):
            assert act.least_loaded_other(snap, i) == env._least_loaded_other(i)
        env.step(rng.integers(0, 3, size=3))


def test_agent_with_empty_buffer_is_a_logged_noop():
    scale = UnitScale()
    nodes = make_nodes((1.0, 1.5, 2.0), scale)
    snap = ClusterSnapshot(step=1, nodes=nodes, pods=[], scale=scale)
    plan = act.plan_step(snap, [act.ACCEPT, act.MIGRATE, act.DEFER])
    assert plan.ops == []
    assert plan.idle_agents == (0, 1, 2)


def test_invalid_action_is_rejected():
    snap = _three_agent_snap()
    with pytest.raises(ValueError, match="no reject"):
        act.plan_step(snap, [3, 0, 0])


def test_expiry_ops_cover_bound_and_pending_pods():
    scale = UnitScale()
    nodes = make_nodes((1.0,), scale)
    pods = [
        make_pod("old-bound", 0.2, 0.2, 1.0, 0, 10, agent=0, node=0, phase="Running"),
        make_pod("old-pending", 0.2, 0.2, 1.0, 0, 10, agent=0),
        make_pod("fresh", 0.2, 0.2, 1.0, 10, 30, agent=0),
    ]
    snap = ClusterSnapshot(step=11, nodes=nodes, pods=pods, scale=scale)   # deadline 10
    ops = act.plan_expiry(snap)
    assert {o.pod for o in ops} == {"old-bound", "old-pending"}
    assert all("OUR rule" in o.reason for o in ops)


def test_dry_run_executor_touches_nothing_and_logs_everything(tmp_path):
    snap = _three_agent_snap()
    sink = (tmp_path / "ops.log").open("w")
    ex = act.DryRunExecutor(sink=sink)
    plan = act.execute_step(snap, [act.ACCEPT, act.MIGRATE, act.DEFER], executor=ex)
    sink.close()
    text = (tmp_path / "ops.log").read_text()
    assert text.count("DRY-RUN") == len(plan.ops)
    assert "bind" in text and "retarget" in text and "noop-defer" in text
    # nothing was applied, and the snapshot was not mutated
    assert all(o.applied is False for o in ex.applied)
    assert all(p.node_index is None for p in snap.pods if p.name.startswith("arr"))


def test_dry_run_jsonl_is_machine_readable(tmp_path):
    snap = _three_agent_snap()
    sink = (tmp_path / "ops.jsonl").open("w")
    act.execute_step(snap, [act.ACCEPT, act.ACCEPT, act.ACCEPT],
                     executor=act.DryRunExecutor(sink=sink, jsonl=True))
    sink.close()
    import json
    rows = [json.loads(l) for l in (tmp_path / "ops.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert {r["kind"] for r in rows} == {"bind"}


def test_snapshot_mutating_dry_run_drives_a_whole_loop_offline():
    snap = _three_agent_snap()
    ex = act.SnapshotMutatingDryRunExecutor(snap)
    act.execute_step(snap, [act.ACCEPT, act.DEFER, act.MIGRATE], executor=ex)
    by_name = {p.name: p for p in snap.pods}
    assert by_name["arr-0"].node_index == 0          # bound
    assert by_name["arr-1"].node_index is None       # still pending
    assert by_name["arr-2"].initial_agent == 0       # retargeted, still unbound
    assert by_name["arr-2"].node_index is None
    # and the observation now reflects the binding
    obs, _ = build_observation(snap)
    assert obs[0][2] == pytest.approx(1 / 15)


def test_plan_summary_reports_the_simulator_energy_charge():
    snap = _three_agent_snap()
    plan = act.plan_step(snap, [act.MIGRATE, act.MIGRATE, act.DEFER])
    assert plan.migrate_count == 2
    assert "0.05" in plan.summary()


def test_live_executor_is_unavailable_without_the_kubernetes_package():
    """Guards the claim in p4/README.md that no dry-run path needs the client."""
    pytest.importorskip("pytest")
    try:
        import kubernetes  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="kubernetes"):
            act.LiveExecutor()
    else:
        pytest.skip("kubernetes package is installed; cannot test the guard")
