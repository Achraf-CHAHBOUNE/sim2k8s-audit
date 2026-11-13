"""Unit tests for trace_to_pods.py. No cluster, no Docker, no network.

Run:  python -m pytest k8s/replay/test_trace_to_pods.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trace_to_pods as ttp  # noqa: E402

REAL_TRACE = ttp.DEFAULT_TRACE


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def tiny_trace(tmp_path: Path) -> Path:
    """A 4-row trace with values chosen so every conversion is exact by hand."""
    p = tmp_path / "tiny.csv"
    p.write_text(
        "arrival_step,cpu,mem,work,slack\n"
        "0,0.1,0.2,1.0,10\n"
        "0,0.5,0.5,5.0,30\n"
        "3,0.25,0.125,2.5,17\n"
        "7,0.4,0.3,3.0,12\n",
        encoding="utf-8",
    )
    return p


def _requests(pod: dict) -> dict:
    return pod["spec"]["containers"][0]["resources"]["requests"]


# --------------------------------------------------------------------------- #
# N rows -> N manifests  (the requirement in the task)
# --------------------------------------------------------------------------- #
def test_n_rows_produce_n_manifests(tiny_trace: Path):
    rows = ttp.load_trace(tiny_trace)
    assert len(rows) == 4
    pods = ttp.rows_to_pods(rows, ttp.ScaleConfig())
    assert len(pods) == len(rows)
    # names are unique and stable
    names = [p["metadata"]["name"] for p in pods]
    assert names == ["task-000000", "task-000001", "task-000002", "task-000003"]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_n_rows_produce_n_manifests_for_any_prefix(tiny_trace: Path, n: int):
    rows = ttp.load_trace(tiny_trace, limit=n)
    assert len(ttp.rows_to_pods(rows, ttp.ScaleConfig())) == n


def test_n_rows_from_the_real_trace(tmp_path: Path):
    rows = ttp.load_trace(REAL_TRACE, limit=250)
    assert len(rows) == 250
    pods = ttp.rows_to_pods(rows, ttp.ScaleConfig())
    assert len(pods) == 250


# --------------------------------------------------------------------------- #
# Resource requests are correctly scaled
# --------------------------------------------------------------------------- #
def test_resource_requests_scaled_at_default_units(tiny_trace: Path):
    rows = ttp.load_trace(tiny_trace)
    cfg = ttp.ScaleConfig()               # 1000 millis / 512 Mi per sim unit
    pods = ttp.rows_to_pods(rows, cfg)

    expected = [
        ("100m", "102Mi"),   # 0.1*1000, round(0.2*512)=102.4 -> 102
        ("500m", "256Mi"),   # 0.5*1000, 0.5*512=256
        ("250m", "64Mi"),    # 0.25*1000, 0.125*512=64
        ("400m", "154Mi"),   # 0.4*1000, round(0.3*512)=153.6 -> 154
    ]
    got = [(_requests(p)["cpu"], _requests(p)["memory"]) for p in pods]
    assert got == expected


def test_resource_requests_scaled_at_custom_units(tiny_trace: Path):
    rows = ttp.load_trace(tiny_trace)
    cfg = ttp.ScaleConfig(cpu_unit_millis=2000, mem_unit_mi=1024)
    pods = ttp.rows_to_pods(rows, cfg)
    assert _requests(pods[0])["cpu"] == "200m"       # 0.1 * 2000
    assert _requests(pods[0])["memory"] == "205Mi"   # round(0.2 * 1024) = 204.8
    assert _requests(pods[1])["cpu"] == "1000m"      # 0.5 * 2000
    assert _requests(pods[1])["memory"] == "512Mi"


def test_scaling_is_proportional_on_the_real_trace():
    """Every emitted request must equal sim_units * unit_scale, exactly (rounded)."""
    rows = ttp.load_trace(REAL_TRACE, limit=500)
    cfg = ttp.ScaleConfig()
    pods = ttp.rows_to_pods(rows, cfg)
    for r, p in zip(rows, pods):
        assert _requests(p)["cpu"] == f"{int(round(r.cpu * cfg.cpu_unit_millis))}m"
        assert _requests(p)["memory"] == f"{int(round(r.mem * cfg.mem_unit_mi))}Mi"
        # and the annotation preserves the untruncated simulator value
        ann = p["metadata"]["annotations"]
        assert float(ann["pltr-ma.io/sim-cpu-units"]) == pytest.approx(r.cpu)
        assert float(ann["pltr-ma.io/sim-mem-units"]) == pytest.approx(r.mem)


def test_no_cpu_or_memory_limits_are_set(tiny_trace: Path):
    """A CPU limit would destroy the node-level contention that stands in for
    cloud_env's per-server `speed`. See MAPPING.md O-1 / O-7."""
    pods = ttp.rows_to_pods(ttp.load_trace(tiny_trace), ttp.ScaleConfig())
    for p in pods:
        assert "limits" not in p["spec"]["containers"][0]["resources"]


# --------------------------------------------------------------------------- #
# Deadline annotation derived from slack
# --------------------------------------------------------------------------- #
def test_deadline_annotation_matches_cloud_env_rule(tiny_trace: Path):
    """cloud_env.py:260 -- deadline_t = arrival_t + int(slack)."""
    rows = ttp.load_trace(tiny_trace)
    pods = ttp.rows_to_pods(rows, ttp.ScaleConfig(step_seconds=2.0))
    for r, p in zip(rows, pods):
        ann = p["metadata"]["annotations"]
        assert ann["pltr-ma.io/arrival-step"] == str(r.arrival_step)
        assert ann["pltr-ma.io/slack-steps"] == str(r.slack)
        assert int(ann["pltr-ma.io/deadline-step"]) == r.arrival_step + r.slack
        assert float(ann["pltr-ma.io/slack-seconds"]) == pytest.approx(r.slack * 2.0)


def test_active_deadline_is_off_by_default_and_warns_when_on(tiny_trace: Path):
    rows = ttp.load_trace(tiny_trace)
    off = ttp.rows_to_pods(rows, ttp.ScaleConfig())
    assert all("activeDeadlineSeconds" not in p["spec"] for p in off)

    on = ttp.rows_to_pods(rows, ttp.ScaleConfig(set_active_deadline=True, step_seconds=1.0))
    assert on[0]["spec"]["activeDeadlineSeconds"] == 10
    assert "active-deadline-warning" in " ".join(on[0]["metadata"]["annotations"])


# --------------------------------------------------------------------------- #
# Honesty stamps
# --------------------------------------------------------------------------- #
def test_manifests_are_stamped_uncalibrated_unless_told_otherwise(tiny_trace: Path):
    rows = ttp.load_trace(tiny_trace)
    default = ttp.rows_to_pods(rows, ttp.ScaleConfig())
    assert default[0]["metadata"]["annotations"]["pltr-ma.io/calibration"] == "UNCALIBRATED"

    calibrated = ttp.rows_to_pods(
        rows, ttp.ScaleConfig(iters_per_work_unit=4242, calibrated=True))
    assert calibrated[0]["metadata"]["annotations"]["pltr-ma.io/calibration"] == "CALIBRATED"
    assert calibrated[0]["metadata"]["annotations"]["pltr-ma.io/spin-iters"] == "4242"


def test_spin_iters_proportional_to_work(tiny_trace: Path):
    rows = ttp.load_trace(tiny_trace)
    cfg = ttp.ScaleConfig(iters_per_work_unit=1000, calibrated=True)
    pods = ttp.rows_to_pods(rows, cfg)
    for r, p in zip(rows, pods):
        iters = int(p["metadata"]["annotations"]["pltr-ma.io/spin-iters"])
        assert iters == int(round(r.work * 1000))
        assert str(iters) in p["spec"]["containers"][0]["command"][-1]


def test_initial_agent_is_labelled_as_synthesized(tiny_trace: Path):
    pods = ttp.rows_to_pods(ttp.load_trace(tiny_trace), ttp.ScaleConfig())
    for p in pods:
        ann = p["metadata"]["annotations"]
        assert ann["pltr-ma.io/initial-agent-origin"].startswith("SYNTHESIZED")
        assert 0 <= int(ann["pltr-ma.io/initial-agent"]) < 3


def test_initial_agent_routing_is_seed_reproducible(tiny_trace: Path):
    rows = ttp.load_trace(tiny_trace)
    a = ttp.assign_initial_agents(rows, 3, seed=7)
    b = ttp.assign_initial_agents(rows, 3, seed=7)
    c = ttp.assign_initial_agents(rows, 3, seed=8)
    assert a == b
    assert len(a) == len(rows)
    assert a != c or len(rows) < 2   # different seed should differ for >=2 rows


def test_pods_are_unbound_and_use_our_scheduler(tiny_trace: Path):
    pods = ttp.rows_to_pods(ttp.load_trace(tiny_trace), ttp.ScaleConfig())
    for p in pods:
        assert "nodeName" not in p["spec"]          # binding is the bridge's job
        assert p["spec"]["schedulerName"] == "pltr-ma-scheduler"
        assert p["spec"]["restartPolicy"] == "Never"


def test_baseline_run_can_target_the_default_scheduler(tiny_trace: Path):
    cfg = ttp.ScaleConfig(scheduler_name="default-scheduler")
    pods = ttp.rows_to_pods(ttp.load_trace(tiny_trace), cfg)
    assert pods[0]["spec"]["schedulerName"] == "default-scheduler"


# --------------------------------------------------------------------------- #
# Trace loading / filtering / schema
# --------------------------------------------------------------------------- #
def test_max_step_filter(tiny_trace: Path):
    rows = ttp.load_trace(tiny_trace, max_step=3)
    assert [r.arrival_step for r in rows] == [0, 0, 3]


def test_bad_header_is_rejected(tmp_path: Path):
    p = tmp_path / "bad.csv"
    p.write_text("step,cpu,mem,work,slack\n0,0.1,0.1,1.0,10\n", encoding="utf-8")
    with pytest.raises(ttp.TraceFormatError):
        ttp.load_trace(p)


def test_empty_file_is_rejected(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ttp.TraceFormatError):
        ttp.load_trace(p)


def test_real_trace_matches_the_documented_schema_and_ranges():
    """Guards the facts quoted in the module docstring and in p4/README.md."""
    rows = ttp.load_trace(REAL_TRACE)
    assert len(rows) == 6988
    assert min(r.arrival_step for r in rows) == 0
    assert max(r.arrival_step for r in rows) == 4999
    assert min(r.cpu for r in rows) == pytest.approx(0.1)
    assert max(r.cpu for r in rows) == pytest.approx(0.5)
    assert min(r.mem for r in rows) == pytest.approx(0.1)
    assert max(r.mem for r in rows) == pytest.approx(0.5)
    assert min(r.work for r in rows) == pytest.approx(1.0)
    assert max(r.work for r in rows) == pytest.approx(5.0)
    assert min(r.slack for r in rows) == 10
    assert max(r.slack for r in rows) == 30


# --------------------------------------------------------------------------- #
# Serialisation + CLI dry-run
# --------------------------------------------------------------------------- #
def test_yaml_round_trip(tiny_trace: Path):
    yaml = pytest.importorskip("yaml")
    pods = ttp.rows_to_pods(ttp.load_trace(tiny_trace), ttp.ScaleConfig())
    docs = [d for d in yaml.safe_load_all(ttp.to_yaml(pods)) if d]
    assert len(docs) == len(pods)
    assert docs[0]["kind"] == "Pod"
    assert docs[0]["spec"]["containers"][0]["resources"]["requests"]["cpu"] == "100m"


def test_cli_dry_run_writes_nothing(tiny_trace: Path, tmp_path: Path, capsys):
    target = tmp_path / "should-not-exist.yaml"
    rc = ttp.main(["--trace", str(tiny_trace), "--dry-run", "--out", str(target)])
    assert rc == 0
    assert not target.exists()
    out = capsys.readouterr()
    assert out.out.count("kind: Pod") == 4
    assert "UNCALIBRATED" in out.err


def test_cli_writes_file_when_asked(tiny_trace: Path, tmp_path: Path):
    target = tmp_path / "pods.yaml"
    rc = ttp.main(["--trace", str(tiny_trace), "--out", str(target), "--format", "json"])
    assert rc == 0
    assert target.exists()
    import json
    doc = json.loads(target.read_text(encoding="utf-8"))
    assert doc["kind"] == "List" and len(doc["items"]) == 4
