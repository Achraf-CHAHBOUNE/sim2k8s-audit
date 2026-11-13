"""Tests for rapl_log.py. Pure functions plus one short live sampling smoke test.

Run:  python -m pytest p4/energy/test_rapl_log.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rapl_log as rl  # noqa: E402


# --------------------------------------------------------------------------- #
# Probes: the honest-reporting contract
# --------------------------------------------------------------------------- #
def test_rapl_is_reported_unavailable_and_says_why():
    """The central negative finding must never silently flip to 'available'."""
    r = rl.probe_rapl_msr()
    assert r.available is False
    assert "kernel-mode driver" in r.verdict
    assert "MSR" in r.detail


def test_probe_all_returns_every_source_with_a_verdict():
    results = rl.probe_all()
    names = {r.name for r in results}
    assert names == {"cpu-time-proxy", "rapl-msr",
                     "power-meter-counter", "battery-dischargerate"}
    assert all(r.verdict for r in results)


@pytest.mark.skipif(rl.psutil is None, reason="psutil not installed")
def test_cpu_time_proxy_is_available_and_labelled_as_not_energy():
    r = rl.probe_cpu_time()
    assert r.available is True
    assert r.unit == "cpu_seconds"
    assert "NOT energy" in r.verdict


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_stats_mean_sd_n():
    m, sd, n = rl._stats([1.0, 2.0, 3.0])
    assert (m, n) == (2.0, 3)
    assert sd == pytest.approx(1.0)


def test_stats_single_sample_has_no_sd():
    m, sd, n = rl._stats([5.0])
    assert m == 5.0 and n == 1 and math.isnan(sd)


def test_stats_empty():
    m, sd, n = rl._stats([])
    assert n == 0 and math.isnan(m) and math.isnan(sd)


# --------------------------------------------------------------------------- #
# summarize: delta over baseline
# --------------------------------------------------------------------------- #
def _write_csv(path: Path, label: str, rates: list[float]) -> None:
    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rl.CSV_FIELDS)
        w.writeheader()
        for i, r in enumerate(rates):
            row = {k: "" for k in rl.CSV_FIELDS}
            row.update({
                "t_unix": f"{1000 + i}", "label": label, "wall_dt_s": "1.0",
                "cpu_seconds_delta": f"{r}", "cpu_seconds_per_wall_second": f"{r}",
                "units": "cpu_seconds", "battery_mw": "0",
            })
            w.writerow(row)


def test_summarize_reports_delta_over_baseline(tmp_path: Path):
    base = tmp_path / "base.csv"
    cond = tmp_path / "cond.csv"
    _write_csv(base, "baseline", [1.0, 1.0, 1.0])
    _write_csv(cond, "policy", [3.0, 3.0, 3.0])
    out = rl.summarize(base, [cond])
    assert "DELTA over base  : +2.0000" in out
    assert "NOT Joules" in out
    assert "policy" in out


def test_summarize_flags_all_zero_battery_as_unusable(tmp_path: Path):
    base = tmp_path / "b.csv"
    cond = tmp_path / "c.csv"
    _write_csv(base, "baseline", [1.0])
    _write_csv(cond, "policy", [2.0])
    out = rl.summarize(base, [cond])
    assert "host on AC" in out


def test_watts_conversion_is_opt_in_and_flagged(tmp_path: Path):
    base = tmp_path / "b.csv"
    cond = tmp_path / "c.csv"
    _write_csv(base, "baseline", [1.0])
    _write_csv(cond, "policy", [2.0])

    plain = rl.summarize(base, [cond])
    assert "ASSUMPTION" not in plain          # off by default

    converted = rl.summarize(base, [cond], watts_per_cpu_second=10.0)
    assert "[ASSUMPTION]" in converted
    assert "+10.000" in converted
    assert "only as good as the constant you supplied" in converted


def test_summarize_separates_interleaved_labels_in_one_file(tmp_path: Path):
    """PROTOCOL.md requires interleaved blocks; they may share a file."""
    import csv
    base = tmp_path / "b.csv"
    _write_csv(base, "baseline", [1.0, 1.0])
    cond = tmp_path / "interleaved.csv"
    _write_csv(cond, "policy", [4.0, 4.0])
    with cond.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rl.CSV_FIELDS)
        for r in (2.0, 2.0):
            row = {k: "" for k in rl.CSV_FIELDS}
            row.update({"label": "kube-scheduler", "wall_dt_s": "1.0",
                        "cpu_seconds_delta": f"{r}",
                        "cpu_seconds_per_wall_second": f"{r}"})
            w.writerow(row)
    out = rl.summarize(base, [cond])
    assert "'policy'" in out and "'kube-scheduler'" in out
    assert "+3.0000" in out and "+1.0000" in out


# --------------------------------------------------------------------------- #
# Live sampling smoke test (short, no installs, no cluster)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(rl.psutil is None, reason="psutil not installed")
def test_sampler_produces_a_complete_row():
    s = rl.Sampler(label="t", interval=0.2)
    import time
    time.sleep(0.25)
    row = s.sample()
    assert set(row) == set(rl.CSV_FIELDS)
    assert float(row["cpu_seconds_delta"]) >= 0.0
    assert float(row["wall_dt_s"]) > 0.0
    assert row["units"] == "cpu_seconds"
    assert row["label"] == "t"


@pytest.mark.skipif(rl.psutil is None, reason="psutil not installed")
def test_log_appends_and_writes_one_header(tmp_path: Path):
    out = tmp_path / "e.csv"
    rl.run_log(out, "baseline", duration=1.2, interval=0.2,
               with_battery=False, with_power_meter=False)
    rl.run_log(out, "policy", duration=1.2, interval=0.2,
               with_battery=False, with_power_meter=False)
    text = out.read_text(encoding="utf-8")
    assert text.count("t_unix,t_iso,label") == 1     # header written once
    assert "baseline" in text and "policy" in text


@pytest.mark.skipif(rl.psutil is None, reason="psutil not installed")
def test_overshoot_is_flagged_not_hidden():
    """A sampler whose own cost dominates the interval must say so."""
    s = rl.Sampler(label="t", interval=0.001)
    import time
    time.sleep(0.05)
    row = s.sample()
    assert "SAMPLING OVERSHOOT" in row["note"]
