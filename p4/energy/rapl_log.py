#!/usr/bin/env python3
r"""rapl_log.py — energy logging for the P4 Kubernetes pilot, on this Windows host.

READ THIS FIRST
===============
**CPU package energy (RAPL) is NOT readable on this machine without installing
third-party software.** That was established by probing the host, not assumed. The
file name is kept because PLAN_P4_KUBERNETES.md §3 speaks of "RAPL if the host
exposes it"; the answer, measured on 2026-09-27, is that it does not.

What was probed, and what came back (host: DESKTOP-PO62NFO, 13th Gen Intel Core
i7-13620H, 10 cores / 16 threads, Windows 11 Pro 26200, non-elevated session):

| Source | Result |
|---|---|
| Intel RAPL MSRs (`MSR_PKG_ENERGY_STATUS`) | no user-space path on Windows; reading an MSR needs a ring-0 driver |
| `pyRAPL` python package | not installed (`find_spec` → None). It is Linux-only: it reads `/sys/class/powercap/intel-rapl` |
| `pythonnet` / `clr` (needed for LibreHardwareMonitorLib) | not installed |
| LibreHardwareMonitor / OpenHardwareMonitor / HWiNFO64 / Intel Power Gadget | none installed (all four install paths absent) |
| `powercfg /srumutil` (Windows energy estimation, SRUM) | **fails**: "An unexpected error (0x1f)… Error: SruUpdateStats failed, status 5" — and needs elevation |
| `\Processor Information(*)` perf counters | only `% Performance Limit` and `Performance Limit Flags`; **no energy or power counter** |
| `root\WMI` `EnergyMeter` class | not present |
| `\Power Meter(_Total)\Power` perf counter | **counter exists**, 2 instances; read **0** in 3 consecutive samples while on AC |
| `root\WMI` `BatteryStatus` | **readable**: `DischargeRate=0`, `Voltage=12569` mV, `Discharging=False`, `PowerOnline=True` while on AC |

So there are two *platform* power sources on this host, both whole-system and both
apparently zero on AC power, and no CPU-package source at all.

THE DECISION
------------
The implemented primary source is the **CPU-time proxy** — the fallback the task
brief names — because it is the only source that works with zero installs, on AC
power, and during a multi-minute cluster replay. The two battery/EMI sources are
implemented as *secondary, opt-in* samplers so that a future run on battery can
record them alongside, but they are **[UNVERIFIED]**: neither has ever returned a
non-zero reading on this machine, and whether they do on battery has not been
tested. No API is invented anywhere in this file; every source either works or
reports precisely why it does not.

WHAT THE PROXY IS AND IS NOT
----------------------------
It logs **CPU-seconds**, not Joules. It is a *work* measure. Claiming Joules from
it would require a Watts-per-CPU-second constant that nothing here can supply, and
`--assume-watts-per-cpu-second` exists only so that such a conversion, if ever
made, is explicit, flagged in the output, and traceable to whoever supplied the
number. By default no conversion is done and the units column says `cpu_seconds`.

Per-node energy is **impossible** in this pilot regardless of the source: the three
k3d "nodes" are containers on one physical package. See ../bridge/MAPPING.md G-4.

Usage
-----
    python rapl_log.py probe                       # what this host exposes, now
    python rapl_log.py log --label baseline --duration 60 --out base.csv
    python rapl_log.py log --label policy   --duration 60 --out pol.csv
    python rapl_log.py summarize --baseline base.csv --condition pol.csv

The measurement protocol (idle baseline, interleaved conditions, delta over
baseline) is in PROTOCOL.md and must be followed for any reported number.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

IS_WINDOWS = sys.platform.startswith("win")
PS = ["powershell", "-NoProfile", "-NonInteractive", "-Command"]


# --------------------------------------------------------------------------- #
# Source probes
# --------------------------------------------------------------------------- #
@dataclass
class ProbeResult:
    name: str
    available: bool
    detail: str
    sample: float | None = None
    unit: str = ""
    verdict: str = ""


def _powershell(cmd: str, timeout: float = 15.0) -> tuple[int, str, str]:
    if not IS_WINDOWS:
        return 1, "", "not Windows"
    try:
        p = subprocess.run(PS + [cmd], capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        return 1, "", f"{type(exc).__name__}: {exc}"


def probe_cpu_time() -> ProbeResult:
    """The primary source. Always available where psutil is."""
    if psutil is None:
        return ProbeResult("cpu-time-proxy", False,
                           "psutil is not installed (pip install psutil)",
                           verdict="UNAVAILABLE")
    t = psutil.cpu_times()
    busy = t.user + t.system
    return ProbeResult(
        "cpu-time-proxy", True,
        f"psutil {psutil.__version__}; cpu_count logical="
        f"{psutil.cpu_count(logical=True)} physical={psutil.cpu_count(logical=False)}; "
        f"cumulative busy since boot = {busy:.1f} s",
        sample=busy, unit="cpu_seconds",
        verdict="USABLE as a work proxy; NOT energy",
    )


def probe_rapl_msr() -> ProbeResult:
    """There is no non-privileged path to RAPL on Windows. Stated, not attempted."""
    detail = ("Intel RAPL is exposed through model-specific registers. Windows has "
              "no user-space MSR interface: reading MSR_PKG_ENERGY_STATUS (0x611) "
              "requires a kernel-mode driver (WinRing0 via LibreHardwareMonitorLib, "
              "or a vendor tool). pyRAPL reads /sys/class/powercap/intel-rapl and is "
              "Linux-only.")
    installed = []
    for p in (r"C:\Program Files\HWiNFO64",
              r"C:\Program Files\LibreHardwareMonitor",
              r"C:\Program Files\OpenHardwareMonitor",
              r"C:\Program Files (x86)\Intel\Power Gadget 3.6",
              r"C:\Program Files\Intel\Power Gadget 3.6"):
        if Path(p).exists():
            installed.append(p)
    if installed:
        detail += f" Candidate tools ARE present: {installed}. Not used by this module."
    else:
        detail += " No such tool is installed on this host (all five paths absent)."
    return ProbeResult("rapl-msr", False, detail,
                       verdict="UNAVAILABLE without installing a kernel-mode driver")


def probe_power_meter_counter() -> ProbeResult:
    """Windows '\\Power Meter(_Total)\\Power' performance counter.

    The counterset exists on this host with 2 instances but read 0 on AC power.
    Units are [UNVERIFIED] (Windows documents the EMI counter in milliwatts on some
    platforms and watts on others), which is exactly why nothing is converted here.
    """
    rc, out, err = _powershell(
        "try { (Get-Counter -Counter '\\Power Meter(_Total)\\Power' -MaxSamples 1 "
        "-ErrorAction Stop).CounterSamples[0].CookedValue } catch { 'ERR:' + "
        "$_.Exception.Message }")
    if rc != 0 or out.startswith("ERR:") or not out:
        return ProbeResult("power-meter-counter", False,
                           f"rc={rc} out={out!r} err={err!r}",
                           verdict="UNAVAILABLE")
    try:
        val = float(out.splitlines()[-1])
    except ValueError:
        return ProbeResult("power-meter-counter", False, f"unparsable: {out!r}",
                           verdict="UNAVAILABLE")
    verdict = ("PRESENT but reading 0 — on this host the counter reads 0 on AC "
               "power. Whole-system, not CPU package. Units [UNVERIFIED]."
               if val == 0 else
               "PRESENT and non-zero — whole-system power, NOT CPU package. "
               "Units [UNVERIFIED]: confirm against a wall meter before use.")
    return ProbeResult("power-meter-counter", val != 0,
                       f"counter read {val}", sample=val,
                       unit="[UNVERIFIED W or mW]", verdict=verdict)


def probe_battery() -> ProbeResult:
    """root\\WMI BatteryStatus — DischargeRate (mW) and Voltage (mV).

    Readable without elevation on this host. Reports 0 while on AC, which is
    expected: a battery that is not discharging has no discharge rate. Whether it
    returns a usable rate on battery is [UNVERIFIED] — untested.
    """
    rc, out, err = _powershell(
        "try { $b = Get-CimInstance -Namespace root\\WMI -ClassName BatteryStatus "
        "-ErrorAction Stop | Select-Object -First 1; "
        "@{ rate=$b.DischargeRate; volt=$b.Voltage; disch=[int]$b.Discharging; "
        "ac=[int]$b.PowerOnline } | ConvertTo-Json -Compress } "
        "catch { 'ERR:' + $_.Exception.Message }")
    if rc != 0 or out.startswith("ERR:") or not out:
        return ProbeResult("battery-dischargerate", False,
                           f"rc={rc} out={out!r} err={err!r}", verdict="UNAVAILABLE")
    try:
        d = json.loads(out.splitlines()[-1])
    except json.JSONDecodeError:
        return ProbeResult("battery-dischargerate", False, f"unparsable: {out!r}",
                           verdict="UNAVAILABLE")
    rate = float(d.get("rate") or 0.0)
    on_ac = bool(d.get("ac"))
    detail = (f"DischargeRate={rate} mW, Voltage={d.get('volt')} mV, "
              f"Discharging={bool(d.get('disch'))}, PowerOnline={on_ac}")
    if on_ac or rate == 0:
        verdict = ("READABLE but 0 — the host is on AC. To use this source the "
                   "laptop must run on battery, which also changes the CPU's "
                   "power/thermal policy and therefore changes the thing being "
                   "measured. Whole-system, not CPU package.")
        avail = False
    else:
        verdict = ("READABLE and non-zero: whole-system discharge power in mW. "
                   "NOT CPU package. Sampling granularity is set by the battery "
                   "firmware and is [UNVERIFIED] (often ~1 Hz or slower).")
        avail = True
    return ProbeResult("battery-dischargerate", avail, detail,
                       sample=rate, unit="mW", verdict=verdict)


def probe_all() -> list[ProbeResult]:
    return [probe_cpu_time(), probe_rapl_msr(),
            probe_power_meter_counter(), probe_battery()]


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
CSV_FIELDS = [
    "t_unix", "t_iso", "label", "wall_dt_s",
    "cpu_seconds_delta", "cpu_util_frac", "cpu_seconds_per_wall_second",
    "user_delta", "system_delta", "idle_delta", "cpu_freq_mhz",
    "battery_mw", "power_meter_raw", "units", "note",
]


@dataclass
class Sampler:
    """Interval sampler. Primary = CPU time; battery/EMI are opt-in and may be None."""

    label: str
    with_battery: bool = False
    with_power_meter: bool = False
    #: Requested sampling interval, used only to detect overshoot.
    interval: float = 1.0
    n_logical: int = field(default=0)
    _last_times: object | None = None
    _last_wall: float | None = None

    def __post_init__(self) -> None:
        if psutil is None:
            raise RuntimeError("psutil is required: pip install psutil")
        self.n_logical = psutil.cpu_count(logical=True) or 1
        self._last_times = psutil.cpu_times()
        self._last_wall = time.perf_counter()

    def sample(self) -> dict:
        now_wall = time.perf_counter()
        t = psutil.cpu_times()
        prev = self._last_times
        dt_wall = now_wall - (self._last_wall or now_wall)

        # Busy = user + system. On Windows psutil also reports `interrupt` and
        # `dpc`; whether those are already included in `system` is [UNVERIFIED],
        # so they are deliberately NOT added (adding them risks double counting).
        # They are recoverable from the raw deltas if ever needed.
        d_user = t.user - prev.user            # type: ignore[union-attr]
        d_sys = t.system - prev.system         # type: ignore[union-attr]
        d_idle = t.idle - prev.idle            # type: ignore[union-attr]
        busy = d_user + d_sys
        denom = busy + d_idle
        util = (busy / denom) if denom > 0 else float("nan")

        freq = None
        try:
            f = psutil.cpu_freq()
            freq = f.current if f else None
        except Exception:  # noqa: BLE001
            freq = None

        row = {
            "t_unix": f"{time.time():.3f}",
            "t_iso": dt.datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "label": self.label,
            "wall_dt_s": f"{dt_wall:.4f}",
            "cpu_seconds_delta": f"{busy:.4f}",
            "cpu_util_frac": "" if math.isnan(util) else f"{util:.6f}",
            "cpu_seconds_per_wall_second": f"{busy / dt_wall:.4f}" if dt_wall > 0 else "",
            "user_delta": f"{d_user:.4f}",
            "system_delta": f"{d_sys:.4f}",
            "idle_delta": f"{d_idle:.4f}",
            "cpu_freq_mhz": "" if freq is None else f"{freq:.0f}",
            "battery_mw": "",
            "power_meter_raw": "",
            "units": "cpu_seconds",
            "note": "",
        }

        if self.with_battery:
            pr = probe_battery()
            row["battery_mw"] = "" if pr.sample is None else f"{pr.sample:.0f}"
            if not pr.available:
                row["note"] = "battery source read 0/unavailable"
        if self.with_power_meter:
            pr = probe_power_meter_counter()
            row["power_meter_raw"] = "" if pr.sample is None else f"{pr.sample:.3f}"

        # The battery and power-meter sources each spawn a PowerShell process,
        # which costs ~1-2 s and burns CPU that this very logger then counts.
        # Measured 2026-09-28: interval=1.0 with both enabled produced
        # wall_dt_s=2.63. Flag it rather than let it silently inflate the proxy.
        if dt_wall > 1.5 * self.interval:
            row["note"] = ((row["note"] + "; ") if row["note"] else "") + (
                f"SAMPLING OVERSHOOT wall_dt={dt_wall:.2f}s vs interval="
                f"{self.interval:.2f}s — the sampler's own cost is in this number")

        self._last_times = t
        self._last_wall = now_wall
        return row


def run_log(out: Path, label: str, duration: float, interval: float,
            with_battery: bool, with_power_meter: bool) -> int:
    s = Sampler(label=label, with_battery=with_battery,
                with_power_meter=with_power_meter, interval=interval)
    new = not out.exists()
    n = 0
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        t_end = time.perf_counter() + duration
        # The first sample is discarded: its deltas span the interval since
        # Sampler construction, which is not an `interval`-long window.
        time.sleep(min(interval, max(0.0, duration)))
        s.sample()
        while time.perf_counter() < t_end:
            time.sleep(interval)
            w.writerow(s.sample())
            f.flush()
            n += 1
    print(f"wrote {n} samples (label={label!r}) to {out}", file=sys.stderr)
    return n


# --------------------------------------------------------------------------- #
# Summarising: delta over baseline
# --------------------------------------------------------------------------- #
def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _stats(vals: list[float]) -> tuple[float, float, int]:
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else float("nan")
    return m, sd, n


def summarize(baseline: Path, conditions: list[Path],
              watts_per_cpu_second: float | None = None) -> str:
    """Report each condition as a delta over the idle baseline (PROTOCOL.md §4)."""
    def rate(rows: list[dict]) -> list[float]:
        return [float(r["cpu_seconds_per_wall_second"])
                for r in rows if r.get("cpu_seconds_per_wall_second")]

    base_rows = _read(baseline)
    b_m, b_sd, b_n = _stats(rate(base_rows))

    lines = [
        "energy proxy summary — units: CPU-seconds per wall-second (NOT Joules)",
        f"  baseline file      : {baseline}",
        f"  baseline rate      : {b_m:.4f} +/- {b_sd:.4f}  (n={b_n} samples)",
        "",
    ]
    for c in conditions:
        rows = _read(c)
        by_label: dict[str, list[dict]] = {}
        for r in rows:
            by_label.setdefault(r.get("label", ""), []).append(r)
        for label, rr in by_label.items():
            m, sd, n = _stats(rate(rr))
            delta = m - b_m
            lines += [
                f"  condition          : {c.name} label={label!r}",
                f"    rate             : {m:.4f} +/- {sd:.4f}  (n={n})",
                f"    DELTA over base  : {delta:+.4f} cpu_seconds per wall_second",
                f"    total proxy work : {sum(float(r['cpu_seconds_delta']) for r in rr):.1f} "
                f"cpu_seconds over {sum(float(r['wall_dt_s']) for r in rr):.1f} s wall",
            ]
            if watts_per_cpu_second is not None:
                lines += [
                    f"    [ASSUMPTION] x {watts_per_cpu_second} W per cpu-second -> "
                    f"{delta * watts_per_cpu_second:+.3f} 'W' over baseline",
                    "    ^ this number is only as good as the constant you supplied; "
                    "record its source or mark it [UNVERIFIED].",
                ]
            # battery, if it was ever non-empty
            bat = [float(r["battery_mw"]) for r in rr if r.get("battery_mw")]
            if bat and max(bat) > 0:
                m2, sd2, n2 = _stats(bat)
                lines.append(f"    battery discharge: {m2:.0f} +/- {sd2:.0f} mW (n={n2}) "
                             "[whole system, NOT CPU package]")
            elif bat:
                lines.append("    battery discharge: all samples 0 (host on AC) — "
                             "unusable, see PROTOCOL.md")
            lines.append("")
    lines += [
        "Interpretation limits (PROTOCOL.md §5):",
        "  * CPU-seconds is a WORK measure. It is not energy and must never be",
        "    reported as Joules or Watts without a stated, sourced constant.",
        "  * It is whole-host: the three k3d nodes cannot be separated, and the",
        "    energy logger, Docker Desktop and Windows itself are all included.",
        "  * Frequency/turbo drift is not captured; cpu_freq_mhz on this host reads",
        "    a constant nominal 2400 MHz, so it does NOT track turbo. [UNVERIFIED]",
        "    whether any non-privileged source on this host tracks real frequency.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Energy logging for P4. Primary source is a CPU-time PROXY; "
                    "CPU package energy is not readable on this host without "
                    "installing a kernel-mode driver.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe", help="report what this host exposes, right now")
    pr.add_argument("--json", action="store_true")

    lg = sub.add_parser("log", help="sample into a CSV")
    lg.add_argument("--out", required=True)
    lg.add_argument("--label", required=True,
                    help="condition name, e.g. baseline | policy | kube-scheduler")
    lg.add_argument("--duration", type=float, required=True, help="seconds")
    lg.add_argument("--interval", type=float, default=1.0, help="seconds (default 1.0)")
    lg.add_argument("--with-battery", action="store_true",
                    help="also sample root\\WMI BatteryStatus (only useful on battery)")
    lg.add_argument("--with-power-meter", action="store_true",
                    help="also sample the '\\Power Meter(_Total)\\Power' counter")

    sm = sub.add_parser("summarize", help="delta of each condition over the baseline")
    sm.add_argument("--baseline", required=True)
    sm.add_argument("--condition", action="append", required=True)
    sm.add_argument("--assume-watts-per-cpu-second", type=float, default=None,
                    help="OPT-IN, off by default. Converts the proxy to pseudo-Watts "
                         "using a constant YOU must source. Output is flagged as an "
                         "assumption.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "probe":
        results = probe_all()
        if args.json:
            print(json.dumps([r.__dict__ for r in results], indent=2))
        else:
            print("energy source probe — host:", sys.platform)
            for r in results:
                mark = "OK " if r.available else "NO "
                print(f"\n[{mark}] {r.name}")
                print(f"      detail : {r.detail}")
                if r.sample is not None:
                    print(f"      sample : {r.sample} {r.unit}")
                print(f"      verdict: {r.verdict}")
            usable = [r.name for r in results if r.available]
            print("\nusable now:", ", ".join(usable) or "NONE")
            print("CPU PACKAGE ENERGY: not available on this host without installing "
                  "third-party software. Proceeding with the CPU-time proxy; see "
                  "PROTOCOL.md.")
        return 0

    if args.cmd == "log":
        run_log(Path(args.out), args.label, args.duration, args.interval,
                args.with_battery, args.with_power_meter)
        return 0

    print(summarize(Path(args.baseline), [Path(c) for c in args.condition],
                    args.assume_watts_per_cpu_second))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
