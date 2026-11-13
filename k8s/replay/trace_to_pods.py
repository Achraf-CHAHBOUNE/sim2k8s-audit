#!/usr/bin/env python3
"""trace_to_pods.py — turn the P3 Alibaba trace into Kubernetes Pod manifests.

Reads ``simulator/trace_alibaba.csv`` — the SAME file that drives the simulator
through ``simulator/trace.py`` — and emits one Pod manifest per trace row, so that
the cluster replay and the simulator episodes consume identical input. Nothing
here contacts a cluster; ``--dry-run`` (the default output mode is stdout) prints
manifests and a scaling report without Kubernetes, Docker or network.

Verified trace facts (measured 2026-09-27 from the file itself, not assumed):

    header      arrival_step,cpu,mem,work,slack
    rows        6988 data rows
    arrival_step 0 .. 4999   (3426 distinct steps populated, max 10 rows/step)
    cpu          0.1   .. 0.5
    mem          0.1   .. 0.5
    work         1.0   .. 5.0
    slack        10.0  .. 30.0

Those ranges match ``cloud_env.CloudAllocationEnv.task_cpu_range`` etc.
(cloud_env.py:135-138), which is why the trace drops into the env without
rescaling.

WHAT THIS SCRIPT DOES **NOT** CLAIM
-----------------------------------
* ``work`` is not observable in Kubernetes; it is *prescribed*. The generated pod
  burns a fixed number of shell-loop iterations. The constant that converts one
  simulator work unit into iterations is a CALIBRATION, not a measurement, and
  it is UNCALIBRATED until someone runs the procedure in ``--help-calibration``.
  Until then every manifest carries ``pltr-ma.io/calibration: UNCALIBRATED``.
* ``slack`` becomes an *annotation*. Kubernetes has no deadline concept and will
  not act on it. Expiry (``cloud_env._expire_overdue``, cloud_env.py:281-300)
  must be implemented by the replay controller. See ../bridge/MAPPING.md O-8.
* The per-server routing of a new arrival (cloud_env.py:261, uniform random
  server) has no Kubernetes counterpart — pending pods are one global queue.
  It is SYNTHESIZED here into ``pltr-ma.io/initial-agent`` so the bridge can
  reproduce the simulator's per-agent arrival buffers. See MAPPING.md O-9.

Usage
-----
    python trace_to_pods.py --dry-run --limit 5
    python trace_to_pods.py --max-step 150 --out /tmp/pods.yaml
    python trace_to_pods.py --help-calibration
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants tied to the simulator. Change these only together with cloud_env.py.
# --------------------------------------------------------------------------- #
TRACE_HEADER = ["arrival_step", "cpu", "mem", "work", "slack"]

#: Default relative path from this file to the trace the simulator uses.
#: P3 revision_v2 copy; byte-identical to simulator/trace_alibaba.csv (sha256 aab04247...).
DEFAULT_TRACE = Path(__file__).resolve().parents[2] / "simulator" / "trace_alibaba.csv"

ANNOT_PREFIX = "pltr-ma.io"

#: Iterations of the busybox arithmetic loop per simulator work unit.
#: [UNVERIFIED] placeholder. Not measured on any machine. See --help-calibration.
DEFAULT_ITERS_PER_WORK_UNIT = 1_000_000

CALIBRATION_HELP = """\
Calibrating --iters-per-work-unit
=================================

A simulator task has `work` work units and a server drains `speed` work units per
step (cloud_env.py:302-320). Kubernetes has no such notion: a pod runs until its
container exits. To keep the mapping honest the container is made CPU-bound with a
FIXED INSTRUCTION COUNT, so its wall-clock duration is an *emergent* consequence of
the CPU share the node gives it -- which is the closest available analogue of
`speed`. Nothing is faked with `sleep`.

That requires one calibration constant. Procedure (run ONCE per host, record the
result with the date in p4/README.md):

  1. Bring up the cluster (p4/cluster/up.sh --create).
  2. Run a single calibration pod on agent-2 (the fastest node) with a known
     iteration count and no other workload pods present:

       kubectl run calib --image=busybox:1.36 --restart=Never \\
         --overrides='{"spec":{"nodeName":"k3d-pltr-p4-agent-2"}}' -- \\
         sh -c 'i=0; while [ $i -lt 1000000 ]; do i=$((i+1)); done'

  3. Measure the elapsed time from the pod's status:
       kubectl get pod calib -o jsonpath='{.status.containerStatuses[0].state.terminated.startedAt} {.status.containerStatuses[0].state.terminated.finishedAt}'
  4. Repeat >= 5 times, take the median, and derive:
       iters_per_work_unit = 1e6 * (target_seconds_per_work_unit / median_seconds)
     where target_seconds_per_work_unit follows from your chosen --step-seconds
     and the simulator's base speed of 1.0 work unit per step on agent-0.
  5. Pass the result as --iters-per-work-unit and record it. Manifests generated
     with the default value are stamped UNCALIBRATED and must not be used for any
     reported measurement.

Known threats to this calibration, to state in the paper:
  * busybox `sh` arithmetic is interpreted; its throughput depends on the CPU's
    turbo/thermal state, which on a laptop drifts over a long replay.
  * Three node containers share one physical package, so a pod's duration depends
    on what the other two nodes are doing. The simulator's servers are
    independent. This is a transfer-gap source, not a bug to hide.
"""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TraceRow:
    """One row of trace_alibaba.csv, in simulator units."""

    arrival_step: int
    cpu: float          # simulator CPU units, 0.1 .. 0.5
    mem: float          # simulator memory units, 0.1 .. 0.5
    work: float         # simulator work units, 1.0 .. 5.0
    slack: int          # steps from arrival to deadline, 10 .. 30

    @property
    def deadline_step(self) -> int:
        """Matches cloud_env.py:260 -- deadline_t = t + int(slack)."""
        return self.arrival_step + int(self.slack)


@dataclass
class ScaleConfig:
    """Simulator-unit -> Kubernetes-unit conversion.

    The defaults are chosen so that the fleet of cloud_env.py:124-125
    (cpu_cap = mem_cap = [1.0, 1.5, 2.0]) lands on node allocatable that a 16-
    thread / 16 GB laptop can actually host: 1000m/1500m/2000m CPU and
    512Mi/768Mi/1024Mi memory. See ../cluster/k3d-cluster.yaml.
    """

    cpu_unit_millis: int = 1000          # 1.0 sim CPU unit -> 1000 milli-CPU
    mem_unit_mi: int = 512               # 1.0 sim memory unit -> 512 MiB
    step_seconds: float = 1.0            # wall-clock seconds per simulator step
    iters_per_work_unit: int = DEFAULT_ITERS_PER_WORK_UNIT
    calibrated: bool = False             # True only if the user passed the flag
    image: str = "busybox:1.36"
    #: None = Kubernetes default. The live campaign sets "Never" with the image
    #: pinned BY DIGEST and pre-loaded on every agent (protocol v1 §11, warm cache).
    image_pull_policy: str | None = None
    namespace: str = "pltr-p4"
    scheduler_name: str = "pltr-ma-scheduler"
    run_id: str = "run0"
    n_agents: int = 3
    route_seed: int = 0
    set_active_deadline: bool = False
    name_prefix: str = "task"

    def cpu_millis(self, sim_cpu: float) -> int:
        return int(round(sim_cpu * self.cpu_unit_millis))

    def mem_mi(self, sim_mem: float) -> int:
        return int(round(sim_mem * self.mem_unit_mi))

    def spin_iters(self, work: float) -> int:
        return int(round(work * self.iters_per_work_unit))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
class TraceFormatError(ValueError):
    pass


def load_trace(path: str | Path, max_step: int | None = None,
               limit: int | None = None) -> list[TraceRow]:
    """Read the trace CSV, validating the header against TRACE_HEADER.

    `max_step` keeps only rows with arrival_step <= max_step (an episode window);
    `limit` caps the number of rows returned after that filter.
    """
    path = Path(path)
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise TraceFormatError(f"{path} is empty")
        if [h.strip() for h in header] != TRACE_HEADER:
            raise TraceFormatError(
                f"{path}: expected header {TRACE_HEADER}, got {header}. "
                "This script is tied to the schema documented in simulator/trace.py."
            )
        rows: list[TraceRow] = []
        for lineno, rec in enumerate(reader, start=2):
            if not rec:
                continue
            if len(rec) != len(TRACE_HEADER):
                raise TraceFormatError(f"{path}:{lineno}: expected 5 fields, got {len(rec)}")
            step = int(float(rec[0]))
            if max_step is not None and step > max_step:
                continue
            rows.append(TraceRow(
                arrival_step=step,
                cpu=float(rec[1]),
                mem=float(rec[2]),
                work=float(rec[3]),
                slack=int(float(rec[4])),
            ))
            if limit is not None and len(rows) >= limit:
                break
    return rows


# --------------------------------------------------------------------------- #
# Simulator-faithful arrival routing (SYNTHESIZED -- see module docstring)
# --------------------------------------------------------------------------- #
def assign_initial_agents(rows: list[TraceRow], n_agents: int, seed: int) -> list[int]:
    """Reproduce cloud_env.py:261 -- each arrival goes to a uniformly random server.

    Uses numpy's Generator.integers with the same call shape as the env so the
    routing is reproducible from (seed, row order). This is NOT a Kubernetes
    behaviour: in Kubernetes a pending pod belongs to no node until it is bound.
    The simulator's per-agent arrival buffers are an artefact of the simulator,
    and this function is where that artefact is made explicit.
    """
    import numpy as np  # local import: the rest of this module is dependency-free

    rng = np.random.default_rng(seed)
    return [int(rng.integers(n_agents)) for _ in rows]


# --------------------------------------------------------------------------- #
# Manifest construction
# --------------------------------------------------------------------------- #
def row_to_pod(row: TraceRow, index: int, cfg: ScaleConfig,
               initial_agent: int) -> dict:
    """One trace row -> one Pod manifest (a plain dict, ready for YAML or JSON).

    The pod is created UNBOUND (no nodeName) with a non-default schedulerName, so
    the default kube-scheduler ignores it and the action bridge owns the binding
    decision. For the kube-scheduler BASELINE run, pass
    --scheduler-name default-scheduler.
    """
    name = f"{cfg.name_prefix}-{index:06d}"
    cpu_m = cfg.cpu_millis(row.cpu)
    mem_mi = cfg.mem_mi(row.mem)
    iters = cfg.spin_iters(row.work)

    annotations = {
        # --- traceability back to the simulator's own units ----------------- #
        f"{ANNOT_PREFIX}/trace-row": str(index),
        f"{ANNOT_PREFIX}/arrival-step": str(row.arrival_step),
        f"{ANNOT_PREFIX}/sim-cpu-units": f"{row.cpu:.6g}",
        f"{ANNOT_PREFIX}/sim-mem-units": f"{row.mem:.6g}",
        f"{ANNOT_PREFIX}/work-units": f"{row.work:.6g}",
        # --- deadline. RELATIVE ONLY. ---------------------------------------- #
        # An absolute wall-clock deadline cannot live in a static manifest: it
        # depends on the replay's t0, which is not known at generation time. The
        # replay controller stamps <prefix>/deadline-unix-ms when it admits the
        # pod. Kubernetes itself never reads any of these.
        f"{ANNOT_PREFIX}/slack-steps": str(row.slack),
        f"{ANNOT_PREFIX}/deadline-step": str(row.deadline_step),
        f"{ANNOT_PREFIX}/slack-seconds": f"{row.slack * cfg.step_seconds:.6g}",
        # --- prescribed, not observed --------------------------------------- #
        f"{ANNOT_PREFIX}/spin-iters": str(iters),
        f"{ANNOT_PREFIX}/calibration": "CALIBRATED" if cfg.calibrated else "UNCALIBRATED",
        # --- simulator artefact, made explicit ------------------------------ #
        f"{ANNOT_PREFIX}/initial-agent": str(initial_agent),
        f"{ANNOT_PREFIX}/initial-agent-origin": "SYNTHESIZED-from-cloud_env.py:261",
    }

    pod: dict = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": cfg.namespace,
            "labels": {
                "app": "pltr-ma-task",
                f"{ANNOT_PREFIX}/run": cfg.run_id,
                f"{ANNOT_PREFIX}/arrival-step": str(row.arrival_step),
                f"{ANNOT_PREFIX}/initial-agent": str(initial_agent),
            },
            "annotations": annotations,
        },
        "spec": {
            "schedulerName": cfg.scheduler_name,
            "restartPolicy": "Never",
            # Fast teardown: MIGRATE/expiry must not wait 30s for SIGTERM grace.
            "terminationGracePeriodSeconds": 0,
            "containers": [{
                "name": "burn",
                "image": cfg.image,
                **({"imagePullPolicy": cfg.image_pull_policy} if cfg.image_pull_policy else {}),
                # Fixed instruction count -> duration is set by the CPU share the
                # node grants, which is the only available analogue of the
                # simulator's per-server `speed`. NOT a sleep.
                "command": ["sh", "-c", f"i=0; while [ $i -lt {iters} ]; do i=$((i+1)); done"],
                "resources": {
                    # REQUESTS ONLY, deliberately.
                    #  * The observation bridge sums requests, exactly as
                    #    cloud_env sums t.cpu over running tasks
                    #    (cloud_env.py:77-84) -- declared demand, not usage.
                    #  * No CPU limit: a limit would pin each pod to its own
                    #    quota and destroy the node-level contention that stands
                    #    in for `speed`. Consequence: QoS class is Burstable and
                    #    a pod CAN use more CPU than it requested. Recorded in
                    #    MAPPING.md O-1.
                    "requests": {"cpu": f"{cpu_m}m", "memory": f"{mem_mi}Mi"},
                },
            }],
        },
    }

    if cfg.set_active_deadline:
        # OFF BY DEFAULT, and it is not the simulator's deadline.
        # activeDeadlineSeconds counts from the moment the pod STARTS, so a
        # deferred pod silently gets a fresh clock, whereas cloud_env expires a
        # task at arrival_t + slack regardless of whether it ever started
        # (cloud_env.py:281-300). Enabling this makes Kubernetes kill pods on a
        # different rule than the one being measured.
        pod["spec"]["activeDeadlineSeconds"] = int(round(row.slack * cfg.step_seconds))
        annotations[f"{ANNOT_PREFIX}/active-deadline-warning"] = (
            "activeDeadlineSeconds counts from pod start, NOT from arrival; "
            "it is not equivalent to cloud_env's deadline_t"
        )

    return pod


def rows_to_pods(rows: list[TraceRow], cfg: ScaleConfig) -> list[dict]:
    """N trace rows -> N Pod manifests, in trace order."""
    agents = assign_initial_agents(rows, cfg.n_agents, cfg.route_seed)
    return [row_to_pod(r, i, cfg, agents[i]) for i, r in enumerate(rows)]


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def to_yaml(pods: list[dict]) -> str:
    try:
        import yaml
    except ImportError:  # pragma: no cover
        raise SystemExit("pyyaml is required for YAML output; use --format json instead")
    return "".join("---\n" + yaml.safe_dump(p, sort_keys=False) for p in pods)


def to_json(pods: list[dict]) -> str:
    return json.dumps({"apiVersion": "v1", "kind": "List", "items": pods}, indent=2)


def scaling_report(rows: list[TraceRow], pods: list[dict], cfg: ScaleConfig) -> str:
    """Human-readable summary, printed to stderr so it never pollutes manifests."""
    if not rows:
        return "no rows selected"
    cpu_m = [int(p["spec"]["containers"][0]["resources"]["requests"]["cpu"][:-1]) for p in pods]
    mem_mi = [int(p["spec"]["containers"][0]["resources"]["requests"]["memory"][:-2]) for p in pods]
    steps = [r.arrival_step for r in rows]
    lines = [
        f"rows selected            : {len(rows)}",
        f"manifests emitted        : {len(pods)}",
        f"arrival_step range       : {min(steps)} .. {max(steps)}",
        f"cpu request range        : {min(cpu_m)}m .. {max(cpu_m)}m  "
        f"(1.0 sim unit = {cfg.cpu_unit_millis}m)",
        f"memory request range     : {min(mem_mi)}Mi .. {max(mem_mi)}Mi  "
        f"(1.0 sim unit = {cfg.mem_unit_mi}Mi)",
        f"total cpu requested      : {sum(cpu_m)}m over the whole selection",
        f"total memory requested   : {sum(mem_mi)}Mi over the whole selection",
        f"scheduler                : {cfg.scheduler_name}",
        f"work->iters calibration  : {'CALIBRATED' if cfg.calibrated else 'UNCALIBRATED [UNVERIFIED]'}",
    ]
    # Capacity sanity check against the cluster definition's aim.
    fleet_cpu_m = int(sum(c * cfg.cpu_unit_millis for c in (1.0, 1.5, 2.0)))
    peak = {}
    for r in rows:
        peak[r.arrival_step] = peak.get(r.arrival_step, 0) + cfg.cpu_millis(r.cpu)
    worst_step, worst = max(peak.items(), key=lambda kv: kv[1])
    lines += [
        f"fleet cpu allocatable    : {fleet_cpu_m}m (aim of cluster/k3d-cluster.yaml)",
        f"busiest single step      : step {worst_step}, {worst}m arriving at once",
    ]
    if worst > fleet_cpu_m:
        lines.append(
            "  NOTE: a single step can request more CPU than the whole fleet can "
            "hold. In the simulator that just inflates the backlog; in Kubernetes "
            "the binding is REFUSED (MAPPING.md A-1)."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert the P3 Alibaba trace into Kubernetes Pod manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Nothing in this script contacts a cluster.",
    )
    p.add_argument("--trace", default=str(DEFAULT_TRACE),
                   help="path to trace_alibaba.csv (default: %(default)s)")
    p.add_argument("--max-step", type=int, default=None,
                   help="keep only rows with arrival_step <= MAX_STEP (episode window)")
    p.add_argument("--limit", type=int, default=None, help="cap the number of rows")
    p.add_argument("--out", default=None,
                   help="write manifests here (default: stdout)")
    p.add_argument("--format", choices=("yaml", "json"), default="yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="print manifests + the scaling report and exit without "
                        "writing any file; implies stdout output")
    p.add_argument("--cpu-unit-millis", type=int, default=ScaleConfig.cpu_unit_millis)
    p.add_argument("--mem-unit-mi", type=int, default=ScaleConfig.mem_unit_mi)
    p.add_argument("--step-seconds", type=float, default=ScaleConfig.step_seconds)
    p.add_argument("--iters-per-work-unit", type=int, default=None,
                   help="calibration constant; if omitted, manifests are stamped "
                        "UNCALIBRATED (see --help-calibration)")
    p.add_argument("--image", default=ScaleConfig.image)
    p.add_argument("--namespace", default=ScaleConfig.namespace)
    p.add_argument("--scheduler-name", default=ScaleConfig.scheduler_name,
                   help="use 'default-scheduler' for the kube-scheduler baseline run")
    p.add_argument("--run-id", default=ScaleConfig.run_id)
    p.add_argument("--agents", type=int, default=ScaleConfig.n_agents,
                   help="N, must match cloud_env's n_servers")
    p.add_argument("--route-seed", type=int, default=ScaleConfig.route_seed,
                   help="seed for the SYNTHESIZED initial-agent routing")
    p.add_argument("--set-active-deadline", action="store_true",
                   help="also set spec.activeDeadlineSeconds. NOT equivalent to "
                        "the simulator's deadline; read row O-8 of MAPPING.md first")
    p.add_argument("--help-calibration", action="store_true",
                   help="print the work-unit calibration procedure and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.help_calibration:
        print(CALIBRATION_HELP)
        return 0

    cfg = ScaleConfig(
        cpu_unit_millis=args.cpu_unit_millis,
        mem_unit_mi=args.mem_unit_mi,
        step_seconds=args.step_seconds,
        iters_per_work_unit=(args.iters_per_work_unit
                            if args.iters_per_work_unit is not None
                            else DEFAULT_ITERS_PER_WORK_UNIT),
        calibrated=args.iters_per_work_unit is not None,
        image=args.image,
        namespace=args.namespace,
        scheduler_name=args.scheduler_name,
        run_id=args.run_id,
        n_agents=args.agents,
        route_seed=args.route_seed,
        set_active_deadline=args.set_active_deadline,
    )

    rows = load_trace(args.trace, max_step=args.max_step, limit=args.limit)
    pods = rows_to_pods(rows, cfg)
    text = to_yaml(pods) if args.format == "yaml" else to_json(pods)

    if args.dry_run or not args.out:
        sys.stdout.write(text)
        print("\n# ---- scaling report ----", file=sys.stderr)
        print(scaling_report(rows, pods, cfg), file=sys.stderr)
        if not cfg.calibrated:
            print("# WARNING: UNCALIBRATED work->iters constant. Manifests are "
                  "usable for dry runs only.", file=sys.stderr)
        return 0

    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {len(pods)} manifests to {args.out}", file=sys.stderr)
    print(scaling_report(rows, pods, cfg), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
