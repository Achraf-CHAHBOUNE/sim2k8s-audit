"""How often does the simulator visit states that cannot exist on Kubernetes?

Read-only use of p3/code. Settings match P3's trace runs: N=3, horizon=150,
trace_load=1 (run_experiment.py defaults), p3/code/trace_alibaba.csv.

Policies:
  random  -- uniform over {accept, defer, migrate}, as at the start of training
  accept  -- always accept (the most aggressive admission policy)

Per step, a state is "unreachable on Kubernetes" if any server has
max(cpu_util, mem_util) > 1: the scheduler does not over-subscribe allocatable.
Also reported: share of agent observations with slot 0 or 1 above 0.5 (the half
of the trained input range that cannot occur), share of steps with c1 > 0, and,
from the trace alone, share of populated steps with more arrivals than the N=3
agents can decide in one step.

Usage: python conference_aicset/measure_unreachable.py [episodes]
"""
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "p3" / "code"))
from cloud_env import CloudAllocationEnv  # noqa: E402
from trace import WorkloadTrace  # noqa: E402

TRACE = ROOT / "p3" / "code" / "trace_alibaba.csv"
N, HORIZON = 3, 150
EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SEED0 = 30000


def run(policy: str) -> dict:
    steps = over = c1_pos = 0
    obs_total = obs_upper = 0
    peak = 0.0
    for ep in range(EPISODES):
        env = CloudAllocationEnv(n_servers=N, horizon=HORIZON,
                                 trace=WorkloadTrace(str(TRACE), horizon=HORIZON),
                                 seed=SEED0 + ep)
        rng = np.random.default_rng(SEED0 + ep)
        obs = env._build_obs()
        while True:
            obs_total += obs.shape[0]
            obs_upper += int(np.sum((obs[:, 0] > 0.5) | (obs[:, 1] > 0.5)))
            if policy == "random":
                a = rng.integers(0, 3, size=N)
            else:
                a = np.zeros(N, dtype=int)
            obs, _, term, trunc, info = env.step(a)
            u = max(max(s.cpu_util(), s.mem_util()) for s in env.servers)
            peak = max(peak, u)
            steps += 1
            over += int(u > 1.0)
            c1_pos += int(info["cost"][0] > 0)
            if term or trunc:
                break
    return {"policy": policy, "episodes": EPISODES, "steps": steps,
            "share_steps_overcommitted": over / steps,
            "share_steps_c1_positive": c1_pos / steps,
            "share_obs_in_upper_half": obs_upper / obs_total,
            "peak_utilization": peak}


def trace_burst() -> dict:
    c = collections.Counter()
    with open(TRACE, newline="") as f:
        for r in csv.DictReader(f):
            c[int(float(r["arrival_step"]))] += 1
    pop = len(c)
    return {"populated_steps": pop,
            "share_populated_steps_above_N": sum(v > N for v in c.values()) / pop,
            "max_arrivals_per_step": max(c.values())}


if __name__ == "__main__":
    out = {"settings": {"N": N, "horizon": HORIZON, "trace_load": 1,
                        "seeds": [SEED0, SEED0 + EPISODES - 1]},
           "trace": trace_burst(),
           "runs": [run("random"), run("accept")]}
    print(json.dumps(out, indent=2))
    (Path(__file__).parent / "measure_unreachable.json").write_text(json.dumps(out, indent=2))
