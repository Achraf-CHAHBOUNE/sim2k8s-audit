# sim2k8s-audit — code for the AICSET'26 paper

Artifact for *What the Simulator Sees and Kubernetes Does Not: An Observability
Audit for Transferring Multi-Agent Reinforcement Learning Schedulers*
(AICSET'26, Paper ID 190).

It contains the multi-agent admission-control simulator, the observation/action
bridge to Kubernetes, the replay controller, the 122 offline tests, the script
that measures unreachable simulator states, and the Kubernetes documentation
snapshots behind every Kubernetes statement in the paper. No cluster is needed
for anything in this repository.

## Layout

| Path | Content |
|---|---|
| `p3/code/` | Simulator (`cloud_env.py`), workload trace loader (`trace.py`), heuristic policies, MAPPO actor/critic and training code; `trace_alibaba.csv` (sha256 `aab04247…e952`) |
| `p3/_results_backup_vps/n3_trace/config.json` | Run configuration read by one test in `p4/policies/` |
| `p4/bridge/` | Observation mapping (`observation.py`) and action mapping (`action.py`); `MAPPING.md` is the component-by-component audit table |
| `p4/controller/` | Replay controller (`replay.py`), simulated and modelled Kubernetes backends (`backends.py`) |
| `p4/replay/` | Trace-to-pod conversion |
| `p4/energy/` | Host energy logging (RAPL) |
| `p4/policies/` | Loading saved actor weights |
| `aicset/measure_unreachable.py` | Rollouts on the trace (N = 3, horizon 150, seeds 30000–30049); writes `measure_unreachable.json` |
| `aicset/k8s_docs/` | Plain-text snapshots of the cited kubernetes.io pages, each with its source URL and retrieval date; `fetch.py` re-downloads them |

## Run

Tested with Python 3.13.11, NumPy 2.4.6, pytest 9.0.3, PyTorch 2.12.1 (CPU).

```
pip install -r requirements.txt
python -m pytest p4                      # 122 tests: 38 bridge, 41 controller,
                                         # 25 replay, 13 energy, 5 policies
python aicset/measure_unreachable.py     # regenerates aicset/measure_unreachable.json
```

`measure_unreachable.py` runs in a few seconds; its output reproduces the
published `measure_unreachable.json` byte for byte.

## Scope

This repository holds the offline study only. The live-cluster paths of the
replay controller (`--backend live`) need a Kubernetes cluster and the
`kubernetes` Python client, and are not exercised by the tests.
