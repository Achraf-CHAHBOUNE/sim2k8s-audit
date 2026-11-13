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
| `simulator/` | Simulator (`cloud_env.py`), workload trace loader (`trace.py`), heuristic policies, MAPPO actor/critic and training code; `trace_alibaba.csv` (sha256 `aab04247…e952`) |
| `simulator/n3_trace_config.json` | Run configuration read by one test in `k8s/policies/` |
| `k8s/bridge/` | Observation mapping (`observation.py`) and action mapping (`action.py`); `MAPPING.md` is the component-by-component audit table |
| `k8s/controller/` | Replay controller (`replay.py`), simulated and modelled Kubernetes backends (`backends.py`) |
| `k8s/replay/` | Trace-to-pod conversion |
| `k8s/energy/` | Host energy logging (RAPL) |
| `k8s/policies/` | Loading saved actor weights |
| `aicset/measure_unreachable.py` | Rollouts on the trace (N = 3, horizon 150, seeds 30000–30049); writes `measure_unreachable.json` |
| `aicset/k8s_docs/` | Plain-text snapshots of the cited kubernetes.io pages, each with its source URL and retrieval date; `fetch.py` re-downloads them |

## Naming

Code comments refer to the simulator study as P3 and to the Kubernetes study
as P4: `simulator/` holds the P3 simulator, `k8s/` the P4 code.

## Run

Tested with Python 3.13.11, NumPy 2.4.6, pytest 9.0.3, PyTorch 2.12.1 (CPU).

```
pip install -r requirements.txt
python -m pytest k8s                     # 122 tests: 38 bridge, 41 controller,
                                         # 25 replay, 13 energy, 5 policies
python aicset/measure_unreachable.py     # regenerates aicset/measure_unreachable.json
```

`measure_unreachable.py` runs in a few seconds and reproduces the values in the
published `measure_unreachable.json`.

## Scope

This repository holds the offline study only. The live-cluster paths of the
replay controller (`--backend live`) need a Kubernetes cluster and the
`kubernetes` Python client, and are not exercised by the tests.
