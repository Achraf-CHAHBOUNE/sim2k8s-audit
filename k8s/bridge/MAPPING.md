# P4 — simulator ↔ Kubernetes mapping

**What this document is.** `simulator/cloud_env.py` defines a 9-dimensional
per-agent observation and a 3-action space. This file records, component by
component, what each one means in the simulator, where it comes from in
Kubernetes, and whether it is

| Category | Meaning |
|---|---|
| **(a) directly observable** | Kubernetes exposes the same quantity, with the same meaning. |
| **(b) approximable** | A Kubernetes quantity exists and can stand in, but its meaning differs in a way that must be stated. |
| **(c) NOT observable / NOT expressible** | There is no Kubernetes counterpart. The value is synthesized by the harness, reproduced by formula, or the behaviour has to be implemented by our own controller. |

**Category (c) is the result of P4, not a defect of it.** The transfer claim the
thesis is allowed to make is bounded by this list. Everything in (c) is a place
where "we deployed the policy on a real cluster" would be an overstatement unless
the qualification travels with it.

Written 2026-09-27, updated 2026-09-28. Sources: `simulator/cloud_env.py` (line
numbers below are from that file), `simulator/trace.py`, `simulator/make_trace.py`,
and the measured properties of `simulator/trace_alibaba.csv`. Nothing here was
measured on a cluster — **no cluster has been created** (`which k3d` → not found,
2026-09-27). Every claim about Kubernetes behaviour is from the API contract, and
each one that a run could contradict is marked **[TO VERIFY ON FIRST RUN]**.

---

## 0. Scoreboard

| | Count | Components |
|---|---|---|
| (a) directly observable | 2 / 9 | O-5 `task_cpu`, O-6 `task_mem` |
| (b) approximable | 5 / 9 | O-1 `cpu_util`, O-2 `mem_util`, O-3 `queue_backlog`, O-7 `task_proc`, O-8 `task_deadline_slack` |
| (c) not observable | 2 / 9 | O-4 `energy`, O-9 `has_task` |

| | Count | Actions |
|---|---|---|
| (a) maps cleanly | 2 / 3 | A-1 `accept`, A-2 `defer` |
| (b) maps with changed cost/semantics | 1 / 3 | A-3 `migrate` |

Plus **10 cross-cutting mismatches** (G-1 … G-10) that belong to no single slot
and that are, collectively, a larger threat to the transfer claim than any
individual observation component. G-3, G-4, G-6 and G-9 are hard (c).

The honest one-line summary: **0 of the 9 observation components are read from
Kubernetes in an unqualified way.** The two (a) components are observable only
because the harness authored the pod that carries them.

---

## 1. Observation vector

`cloud_env.py:349-370` builds, per agent *i*:

```
obs[i] = [ cpu_util, mem_util, queue_backlog, energy,
           task_cpu, task_mem, task_proc, task_deadline_slack, has_task ]
```

Row ids O-1…O-9 correspond to slots 0…8. The machine-readable copy of this table
is `OBS_COMPONENTS` in `observation.py`; `test_bridge.py` fails if the two drift.

### O-1 `cpu_util` — (b) approximable

| | |
|---|---|
| **Simulator** | `sum(t.cpu for t in srv.running) / srv.cpu_cap`, clamped at 2.0 then halved (`cloud_env.py:77-78, 364`). A sum of **declared demands** of accepted tasks, not of measured usage. |
| **Kubernetes** | Σ container CPU **requests** of non-terminal pods bound to the node ÷ `node.status.allocatable.cpu`. |
| **Why (b)** | Three distortions, all one-directional. |

1. **The overcommit regime is unreachable.** The simulator lets `cpu_util` reach
   2.0 because `ACCEPT` always succeeds (`cloud_env.py:200-202`) — nothing checks
   capacity. Kubernetes refuses a binding whose requests exceed allocatable. So
   slot 0 is confined to [0, 1] on the cluster while training data covered [0, 2].
   **Half the observation range the policy was trained on cannot occur.** This is
   the single most consequential row in this document; see also G-9.
2. **Requests ≠ usage, and no CPU limit is set.** `trace_to_pods.py` deliberately
   emits requests without limits (so that node-level CPU contention can stand in
   for `speed`, G-6). A pod can therefore burn more CPU than it requested. Slot 0
   tracks the *booking*, not the machine. This matches the simulator's semantics,
   which is why it is the right choice — but it means slot 0 is not a measurement
   of the node's real load, and must never be described as one.
3. **Foreign pods consume allocatable.** kube-system and DaemonSet pods hold
   requests the simulator has no notion of. `NodeState.foreign_*_request_*`
   subtracts them so that "no replayed task" reads as util 0. The subtraction is a
   modelling choice, and it makes the effective capacity depend on what k3s
   happens to be running. **[TO VERIFY ON FIRST RUN]** how much allocatable the
   k3s system pods actually hold on each agent.

### O-2 `mem_util` — (b) approximable

| | |
|---|---|
| **Simulator** | `sum(t.mem for t in srv.running) / srv.mem_cap`, same clamp (`cloud_env.py:80-81, 365`). |
| **Kubernetes** | Σ container memory requests of non-terminal bound pods ÷ `node.status.allocatable.memory`. |
| **Why (b)** | Everything in O-1, and one thing worse. |

Memory is **incompressible**. In the simulator, exceeding `mem_cap` merely raises
the capacity cost `c_1` and degrades smoothly. On a real node the consequence is
an OOMKill: the pod dies, which the simulator models nowhere. The failure modes
are not the same shape, so a policy that learned "memory pressure is a gradual
cost" is learning something false about Kubernetes.

### O-3 `queue_backlog` — (b) approximable

| | |
|---|---|
| **Simulator** | `min(len(srv.running), queue_cap) / queue_cap` (`cloud_env.py:83-84, 366`). Count of tasks accepted onto this server and not yet finished. |
| **Kubernetes** | Count of pods bound to the node in `Pending`/`Running` carrying our run label. |
| **Why (b)** | The integer is the same; the dynamics behind it are not. |

In the simulator, all tasks in `running` hold their CPU/memory, but **only the
FIFO head is served** (`cloud_env.py:302-320`) — a backlog of 10 means 9 tasks are
frozen. In Kubernetes all 10 bound pods execute concurrently, sharing CPU under
CFS in proportion to their requests. Consequences:

- A given backlog means *much* more progress in Kubernetes than in the simulator.
- The simulator's tight coupling "backlog ⇒ latency" (it literally defines the
  latency objective as the in-system count, `cloud_env.py:228-229`) is weaker on
  the cluster. Expect measured latency to beat simulator-predicted latency for
  structural reasons that have nothing to do with the policy. **This is a prime
  candidate for the "transfer gap" number** PLAN_P4 §6 asks for, and it must not
  be reported as the policy performing better than predicted.
- `queue_cap = 15` (`cloud_env.py:115`) is a simulator constant. Kubernetes' only
  analogue is kubelet `max-pods` (default 110). `k3d-cluster.yaml` sets
  `max-pods=15` so the normalisation denominator means something — but that makes
  the node **refuse** binding 16, which the simulator never does (A-1, G-9).

### O-4 `energy` — **(c) NOT observable**

| | |
|---|---|
| **Simulator** | `(p_idle + p_dyn·min(cpu_util,2)) / (p_idle + p_dyn·2)` with `p_idle=0.1`, `p_dyn=0.9` (`cloud_env.py:130-131, 360-362`). |
| **Kubernetes** | **Nothing.** There is no per-node, per-pod or per-container energy counter in any Kubernetes API. |

Three separate reasons this is (c), any one of which is sufficient:

1. **It is not a measurement even in the simulator.** It is a closed-form function
   of slot 0 — algebraically redundant, carrying zero extra information. The
   policy's "energy input" is a rescaled copy of its CPU-utilisation input.
2. **Per-node energy is unmeasurable in this topology.** The three agents are
   containers on **one** physical CPU package (i7-13620H). Even with full RAPL
   access there is no per-cgroup energy counter; package energy cannot be
   attributed to a node.
3. **Package energy is not readable on this host at all without installing a
   kernel-mode driver.** Verified 2026-09-27 — see `../energy/PROTOCOL.md` for the
   probe results.

**Bridge decision:** `observation.py` reproduces the slot **by formula**, with the
simulator's own constants. Rationale: the policy was trained on that function, so
substituting a real measurement (even if one existed) would push the input
off-distribution and confound the transfer result with an input-distribution
change. Measured or proxied energy is reported as an **outcome metric**, never fed
into the observation. `ObservationReport.formula_slots` records this at every step.

> **Claim boundary.** P4 may say: "the energy *objective* is a model, identical in
> simulation and on the cluster; the energy *outcome* is reported as a host-level
> proxy." P4 may **not** say it measured energy on Kubernetes, and may not report
> a "measured vs predicted energy transfer gap" — both sides of that comparison
> would be the same formula.

### O-5 `task_cpu` — (a) directly observable

| | |
|---|---|
| **Simulator** | `arrivals[0].cpu`, the CPU demand of the head undecided arrival (`cloud_env.py:355, 367`). |
| **Kubernetes** | CPU request of the head pending pod assigned to this agent, ÷ `cpu_unit_millis`. |

Exact, both directions, with one honesty note: it is observable **because the
harness authored the pod from the trace**. It is a value we injected and then read
back, not a property of the cluster we discovered. The same is true of O-6.

### O-6 `task_mem` — (a) directly observable

As O-5, with the memory request and `mem_unit_bytes`. `UnitScale.assert_consistent_with`
guards against the generator and the bridge disagreeing on the unit scale, which
would mis-scale slots 4 and 5 silently.

### O-7 `task_proc` — (b) approximable

| | |
|---|---|
| **Simulator** | `arrivals[0].remaining / 5.0` (`cloud_env.py:356`, `_max_work = task_work_range[1] = 5.0`). |
| **Kubernetes** | `pltr-ma.io/work-units` annotation ÷ 5.0. |
| **Why (b)** | Exact for the slot; a fiction as a general capability. |

- For the slot, it is **exact**: `_build_obs` reads `arrivals[0]`, which has never
  been processed, so `remaining == work` always. The annotation reproduces it.
- But the value is known only because the replay **prescribed** it. A container's
  remaining work is not observable in Kubernetes — a container is a black box that
  exits when it exits. `PodState.remaining_work_units` therefore defaults to
  `None` ("unknown"), and is only ever non-`None` if a future instrumented
  workload self-reports progress.
- The conversion "work units → a fixed instruction count" needs a calibration
  constant that is **[UNVERIFIED]**: nothing has been measured on this or any
  host. Manifests generated with the default are stamped
  `pltr-ma.io/calibration: UNCALIBRATED`, and `trace_to_pods.py --help-calibration`
  holds the procedure.

### O-8 `task_deadline_slack` — (b) approximable

| | |
|---|---|
| **Simulator** | `max(0, arrivals[0].deadline_t - t) / 30.0` (`cloud_env.py:357`, `_max_slack = 30`). `deadline_t = arrival_t + int(slack)` (`cloud_env.py:260`). |
| **Kubernetes** | `(pltr-ma.io/deadline-step − current replay step) / 30.0`. |
| **Why (b)** | The number is available; the *semantics* are absent. |

**Kubernetes has no deadline concept.** Nothing in the control plane will act on
this annotation. Specifically:

- The simulator drops any task, queued or running, once `t > deadline_t`
  (`_expire_overdue`, `cloud_env.py:281-300`), and counts it as an SLA violation.
  On the cluster, **our replay controller must delete those pods itself.** The SLA
  metric therefore measures a rule we implement, not a cluster behaviour (G-7).
- `spec.activeDeadlineSeconds` is **not** equivalent and is off by default: it
  clocks from pod *start*, so a deferred pod silently receives a fresh deadline,
  whereas the simulator's deadline runs from *arrival* regardless of whether the
  task ever started. `trace_to_pods.py --set-active-deadline` exists but stamps a
  warning annotation.
- Slack is expressed in *steps*; converting to seconds requires the step↔wall-clock
  binding of G-2, which is a free parameter, not a measured quantity.

### O-9 `has_task` — **(c) NOT observable**

| | |
|---|---|
| **Simulator** | `1.0` if `srv.arrivals` is non-empty (`cloud_env.py:353, 368`). |
| **Kubernetes** | **Nothing.** There is no per-node arrival buffer. |

An unbound pod belongs to **no node** — that is the defining property of a pending
pod. The simulator, by contrast, pre-routes every arrival to a uniformly random
server *before* any agent decides (`cloud_env.py:261,
`self.servers[int(self.rng.integers(self.N))].arrivals.append(task)`), and each
agent then sees only its own buffer.

So the simulator's per-agent arrival buffers are an **artefact of the simulator's
own construction**. The harness re-injects them: `trace_to_pods.py` synthesizes the
routing with the same uniform draw and records it in
`pltr-ma.io/initial-agent`, stamped
`pltr-ma.io/initial-agent-origin: SYNTHESIZED-from-cloud_env.py:261`.
`ObservationReport.synthesized_slots` flags slot 8 at every step.

Consequences to state in the paper:

- The decomposition of one global pending queue into N private queues is imposed
  by us. A real Kubernetes scheduler sees all pending pods and picks; our agents
  cannot see a pod routed to a sibling. **The multi-agent structure is
  partly artificial**, and `MIGRATE` exists mainly to undo bad routing that a real
  scheduler would never have committed to (A-3).
- FIFO order within a buffer is also ours, and it is **not recoverable from
  observable data**. The simulator's buffer is a plain insertion-ordered list, and
  `MIGRATE` *appends* to the destination (`cloud_env.py:208`), so after any
  migration the buffer order differs from arrival-time order. Sorting pending pods
  by arrival step therefore picks the **wrong head task** — this was caught by the
  equivalence test `test_matches_cloud_env_build_obs`, which failed on slots 4-7
  until an explicit `PodState.enqueue_seq` counter was added. Kubernetes holds no
  such counter: the replay controller must maintain it and bump it to the tail on
  every retarget. One more piece of simulator state with no cluster home.

---

## 2. Action space

`cloud_env.py:194-209`. Each agent acts on **the head of its own arrival buffer
only**, once per step. `0 = accept`, `1 = defer`, `2 = migrate`; there is
deliberately **no reject** (`cloud_env.py:26-28`).

### A-1 `ACCEPT` — (a) maps cleanly

| | |
|---|---|
| **Simulator** | `srv.arrivals.pop(0); srv.running.append(task)` (`cloud_env.py:200-202`). Always succeeds; capacity is never checked. |
| **Kubernetes** | `POST /api/v1/namespaces/{ns}/pods/{name}/binding` — a `Binding` object naming the target node. Exactly what a scheduler does; `action.py` uses `create_namespaced_pod_binding`. |

This is the cleanest correspondence in the project. Three divergences to record:

1. **The cluster can refuse.** Binding fails if the node no longer fits the pod,
   and admission can fail *after* a successful binding (kubelet rejects it, pod →
   `Failed`). The simulator has no notion of a refused action. `action.py` returns
   an `OpResult` with `refused=True` and the reason; the replay controller must log
   every refusal, because a refusal is the cluster overriding the policy and any
   run with refusals is not a clean measurement of the policy. **[TO VERIFY ON
   FIRST RUN]** what fraction of ACCEPTs are refused at `max-pods=15`.
2. **Binding is irreversible.** There is no unbind. The simulator's state
   transition is reversible in principle (MIGRATE moves queued tasks freely);
   binding is a one-way door.
3. **Binding ≠ starting.** After binding, the kubelet must pull/start the
   container; the pod is `Pending` for a while yet. The simulator's accepted task
   begins accruing service at the next step. Start latency is real and unmodelled
   (G-2).

### A-2 `DEFER` — (a) maps cleanly

| | |
|---|---|
| **Simulator** | no-op; the task stays at the **head** of the buffer and is re-decided next step (`cloud_env.py:203-204`). |
| **Kubernetes** | Do nothing. The pod stays `Pending`. |

Deferral is the natural state of a pod a scheduler has not bound, so this is
faithful. Two conditions:

1. The pod must carry `schedulerName: pltr-ma-scheduler` (as `trace_to_pods.py`
   emits) or **the default kube-scheduler will bind it out from under the
   policy** — and a "deferred" pod would start running anyway. This is the single
   easiest way to invalidate the whole experiment, and the baseline run is exactly
   the same manifests with `--scheduler-name default-scheduler`.
2. Head-of-line order across steps is maintained by us (O-9), not by Kubernetes.

Note the *cost* asymmetry: in the simulator, deferring keeps the task in the
in-system count, so it is directly penalised through the latency objective
(`cloud_env.py:228-229`). On the cluster, a pending pod also costs nothing real —
it holds no resources — so the penalty is again a model, consistent with the
simulator by construction rather than by measurement.

### A-3 `MIGRATE` — (b) maps, but the cost model does not

| | |
|---|---|
| **Simulator** | `srv.arrivals.pop(0)`; append to `servers[_least_loaded_other(i)].arrivals` (`cloud_env.py:205-209`). Charges `migrate_energy = 0.05` (`cloud_env.py:132, 327`). |
| **Kubernetes** | **Re-target a pending pod** — update our own routing bookkeeping (and the `pltr-ma.io/initial-agent` annotation for auditability). No cluster operation is required. |

**This row corrects the project plan.** `PLAN_P4_KUBERNETES.md` §2 and the P4 task
brief both describe `migrate` as "eviction + rebinding". It is not:
`cloud_env.MIGRATE` operates on `arrivals[0]` — a task that has **never been
accepted** and is **not running**. The simulator never migrates a running task.
Therefore:

- The faithful Kubernetes mapping is a **re-route of an unbound pod**, which costs
  nothing: no state moves, no container restarts, no memory is copied.
- Implementing eviction + rebinding would model a behaviour **the simulator does
  not have**, and the trained policy has never taken such an action. `action.py`
  contains `EvictOp` but it is **disabled by default** and raises unless
  `allow_eviction=True` is passed explicitly, with the reason recorded.
- Live migration of a *running* pod is in any case **(c) not expressible**: a pod
  is immutable in its node assignment; "migration" in Kubernetes is delete +
  recreate, and the replacement pod **restarts the work from zero**. The
  simulator's migrated task keeps its `remaining` work and its `arrival_t`
  (`cloud_env.py:208` moves the object itself). Container checkpoint/restore
  (CRIU) is not available in k3s. So the simulator's migration semantics —
  preserve progress, preserve arrival time, pay 0.05 energy — have **no
  Kubernetes realisation** for a running task.

Two further divergences:

- **The destination is not a policy choice.** `_least_loaded_other` picks the
  minimum-backlog node by count, breaking ties by index
  (`cloud_env.py:277-279`), where "load" ignores CPU, memory and task size. A real
  scheduler would score nodes. The bridge reproduces the simulator's rule exactly,
  so that a migration decision means the same thing — at the price of encoding a
  scheduling heuristic far weaker than kube-scheduler's, which flatters the
  baseline.
- **The cost is charged for something free.** The simulator pays 0.05 energy per
  migration; the real operation is a bookkeeping update. The simulator therefore
  *over*-penalises migration, and a policy trained in it should be expected to
  under-migrate on the cluster relative to optimal. That is a **falsifiable,
  pre-registered prediction** and a good answer to PLAN_P4 §6's "include a failure
  mode": it was written down before any run.

### A-4 the missing `reject` — (a) consistent, worth stating

The simulator has no voluntary drop (`cloud_env.py:26-28`), so a task leaves only
by completing or expiring. Kubernetes likewise gives a custom scheduler no
"refuse forever" primitive — leaving a pod pending indefinitely *is* deferral.
The absence transfers cleanly. Worth one sentence in the paper because it removes
the degenerate policy that the P3 design notes already flag.

---

## 3. Cross-cutting mismatches

These belong to no single slot. Several are more damaging to the transfer claim
than any individual observation row.

### G-1 Atomicity and staleness of the observation — (c)

The simulator's `_build_obs` reads a consistent, instantaneous global state. A
Kubernetes observation is assembled from **separate API calls** and from status
that lags reality: kubelet posts node status periodically, and a pod's phase
change reaches the API server after the fact. There is no atomic cluster read.
`ClusterSnapshot.max_info_age_s` records the age of the oldest datum and
`build_observation` emits a `STALE` warning past a budget. **The policy acts on a
past state, which is not the MDP it was trained on.** **[TO VERIFY ON FIRST RUN]**
the actual observation age distribution.

### G-2 Step ↔ wall clock — (c), a free parameter

A simulator step is dimensionless. `speed` is work units per *step*; arrivals are
indexed by `arrival_step`; deadlines are in steps. Mapping a step to seconds
(`--step-seconds`, default 1.0) is a **choice with no correct value**, and it sets
the difficulty of the whole experiment: too short and every pod misses its
deadline during container startup; too long and the cluster is idle. Also:
`PLAN_P4_KUBERNETES.md` risk 3 already records that a real replay costs real
minutes — at 1 s/step a 200-step episode (`horizon=200`, `cloud_env.py:105`) is
~3.3 min plus startup, so seed counts will be far below P3's 10. **Plan and report
the seed count in advance.**

### G-3 Server heterogeneity — cpu/mem (b), `speed` (c)

`cloud_env.py:124-126` builds `cpu_cap = mem_cap = [1.0, 1.5, 2.0]` and
`speed = [1.0, 1.25, 1.5]`.

- **cpu_cap / mem_cap: (b).** Reproduced by shrinking *allocatable* with kubelet
  `system-reserved` (`k3d-cluster.yaml`). But `system-reserved` is **accounting
  only** — it changes what the scheduler thinks the node has, not what the node
  can compute. An agent "full" at 1000m still has 16 real threads behind it. So
  capacity heterogeneity is real to the scheduler and fictional to the CPU. The
  optional `docker update --cpus` step in `PREREQUISITES.md` is the only way to
  make it physical, and it has **not been run**.
- **`speed`: (c).** There is no Kubernetes setting for "this node drains work
  faster". The design hope is that it *emerges*: CPU-bound pods with a fixed
  instruction count and no CPU limit finish faster on a node with more real CPU.
  That requires the physical limits above, and is **[UNVERIFIED]** — no
  calibration has been run (O-7).

### G-4 Per-node energy — (c)

Three containers, one CPU package. No per-container energy counter exists at any
level of the stack. Nothing that can be measured on this host attributes energy to
a node. See `../energy/PROTOCOL.md`.

### G-5 `queue_cap = 15` — (c) a simulator constant

Used as the normalisation denominator of slot 2 and as the backlog ceiling in cost
`c_2` (`cloud_env.py:336-340`). Kubernetes' `max-pods` is the nearest analogue and
behaves differently: it is a hard admission limit, not a soft cost. Setting
`max-pods=15` makes the denominator meaningful and the overflow regime
*unreachable* — the same trade-off as O-1/G-9.

### G-6 Service discipline — (c)

The simulator serves **one FIFO head task per server per step, at rate `speed`**,
and explicitly discards leftover budget when the head does not finish
(`cloud_env.py:319`). Kubernetes runs every bound pod concurrently under CFS with
no notion of a head. There is no configuration that makes a node serialise its
pods. This is not approximable: it is a different queueing system. It is the root
cause of the O-3 latency divergence and arguably **the largest single source of
transfer gap**.

### G-7 Task expiry and the SLA metric — (b) by our own controller

`_expire_overdue` (`cloud_env.py:281-300`) deletes overdue tasks and increments
`sla_violations`; those counts feed reward component 3 and cost `c_3`. Kubernetes
does none of this. Our controller must delete overdue pods on the simulator's
rule. The SLA number is therefore **produced by our code on both sides** of the
comparison — it is consistent, but it is not a cluster-native measurement, and the
kube-scheduler baseline must be subjected to the identical expiry rule or the
comparison is void.

### G-8 Decision rate — (c) an artefact

Each agent decides **one** task per step, so N=3 gives at most 3 admissions per
step. The trace delivers up to **10 arrivals in a single step** (measured on
`trace_alibaba.csv`, 2026-09-27: 6988 rows, 3426 populated steps, max 10/step,
mean 1.40/step over steps 0-4999). Real schedulers bind many pods per second, so
the per-step decision cap is a simulator artefact that manufactures backlog. Keep
it — the policy was trained under it — but do not present the resulting backlog as
a property of Kubernetes.

### G-9 Constraint `c_1` is structurally unreachable — (c)

`c_1 = mean over servers of max(0, max(cpu_util, mem_util) − 1)`
(`cloud_env.py:333-335`). With honest requests, Kubernetes cannot bind past
allocatable, so `c_1 ≡ 0` on the cluster **by construction, not by policy**.
`observation.py.constraint_costs` computes it anyway and the docstring says why
the result is circular. **P4 must not report "the capacity constraint was
satisfied on the real cluster" as a transfer result.** The only honest options are
(i) report `c_1 ≡ 0` and explain that the platform enforces it, or (ii) deliberately
under-declare requests to re-open the regime, which then measures a cluster we
lied to. Option (i) is recommended; it is a finding about where the constraint
lives, which is more interesting than a satisfied bound.

### G-10 Objectives — (b)/(c)

`cloud_env.py:231-234` returns `[-latency, -energy, -sla]` where

- **latency** = the in-system count (`cloud_env.py:228`), a Little's-law proxy,
  not a time. On the cluster real completion *times* are available
  (`status.containerStatuses[].state.terminated.{startedAt,finishedAt}`), which is
  a **better** measurement — and therefore **not the same quantity**. Report both,
  clearly separated: the in-system proxy for comparability with P3, and real
  completion time as the cluster-native metric. Do not silently swap one for the
  other inside a "transfer gap" table.
  **[TO VERIFY ON FIRST RUN]** `terminated` timestamps have 1-second granularity,
  which may be coarse relative to short pods.
- **energy** = the model of O-4/G-4. (c).
- **sla** = the count of G-7. (b), by our controller.

---

## 4. What P4 can and cannot claim

**Can claim**, if the runs are done as specified:

- A policy trained in `cloud_env` was executed in a real Kubernetes control loop,
  as a custom scheduler, driven by the same trace, against the default
  kube-scheduler on identical manifests.
- Measured outcomes: real completion times, deadline misses under an explicitly
  stated expiry rule, binding refusal rates, and a host-level energy proxy.
- A **quantified transfer gap** for latency and SLA — with the structural causes
  (G-3, G-6, G-8) named as candidate explanations rather than attributed to the
  policy.
- The bridges themselves as a reusable artefact, and **this document** as the
  substantive contribution: an itemised account of which simulator observables and
  actions survive contact with Kubernetes. PLAN_P4 §6 is right that this is the
  part other groups would cite.

**Cannot claim**, on this design:

1. That energy was measured on Kubernetes, or that per-node energy was measured at
   all (O-4, G-4).
2. That the capacity constraint `c_1` was satisfied *by the policy* (G-9).
3. That the policy operated on the same observation space it was trained on: the
   overcommit half of slots 0-1 is unreachable (O-1, O-2), slot 8 is synthesized
   (O-9), slot 3 is a formula (O-4).
4. That "migration" was demonstrated in the Kubernetes sense of moving a running
   workload (A-3). What is demonstrated is re-routing of unscheduled pods.
5. That the queueing dynamics transferred (G-6) — they demonstrably do not, and
   the direction of the resulting bias is predictable in advance.
6. That the cluster is a distributed system in the failure-domain sense: three
   containers on one laptop CPU share fate, cache, thermal budget and kernel.
7. Any statistical claim at P3's strength: the seed count will be far lower (G-2)
   and must be reported, not glossed.

**Decisions this document forces back to Achraf** (PLAN_P4 §7):

1. Energy: the honest options are a host-level proxy reported as an outcome, or
   dropping energy from P4's objectives. Per-node energy is off the table.
   → recommend: keep the modelled energy objective for comparability with P3, and
   report the host proxy separately with its baseline protocol.
2. `c_1`: report `≡ 0` with the platform explanation, or re-open the regime by
   under-declaring requests. → recommend option (i).
3. Whether to keep `max-pods=15` (meaningful slot-2 normalisation, refusals) or
   raise it (no refusals, slot 2 permanently clamped). → recommend keeping 15 and
   logging every refusal.

---

## 5. Addendum 2026-10-04 — what building the replay controller exposed

Written while building `k8s/controller/replay.py`. Still **no cluster has been
created**; items marked (code) were found by running code offline, items marked
(model) come from the `KubeModelBackend`, which is a model, not a measurement.

**C-1 Intra-step order is simulator state too — (c), fixed (code).**
`cloud_env.step` applies the N actions in a loop that mutates state as it goes:
a lower-index ACCEPT raises a backlog before a higher-index MIGRATE picks its
target, and a task migrated into an *empty* higher-index buffer is decided again
by that agent in the same step. The first `plan_step` planned every agent
against the pre-action snapshot. With it, the controller diverged from
`env.step` for the random and least-loaded policies (seed 30000, horizon 150);
`always-migrate` happened to match. `plan_step` is now sequential, and
`test_replay.py::test_sim_backend_reproduces_env_step` checks it bit for bit.
Kubernetes has no notion of this order. It is one more rule the harness carries.

**C-2 A refused ACCEPT is asynchronous, and the pod is lost — worse than A-1 said.**
The API server does not check fit on a `Binding`; it only rejects a pod that is
already bound (409). Fit is checked by the **kubelet after binding**: the pod
goes to `phase=Failed`, `reason=OutOf{cpu,memory,pods}`, and a Failed pod can
never be bound again. So (i) a refusal is seen one snapshot late, not as an error
on the call; (ii) the task can only come back as a **new pod**. The controller
re-creates it (`<name>-rN`) with the original arrival, deadline and FIFO
position, and counts it in `refused_kubelet` and `requeues`. That is a second
container start, with its latency. Exact reason strings: [TO VERIFY ON FIRST RUN].

**C-3 The comparison with kube-scheduler is dominated by G-8 (model).**
The agent policies decide at most one head task per agent per step (G-8).
kube-scheduler binds every pending pod that fits, as soon as it can. In the
dry run the *emulated* scheduler did better than every agent heuristic, both in
the simulator backend and in the Kubernetes model. So "policy vs default
scheduler" measures the decision-rate cap at least as much as the policy. Report
it that way, or add a capped-scheduler control.

**C-4 Two of the three heuristics need simulator state the cluster does not have.**
`LeastLoadedPolicy` reads every server's requests and capacity. That is
available from the API (b). `DriftPlusPenaltyPolicy` clones the env and steps it,
so on the cluster it plans with **cloud_env as its model of Kubernetes**. That
model has the wrong service discipline (G-6). It also needs the remaining work of
running pods, which is not observable (O-7). The controller supplies it as a
belief advanced by the simulator's FIFO rule (exact on the simulator backend,
tested with `remaining` hidden). It also supplies the server speed from the fleet
spec (G-3). Every log records each policy's `info_class`. Only `heur_local*` and
the learned actors read nothing beyond their 9-dim observation.

**C-5 The tau sweep partly collapses on Kubernetes (model).**
`LocalOnlyPolicy` with tau > 1 tries to overcommit. On the cluster those
bindings are refused (C-2), so tau = 1.0 and tau = 2.0 gave the same returns in
the model. That leaves fewer distinct points on the heuristic's front than in
simulation. The hypervolume comparison has to be recomputed on cluster points,
not carried over from simulation.

**C-6 Live-path defects found before first contact (code, all fixed).**
(i) `k3d-cluster.yaml` put `pltr-ma.io/role=agent` under `options.runtime.labels`.
Those are Docker labels, so the node selector would have matched nothing. Moved
to `options.k3s.nodeLabels`. (ii) `system-reserved` assumed 16 host threads. On
the 12-core VPS it reserved more than the capacity. It is now rendered per host
by `up.sh`. (iii) `LiveExecutor._retarget` patched the label but not the
`initial-agent` annotation the snapshot reads, so every MIGRATE would have been
undone on the next read. (iv) `snapshot_from_kubernetes` subtracted only foreign
pods inside our namespace, so CoreDNS (kube-system, on an agent) was never
subtracted. (v) It did not treat pods being deleted as terminal.

**C-7 Horizon.** P3 trained and evaluated every policy at `horizon=150`
(ExperimentConfig, run_experiment.py, `n3_trace/config.json`). G-2's "200-step
episode" is cloud_env's own default, not what P3 used. The controller defaults
to 150, so an episode at 1 s/step is about 2.5 min plus startup. Its seeds are
30000-30004, the same episodes as P3's `evaluate_omega` and heuristic fronts, so
sim-vs-cluster pairs are matched by episode.
