#!/usr/bin/env python3
"""learned.py — save, reload and deploy trained PLTR-MA / conditioned-network actors.

WHY THIS EXISTS. P3's harness (simulator/train.py) trains, evaluates and keeps only
the returns; no weights are written anywhere, so there was nothing to deploy on
the cluster. This module adds weight saving WITHOUT CHANGING ANY P3 FILE:

  * `train_variant` calls exactly what train._front_job / train._cn_job call —
    `train.build_variant(name, xc, seed)` then `algo.train(verbose=False)` — so
    the network it returns is the network P3 evaluated for that (config, seed).
    `verify` proves it: the ω-sweep returned by p3's own `_front_job` equals,
    bit for bit, the sweep of the network this module trained.
  * `save_actors` writes ONLY the decentralized actors (the critic is discarded
    at execution under CTDE; mappo.py docstring) plus provenance metadata and a
    sha256 sidecar.
  * `LoadedActorPolicy` rebuilds the actors and exposes `.act(obs, cond)` with the
    SAME arithmetic as `MAPPO.act` (mappo.py), so the replay controller can drive
    it like any other policy.

Because simulator/train.py and run_experiment.py are untouched, P3's defaults are
unchanged by construction (sha256 of every simulator file is recorded in each
weight file's metadata, and can be compared with the published image).

CLI:
    python k8s/policies/learned.py train --variant pltr_ma --seed 0 \
        --config p3/_results_backup_vps/n3_trace/config.json \
        --trace simulator/trace_alibaba.csv --out k8s/policies/weights/pltr_ma_s0.pt
    python k8s/policies/learned.py verify --variant pltr_ma --tiny
    python k8s/policies/learned.py info k8s/policies/weights/pltr_ma_s0.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
P3_CODE = REPO / "simulator"
if str(P3_CODE) not in sys.path:
    sys.path.insert(0, str(P3_CODE))

FORMAT = "pltr-ma-p4-actors/1"
#: Variants trained through ParetoMAPPO.train (the front jobs). The convergence
#: variants (mappo, lyapunov, cpo, lagrangian) are trained by train.fit with
#: periodic evaluation interleaved, which consumes RNG differently; they are not
#: offered here so that "the deployed network = the network P3 reported" holds.
TRAIN_VARIANTS = ("pltr_ma", "cn", "cn_lyap", "cn_tr")
P3_FILES = ("cloud_env.py", "mappo.py", "pareto.py", "trust_region.py", "lyapunov.py",
            "train.py", "trace.py", "run_experiment.py", "lagrangian_ppo.py")


def p3_fingerprint() -> dict:
    return {f: hashlib.sha256((P3_CODE / f).read_bytes()).hexdigest()
            for f in P3_FILES if (P3_CODE / f).exists()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def config_from_json(path: str | Path | None, **overrides):
    """ExperimentConfig from a P3 run's config.json (unknown keys rejected)."""
    from train import ExperimentConfig
    names = {f.name for f in fields(ExperimentConfig)}
    data = {}
    if path is not None:
        data = json.loads(Path(path).read_text())
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"{path}: keys not in ExperimentConfig: {sorted(unknown)}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    for k in ("V_grid", "delta_grid", "extra_fronts", "extra_conv"):
        if k in data and isinstance(data[k], list):
            data[k] = tuple(data[k])
    return ExperimentConfig(**data)


# --------------------------------------------------------------------------- #
# Train (P3's own call path) and save
# --------------------------------------------------------------------------- #
def train_variant(variant: str, xc, seed: int):
    """Exactly train._front_job / _cn_job / _variant_front_job up to the sweep."""
    if variant not in TRAIN_VARIANTS:
        raise ValueError(f"variant must be one of {TRAIN_VARIANTS}")
    from train import build_variant
    algo, _ = build_variant(variant, xc, seed=seed)
    algo.train(verbose=False)
    return algo


def save_actors(actors, path: str | Path, meta: dict | None = None,
                cond_dim: int = 0) -> Path:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sds = [{k: v.detach().cpu().clone() for k, v in a.state_dict().items()} for a in actors]
    first = sds[0]["net.0.weight"]
    blob = {
        "format": FORMAT,
        "n_agents": len(actors),
        "in_dim": int(first.shape[1]),
        "cond_dim": int(cond_dim),
        "obs_dim": int(first.shape[1]) - int(cond_dim),
        "hidden": int(first.shape[0]),
        "n_actions": int(sds[0]["net.4.weight"].shape[0]),
        "state_dicts": sds,
        "meta": dict(meta or {}),
        "torch_version": str(torch.__version__),
        "saved_unix": time.time(),
    }
    torch.save(blob, path)
    side = {k: v for k, v in blob.items() if k != "state_dicts"}
    side["sha256"] = _sha256(path)
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(side, indent=2, default=str))
    return path


def save_algo(algo, path, meta: dict | None = None) -> Path:
    return save_actors(algo.actors, path, meta=meta, cond_dim=algo.cond_dim)


def load_blob(path: str | Path) -> dict:
    import torch
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    if blob.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file (format={blob.get('format')!r})")
    side = Path(str(path) + ".json")
    if side.exists():
        want = json.loads(side.read_text()).get("sha256")
        if want and want != _sha256(Path(path)):
            raise ValueError(f"{path}: sha256 does not match its sidecar {side.name}")
    return blob


def build_actors(blob: dict):
    from mappo import Actor
    actors = []
    for sd in blob["state_dicts"]:
        a = Actor(blob["obs_dim"], blob["n_actions"], blob["hidden"], blob["cond_dim"])
        a.load_state_dict(sd, strict=True)
        a.eval()
        actors.append(a)
    return actors


# --------------------------------------------------------------------------- #
# P3 corrected production policies (protocol v1 §4) — the networks P4 deploys
# --------------------------------------------------------------------------- #
#: cloud_env's per-agent observation length (cloud_env.py `_build_obs`).
OBS_DIM = 9
#: P3's frozen production code (p3/revision_v2/FROZEN.json, 7 Nov 2026).
P3_PRODUCTION_CODE = "4e40af2b25305dc5c0b6a185829e51373b6a19fa80be2893e1e508dceedc7348"


def p3_production_meta(npz_path: str | Path) -> dict:
    """The `meta` JSON of a P3 production result file (runner.run_job)."""
    with np.load(Path(npz_path), allow_pickle=False) as z:
        return json.loads(str(z["meta"]))


def load_p3_production(pt_path: str | Path, npz_path: str | Path | None = None,
                       greedy: bool = True, expect_code: str | None = P3_PRODUCTION_CODE,
                       require_npz: bool = True) -> "LoadedActorPolicy":
    """Load the actors P3's corrected production run saved for one job.

    `runner.run_job` writes `<job_id>.pt` = `torch.save([actor.state_dict(), ...])`
    and records `policy_sha256` in `<job_id>.npz`'s meta
    (p3/revision_v2/src/runner.py:222-226). Gate W1 part (a): the file's sha256
    must equal that value, and the job's code_version must be the frozen code.
    Nothing is re-trained; the deployed network IS the one P3 evaluated.
    """
    import torch
    pt_path = Path(pt_path)
    npz_path = Path(npz_path) if npz_path is not None else pt_path.with_suffix(".npz")
    meta: dict = {}
    if npz_path.exists():
        meta = p3_production_meta(npz_path)
        want = meta.get("policy_sha256")
        got = _sha256(pt_path)
        if want != got:
            raise ValueError(f"{pt_path}: sha256 {got} != policy_sha256 {want} in {npz_path.name}")
        if expect_code is not None and meta.get("code_version") != expect_code:
            raise ValueError(f"{npz_path.name}: code_version {meta.get('code_version')} "
                             f"is not the frozen production code {expect_code}")
    elif require_npz:
        raise FileNotFoundError(f"{npz_path} not found: the policy hash cannot be checked")
    sds = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not isinstance(sds, list) or not sds:
        raise ValueError(f"{pt_path}: expected a list of actor state_dicts")
    first = sds[0]["net.0.weight"]
    in_dim = int(first.shape[1])
    blob = {
        "format": "p3-production-actors", "n_agents": len(sds), "in_dim": in_dim,
        "cond_dim": in_dim - OBS_DIM, "obs_dim": OBS_DIM, "hidden": int(first.shape[0]),
        "n_actions": int(sds[0]["net.4.weight"].shape[0]), "state_dicts": sds,
        "meta": {"variant": (meta.get("job", {}) or {}).get("method", "p3-production"),
                 "source": str(pt_path), "policy_sha256": _sha256(pt_path),
                 "job_id": meta.get("job_id"), "code_version": meta.get("code_version")},
    }
    if blob["cond_dim"] < 0:
        raise ValueError(f"{pt_path}: input width {in_dim} < observation width {OBS_DIM}")
    return LoadedActorPolicy(build_actors(blob), blob, greedy=greedy)


#: The three P3 production jobs P4 deploys (protocol v1 §4): setting trace_n3,
#: condition R, method pltr_ma@400, seeds 0-2. Their identity fields are rebuilt
#: from P3's OWN protocol objects, never typed by hand here.
PRODUCTION_METHOD = "pltr_ma@400"
PRODUCTION_SEEDS = (0, 1, 2)
#: role of a policy artifact; only "production" may enter a live primary run or
#: a scientific report
ROLES = ("production", "rehearsal")


def production_job_identity(seed: int) -> dict:
    import protocol as P
    j = P.Job("p4", P.T3, P.R, PRODUCTION_METHOD, int(seed), None, P.EvalPlan())
    d = json.loads(json.dumps(j.config_dict(), default=str))
    return {"job_id": j.job_id, **{k: d[k] for k in ("setting", "condition", "method",
                                                     "seed", "omega_index")}}


class AdmissionError(ValueError):
    pass


def admit_production_policy(pt_path, seed: int, w1_report, npz_path=None) -> dict:
    """Admission of one P3 production policy to the live primary run.

    Passes only if ALL hold:
      * file integrity and frozen code (load_p3_production);
      * the .npz job identity (setting, condition, method, seed, omega_index and
        job_id) equals P3's production job for `seed`;
      * `w1_report` (qa/w1_gate.py output) passed, and is bound to THIS .pt
        sha256, THIS .npz sha256, the same job_id and the frozen code.
    Returns the admission record stored in POLICIES.json (role "production")."""
    pt_path = Path(pt_path)
    npz_path = Path(npz_path) if npz_path is not None else pt_path.with_suffix(".npz")
    pol = load_p3_production(pt_path, npz_path, greedy=True)
    meta = p3_production_meta(npz_path)
    want = production_job_identity(seed)
    job = meta.get("job") or {}
    got = {"job_id": meta.get("job_id"), **{k: job.get(k) for k in want if k != "job_id"}}
    if got != want:
        raise AdmissionError(f"{pt_path.name}: job identity {got} is not P3 production "
                             f"{want}")
    rep_path = Path(w1_report)
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    pt_sha, npz_sha = _sha256(pt_path), _sha256(npz_path)
    checks = {"pass": rep.get("pass") is True,
              "policy_sha256": rep.get("policy_sha256") == pt_sha,
              "npz_sha256": rep.get("npz_sha256") == npz_sha,
              "job_id": rep.get("job_id") == want["job_id"],
              "code_version": rep.get("code_version") == P3_PRODUCTION_CODE,
              "n_checks": int(rep.get("n_checks") or 0) > 0}
    if not all(checks.values()):
        raise AdmissionError(f"{rep_path.name}: W1 report not valid for {pt_path.name}: "
                             f"{[k for k, v in checks.items() if not v]}")
    return {"role": "production", "pt": str(pt_path.resolve()), "sha256": pt_sha,
            "npz": str(npz_path.resolve()), "npz_sha256": npz_sha,
            "job_id": want["job_id"], "seed": int(seed),
            "code_version": P3_PRODUCTION_CODE, "w1_report": str(rep_path.resolve()),
            "w1_sha256": _sha256(rep_path), "variant": pol.blob["meta"]["variant"]}


def recheck_admission(rec: dict) -> list[str]:
    """Files of an admission record unchanged since admission? [] = yes."""
    bad = []
    for k_path, k_sha in (("pt", "sha256"), ("npz", "npz_sha256"), ("w1_report", "w1_sha256")):
        if k_path not in rec:
            bad.append(f"{k_path} missing from the record")
            continue
        p = Path(rec[k_path])
        if not p.exists() or _sha256(p) != rec[k_sha]:
            bad.append(f"{k_path} changed or missing: {p}")
    return bad


def load_into_algo(algo, path) -> dict:
    """Copy saved actor weights into an existing MAPPO-family object."""
    blob = load_blob(path)
    if blob["n_agents"] != len(algo.actors) or blob["cond_dim"] != algo.cond_dim:
        raise ValueError("saved actors do not match this algorithm's shape")
    for a, sd in zip(algo.actors, blob["state_dicts"]):
        a.load_state_dict(sd, strict=True)
    return blob


class LoadedActorPolicy:
    """Deployable decentralized actors. `.act(obs, cond)` == MAPPO.act.

    Each actor sees only its own 9-dim observation (information class
    "local-obs"), exactly as at P3 evaluation. Sampling uses torch's global RNG,
    as MAPPO.act does; seed it with torch.manual_seed for a reproducible run.
    """

    reads_env = False
    native_scheduler = False

    def __init__(self, actors, blob: dict, greedy: bool = False):
        self.actors = actors
        self.blob = blob
        self.greedy = bool(greedy)
        self.cond_dim = int(blob["cond_dim"])
        meta = blob.get("meta", {})
        self.name = f"learned:{meta.get('variant', 'actor')}" + (":greedy" if greedy else "")
        self.info_class = "local-obs (learned actor)"

    @classmethod
    def load(cls, path, greedy: bool = False) -> "LoadedActorPolicy":
        blob = load_blob(path)
        return cls(build_actors(blob), blob, greedy=greedy)

    def act(self, obs, cond=None, greedy: bool | None = None):
        import torch
        greedy = self.greedy if greedy is None else greedy
        n = len(self.actors)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            cond_t = None
            if self.cond_dim:
                if cond is None:
                    raise ValueError("this actor is preference-conditioned; pass cond=omega")
                c = np.broadcast_to(cond, (n, self.cond_dim)).astype(np.float32)
                cond_t = torch.as_tensor(c)
            actions, logps = [], []
            for i, actor in enumerate(self.actors):
                d = actor.dist(obs_t[i:i + 1], None if cond_t is None else cond_t[i:i + 1])
                a = d.probs.argmax(dim=-1) if greedy else d.sample()
                actions.append(int(a.item()))
                logps.append(float(d.log_prob(a).item()))
        return np.array(actions), np.array(logps, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Verification: deployed network == the network P3 evaluated
# --------------------------------------------------------------------------- #
def tiny_config(trace: str | None):
    return config_from_json(None, n_servers=3, horizon=20, n_iterations=3,
                            eval_episodes=2, pareto_grid=2, pareto_seeds=1,
                            hv_mc=2000, n_workers=1, trace_path=trace, out_dir="unused")


def verify(variant: str, xc, seed: int, workdir: Path) -> dict:
    """Three bit-identity checks; returns a report dict (all must be True).

    1. p3's own `_front_job`/`_cn_job` sweep == sweep of `train_variant`'s
       network (this module trains the same network P3 evaluated);
    2. save -> reload: every actor tensor is torch.equal;
    3. the reloaded actors, put in a FRESH algorithm object, produce the same
       stochastic AND greedy ω-sweep as the trained one, under the same torch seed.
    """
    import torch
    import train as T

    job = {"pltr_ma": T._front_job, "cn": T._cn_job}.get(variant)
    if job is not None:
        ref = job((xc, seed))
        ref = ref[0] if isinstance(ref, tuple) else ref
    else:
        ref = T._variant_front_job((xc, variant, seed))
    algo = train_variant(variant, xc, seed)
    _, mine, _ = algo.sweep_front(n_grid=xc.pareto_grid, n_ep=xc.eval_episodes)
    check1 = bool(np.array_equal(ref, mine))

    path = save_algo(algo, workdir / f"{variant}_s{seed}.pt",
                     meta={"variant": variant, "seed": seed, "config": asdict(xc)})
    blob = load_blob(path)
    check2 = all(torch.equal(v, a.state_dict()[k])
                 for a, sd in zip(algo.actors, blob["state_dicts"]) for k, v in sd.items())

    fresh, _ = T.build_variant(variant, xc, seed=seed + 1000)   # different init
    load_into_algo(fresh, path)
    out = {}
    for greedy in (False, True):
        torch.manual_seed(1234)
        _, r_a, _ = algo.sweep_front(n_grid=xc.pareto_grid, n_ep=xc.eval_episodes, greedy=greedy)
        torch.manual_seed(1234)
        _, r_b, _ = fresh.sweep_front(n_grid=xc.pareto_grid, n_ep=xc.eval_episodes, greedy=greedy)
        out["greedy" if greedy else "stochastic"] = bool(np.array_equal(r_a, r_b))
    # and the standalone deployable policy agrees with MAPPO.act action-for-action
    pol = LoadedActorPolicy.load(path)
    env = algo.env
    obs, _ = env.reset(seed=30000)
    omega = np.full(3, 1 / 3, dtype=np.float32)
    same = True
    for _ in range(xc.horizon):
        torch.manual_seed(77)
        a1, l1 = algo.act(obs, cond=omega)
        torch.manual_seed(77)
        a2, l2 = pol.act(obs, omega)
        same &= bool(np.array_equal(a1, a2) and np.array_equal(l1, l2))
        obs, *_ = env.step(a1)
    return {"p3_job_equals_p4_training": check1, "tensors_round_trip": bool(check2),
            "reloaded_sweep_stochastic": out["stochastic"],
            "reloaded_sweep_greedy": out["greedy"], "policy_act_equals_mappo_act": same,
            "ref_returns": np.asarray(ref).tolist(), "weights": str(path)}


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--variant", default="pltr_ma", choices=TRAIN_VARIANTS)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--config", default=None, help="a P3 run's config.json")
    t.add_argument("--trace", default=None, help="local trace path (overrides config)")
    t.add_argument("--n-iterations", type=int, default=None)
    t.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--variant", default="pltr_ma", choices=TRAIN_VARIANTS)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--trace", default=str(REPO / "simulator" / "trace_alibaba.csv"))
    v.add_argument("--workdir", default=None)
    i = sub.add_parser("info")
    i.add_argument("path")
    ns = ap.parse_args(argv)

    if ns.cmd == "info":
        blob = load_blob(ns.path)
        print(json.dumps({k: v for k, v in blob.items() if k != "state_dicts"},
                         indent=2, default=str))
        return 0

    import torch
    torch.set_num_threads(1)
    if ns.cmd == "train":
        xc = config_from_json(ns.config, trace_path=ns.trace, n_iterations=ns.n_iterations,
                              n_workers=1, device="cpu")
        t0 = time.time()
        algo = train_variant(ns.variant, xc, ns.seed)
        meta = {"variant": ns.variant, "seed": ns.seed, "config": asdict(xc),
                "config_source": ns.config, "train_wall_s": time.time() - t0,
                "p3_code_sha256": p3_fingerprint(),
                "trained_by": "k8s/policies/learned.py: train.build_variant + "
                              "algo.train(verbose=False), as train._front_job"}
        p = save_algo(algo, ns.out, meta)
        print(f"saved {p}  ({meta['train_wall_s']:.1f}s)")
        return 0

    import tempfile
    work = Path(ns.workdir or tempfile.mkdtemp(prefix="p4verify_"))
    rep = verify(ns.variant, tiny_config(ns.trace), ns.seed, work)
    print(json.dumps(rep, indent=2))
    ok = all(v for k, v in rep.items() if isinstance(v, bool))
    print("VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
