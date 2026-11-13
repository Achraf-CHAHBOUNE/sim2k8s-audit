"""Weight save/load round-trip. Needs torch — run on a host where it imports
(on this laptop torch may be blocked by Windows Application Control; the tests
then SKIP, they do not pass vacuously — see the skip reason)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "controller"))
import _paths  # noqa: E402,F401  (P3 tree, same rule everywhere)

torch = pytest.importorskip("torch")
import learned  # noqa: E402

TRACE = str(_paths.DEFAULT_TRACE)


@pytest.mark.parametrize("variant", ["pltr_ma", "cn"])
def test_verify_bit_identity(variant, tmp_path):
    rep = learned.verify(variant, learned.tiny_config(TRACE), 0, tmp_path)
    for k, v in rep.items():
        if isinstance(v, bool):
            assert v, k


def test_sha256_sidecar_detects_tampering(tmp_path):
    from mappo import Actor
    actors = [Actor(9, 3, 8, cond_dim=3) for _ in range(3)]
    p = learned.save_actors(actors, tmp_path / "a.pt", meta={"variant": "t"}, cond_dim=3)
    learned.load_blob(p)
    data = bytearray(p.read_bytes())
    data[-10] ^= 0xFF
    p.write_bytes(bytes(data))
    with pytest.raises(Exception):
        learned.load_blob(p)


def test_conditioned_actor_requires_omega(tmp_path):
    from mappo import Actor
    actors = [Actor(9, 3, 8, cond_dim=3) for _ in range(3)]
    p = learned.save_actors(actors, tmp_path / "a.pt", cond_dim=3)
    pol = learned.LoadedActorPolicy.load(p, greedy=True)
    with pytest.raises(ValueError):
        pol.act(np.zeros((3, 9), np.float32), None)
    a, _ = pol.act(np.zeros((3, 9), np.float32), np.full(3, 1 / 3, np.float32))
    assert a.shape == (3,)


def test_config_from_published_run_json():
    cfg = HERE.parents[1] / "simulator" / "n3_trace_config.json"
    if not cfg.exists():
        pytest.skip("published config not present")
    xc = learned.config_from_json(cfg, trace_path=TRACE)
    assert (xc.n_servers, xc.horizon, xc.n_iterations) == (3, 150, 400)
    assert xc.V_grid == (0.1, 0.3, 0.5, 1.0, 3.0)
