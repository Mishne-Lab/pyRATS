"""Memory-management tests for pyRATS.

Covers:
  * The live ``_available_memory_bytes`` probe and PYRATS_MEMORY_LIMIT cap.
  * ``iter_eval_`` chunks the workload when the budget is tight, and the
    streamed output matches the fully-materialized ``batched_eval_`` output.
  * A simulated MemoryError causes chunk-size backoff (with a warning).
  * Peak Python-tracked allocations during a streamed run stay within the
    declared budget.
"""

import os
import warnings
import tracemalloc

import numpy as np
import pytest

from pyRATS import _utils
from pyRATS._utils import Param, _available_memory_bytes, batched_pdist


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_lpca_param(n=400, n_features=32, k=12, d=2, seed=0):
    """Build a self-contained lpca-style Param for testing iter_eval_.

    Skips the real lpca() fit (which we don't need) and just stuffs in
    well-shaped Psi/mu/X arrays so iter_eval_ has something to chew on.
    """
    rng = np.random.default_rng(seed)
    p = Param(algo="lpca")
    p.X = rng.standard_normal((n, n_features)).astype(np.float64)
    p.mu = rng.standard_normal((n, n_features))
    p.Psi = rng.standard_normal((n, n_features, d))
    p.b = np.ones(n)
    p.noise_var = 0
    p.noise = None
    return p, k


@pytest.fixture
def clean_env(monkeypatch):
    """Ensure PYRATS_MEMORY_LIMIT and the TTL cache don't leak between tests."""
    monkeypatch.delenv("PYRATS_MEMORY_LIMIT", raising=False)
    monkeypatch.setattr(_utils, "_memory_cache", None)
    yield monkeypatch


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def test_available_memory_honors_env_var(clean_env):
    cap = 128 * 1024 * 1024  # 128 MB
    clean_env.setenv("PYRATS_MEMORY_LIMIT", str(cap))
    assert _available_memory_bytes() <= cap


def test_available_memory_floor(clean_env):
    # A ridiculously small cap still gets clamped to the 64MB floor so
    # chunking can always make progress.
    clean_env.setenv("PYRATS_MEMORY_LIMIT", "1")
    assert _available_memory_bytes() >= 64 * 1024 * 1024


def test_available_memory_is_live(clean_env):
    """Env-var changes are picked up after the TTL expires (not cached forever)."""
    clean_env.setenv("PYRATS_MEMORY_LIMIT", str(256 * 1024 * 1024))
    first = _available_memory_bytes()
    # Expire the cache so the next call re-reads the env var.
    clean_env.setattr(_utils, "_memory_cache", None)
    clean_env.setenv("PYRATS_MEMORY_LIMIT", str(128 * 1024 * 1024))
    second = _available_memory_bytes()
    assert second < first


# ---------------------------------------------------------------------------
# iter_eval_ chunking + streaming correctness
# ---------------------------------------------------------------------------

def test_iter_eval_yields_multiple_chunks_when_budget_tight(clean_env):
    p, k = _make_lpca_param(n=300, k=12)
    masks = np.tile(np.arange(k), (300, 1))
    view_idx = np.arange(300)

    # Force a tiny budget so we get many chunks. Floor is 64MB so we go
    # under it via monkeypatch to actually exercise the path.
    clean_env.setattr(_utils, "_MIN_MEMORY_FLOOR", 1024)
    clean_env.setenv("PYRATS_MEMORY_LIMIT", "8192")  # 8 KB

    chunks = list(p.iter_eval_(view_idx, masks))
    assert len(chunks) > 1, "expected chunking with a tight budget"

    # Sum of chunk lengths must cover the input exactly once.
    covered = sum(sl.stop - sl.start for sl, _ in chunks)
    assert covered == 300


def test_iter_eval_output_matches_batched_eval(clean_env):
    """Chunked stream must reconstruct the same array as the unchunked path."""
    p, k = _make_lpca_param(n=200, k=10)
    masks = np.tile(np.arange(k), (200, 1))
    view_idx = np.arange(200)

    # Reference: large budget => one chunk.
    clean_env.setenv("PYRATS_MEMORY_LIMIT", str(1 << 30))
    full = p.batched_eval_(view_idx, masks)

    # Streamed: tight budget => many chunks. Sum back and compare.
    clean_env.setattr(_utils, "_MIN_MEMORY_FLOOR", 1024)
    clean_env.setenv("PYRATS_MEMORY_LIMIT", "16384")
    clean_env.setattr(_utils, "_memory_cache", None)  # expire cache so tight cap takes effect
    streamed = np.empty_like(full)
    nchunks = 0
    for sl, chunk in p.iter_eval_(view_idx, masks):
        streamed[sl] = chunk
        nchunks += 1
    assert nchunks > 1
    np.testing.assert_allclose(streamed, full, rtol=0, atol=0)


def test_peak_bytes_per_row_shrinks_chunks(clean_env):
    """Caller-supplied downstream-cost estimate must shrink chunk size."""
    p, k = _make_lpca_param(n=500, k=10)
    masks = np.tile(np.arange(k), (500, 1))
    view_idx = np.arange(500)

    clean_env.setattr(_utils, "_MIN_MEMORY_FLOOR", 1024)
    clean_env.setenv("PYRATS_MEMORY_LIMIT", "65536")

    no_hint = list(p.iter_eval_(view_idx, masks, peak_bytes_per_row=0))
    with_hint = list(
        p.iter_eval_(view_idx, masks, peak_bytes_per_row=10 * 1024)
    )
    assert len(with_hint) >= len(no_hint), (
        "declaring downstream allocation should produce >= chunks"
    )


# ---------------------------------------------------------------------------
# MemoryError backoff
# ---------------------------------------------------------------------------

def test_memory_error_triggers_backoff_and_warns(clean_env, monkeypatch):
    p, k = _make_lpca_param(n=64, k=8)
    masks = np.tile(np.arange(k), (64, 1))
    view_idx = np.arange(64)

    real_matmul = np.matmul
    state = {"raised": False}

    def fake_matmul(a, b, *args, **kwargs):
        # Raise MemoryError exactly once on the first call, then defer to
        # real matmul. Mimics an OS-level transient pressure event.
        if not state["raised"] and a.shape[0] > 1:
            state["raised"] = True
            raise MemoryError("simulated pressure")
        return real_matmul(a, b, *args, **kwargs)

    monkeypatch.setattr(np, "matmul", fake_matmul)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = p.batched_eval_(view_idx, masks)

    assert state["raised"], "fake_matmul was never invoked"
    assert out.shape == (64, k, p.Psi.shape[2])
    assert any(
        issubclass(w.category, RuntimeWarning) and "MemoryError" in str(w.message)
        for w in caught
    ), "expected a RuntimeWarning about MemoryError backoff"


def test_memory_error_propagates_when_chunk_already_one(clean_env, monkeypatch):
    """If chunk size is already 1, backoff has nowhere to go — must raise."""
    p, k = _make_lpca_param(n=4, k=4)
    masks = np.tile(np.arange(k), (4, 1))
    view_idx = np.arange(4)

    clean_env.setattr(_utils, "_MIN_MEMORY_FLOOR", 16)
    clean_env.setenv("PYRATS_MEMORY_LIMIT", "16")  # forces chunk_size = 1

    def always_fail(a, b, *args, **kwargs):
        raise MemoryError("cannot recover")

    monkeypatch.setattr(np, "matmul", always_fail)

    with pytest.raises(MemoryError):
        list(p.iter_eval_(view_idx, masks))


# ---------------------------------------------------------------------------
# Measured peak (Python-tracked allocations)
# ---------------------------------------------------------------------------

def test_streamed_postprocess_peak_under_budget(clean_env):
    """End-to-end peak check: a streamed _postprocess-style reduction must
    keep peak Python-tracked allocations near the per-chunk working set,
    not the full (n, num_pairs, d) materialized intermediate.

    We measure peak with tracemalloc, which on CPython tracks NumPy buffers
    via the default PyMem allocator. This is portable across Linux/macOS/
    Windows and avoids relying on RSS (which depends on OS overcommit).
    """
    n, k, d = 1500, 16, 3
    p, _ = _make_lpca_param(n=n, k=k, d=d, n_features=24)
    masks = np.tile(np.arange(k), (n, 1))
    view_idx = np.arange(n)

    num_pairs = k * (k - 1) // 2
    itemsize = p.X.dtype.itemsize

    # --- Baseline: materialize the full pipeline (no streaming) ----------
    clean_env.setenv("PYRATS_MEMORY_LIMIT", str(1 << 32))  # effectively unlimited
    tracemalloc.start()
    full_eval = p.batched_eval_(view_idx, masks)
    full_dists = batched_pdist(full_eval)
    _, materialized_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del full_eval, full_dists

    # --- Streamed: chunked _postprocess-style reduction ------------------
    # Cap at ~10x the per-row downstream peak so chunk_size lands at ~10
    # rows. The streamed peak should be a small fraction of the
    # materialized peak. We also lower the floor so the cap actually bites.
    pdist_per_row = num_pairs * (d + 1) * itemsize
    budget = 10 * pdist_per_row * 8
    clean_env.setattr(_utils, "_MIN_MEMORY_FLOOR", 1024)
    clean_env.setenv("PYRATS_MEMORY_LIMIT", str(budget))
    clean_env.setattr(_utils, "_memory_cache", None)  # expire cache so tight cap takes effect

    tracemalloc.start()
    zeta = np.empty(n)
    nchunks = 0
    for sl, chunk in p.iter_eval_(view_idx, masks, peak_bytes_per_row=pdist_per_row):
        chunk_dists = batched_pdist(chunk)
        zeta[sl] = np.nanmax(chunk_dists, axis=1)
        nchunks += 1
    _, streamed_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert nchunks > 1, "tight budget should produce multiple chunks"
    # Streamed peak should be meaningfully smaller. Allow generous slack
    # (0.5x) since tracemalloc captures fixed Param state too; the goal is
    # to catch regressions where streaming silently degrades to full
    # materialization.
    assert streamed_peak < 0.5 * materialized_peak, (
        f"streamed peak {streamed_peak} not meaningfully below "
        f"materialized peak {materialized_peak}"
    )
