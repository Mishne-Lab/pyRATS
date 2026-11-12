"""Tests for the ``random_state`` parameter of RATS.

That the default (42) reproduces the results of earlier versions is checked
against a baseline commit by ``test_e2e.py``. These tests check that the seed
is honored: the same seed gives the same embedding and another seed another
one, the seed reaches the worker processes and the tear coloring, and NumPy's
global random state is left alone.

The data are generated here, as in ``examples/datasets.py`` but without its
optional dependencies, and are small, so that every fit takes a few seconds
on one core.
"""

import numpy as np
import pytest

from pyRATS import rats


def sphere_with_hole(n=600, hole_frac=1 / 6):
    """Deterministic points on a sphere with a hole (a Fibonacci lattice)."""
    radius = np.sqrt(1 / (4 * np.pi))
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + np.sqrt(5)) * i
    X = radius * np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        axis=1,
    )
    top = X[:, 2].max()
    hole = X[:, 0] ** 2 + X[:, 1] ** 2 + (X[:, 2] - top) ** 2 < (hole_frac * radius) ** 2
    return X[~hole]


def klein_bottle(n=600, seed=42):
    """Points on a Klein bottle in 4-D, a closed surface that RATS tears."""
    rng = np.random.RandomState(seed)
    side_x, side_y = 2.0, 0.5
    r_out, r_in = side_x / (2 * np.pi), side_y / (2 * np.pi)
    u = rng.uniform(0, side_x, n) / r_out
    v = rng.uniform(0, side_y, n) / r_in
    ring = r_out + r_in * np.cos(v)
    return np.stack(
        [
            ring * np.cos(u),
            ring * np.sin(u),
            r_in * np.sin(v) * np.cos(u / 2),
            r_in * np.sin(v) * np.sin(u / 2),
        ],
        axis=1,
    )


def embed(X, **kwargs):
    kwargs.setdefault("n_jobs", 1)
    y = rats.RATS(**kwargs).fit_transform(X)
    return y[0] if isinstance(y, tuple) else y


@pytest.fixture(scope="module")
def sphere():
    return sphere_with_hole()


def test_same_seed_same_embedding(sphere):
    np.testing.assert_array_equal(
        embed(sphere, random_state=7), embed(sphere, random_state=7)
    )


def test_different_seeds_different_embeddings(sphere):
    assert not np.array_equal(
        embed(sphere, random_state=0), embed(sphere, random_state=1)
    )


def test_default_seed_is_42(sphere):
    assert rats.RATS().random_state == 42
    np.testing.assert_array_equal(embed(sphere), embed(sphere, random_state=42))


@pytest.mark.parametrize("make", [np.random.RandomState, np.random.default_rng])
def test_generators_supply_one_seed_per_fit(make):
    # Two fresh generators with the same seed give the same seed for a fit...
    assert rats._check_random_seed(make(7)) == rats._check_random_seed(make(7))
    # ...and one generator gives a new seed for every fit.
    generator = make(7)
    assert rats._check_random_seed(generator) != rats._check_random_seed(generator)


def test_none_draws_the_seed_from_the_global_generator():
    np.random.seed(3)
    first = rats._check_random_seed(None)
    np.random.seed(3)
    assert rats._check_random_seed(None) == first


def test_seed_reaches_worker_processes():
    # With 1000 points or more and n_jobs > 1, the local PCA runs in worker
    # processes, which must receive the same seed.
    X = sphere_with_hole(1200)
    np.testing.assert_array_equal(
        embed(X, random_state=7, n_jobs=1), embed(X, random_state=7, n_jobs=2)
    )


def test_global_random_state_is_untouched(sphere):
    np.random.seed(123)
    expected = np.random.random_sample(5)
    np.random.seed(123)
    embed(sphere, random_state=7)
    np.testing.assert_array_equal(np.random.random_sample(5), expected)


@pytest.fixture(scope="module")
def torn():
    X = klein_bottle()
    models = [rats.RATS(tear=True, random_state=7, n_jobs=1) for _ in range(2)]
    ys = []
    for model in models:
        y = model.fit_transform(X)
        ys.append(y[0] if isinstance(y, tuple) else y)
    return models, ys


def test_same_seed_same_tear_colors(torn):
    models, ys = torn
    colors = [m.compute_color_of_pts_on_tear(y, [1, 2]) for m, y in zip(models, ys)]
    assert colors[0] is not None
    np.testing.assert_array_equal(colors[0], colors[1])


def test_tear_colors_use_the_seed_of_the_fit(torn, monkeypatch):
    models, ys = torn
    received = {}
    monkeypatch.setattr(
        rats, "compute_color_of_pts_on_tear", lambda *a, **kw: received.update(kw)
    )
    models[0].compute_color_of_pts_on_tear(ys[0])
    assert received["random_state"] == 7


@pytest.mark.parametrize("bad", [-1, 2**32, "a"])
def test_invalid_random_state(sphere, bad):
    with pytest.raises(ValueError, match="random_state"):
        rats.RATS(random_state=bad).fit_transform(sphere)
