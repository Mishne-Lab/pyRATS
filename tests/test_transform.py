"""Tests for fit and transform: embedding new points with the views of a fit.

transform must reproduce the embedding of the training points, place held-out
points next to their neighbors (also near a tear, where averaging neighbors
across the cut fails), agree with a refit that includes the new points, and
flag points far from the data. The data are small and generated in the files,
so that every fit takes a few seconds on one core.
"""

import warnings

import numpy as np
import pytest
from scipy.linalg import orthogonal_procrustes
from sklearn.neighbors import NearestNeighbors

from pyRATS import rats
from test_random_state import klein_bottle, sphere_with_hole


def split(X, frac=0.1, seed=0):
    """Hold out a random fraction of the points."""
    out = np.random.RandomState(seed).rand(len(X)) < frac
    return X[~out], X[out]


def jumps(y_train, X_train, X_new, y_new):
    """Distance of each new point's embedding to the embedding of its nearest
    training point, in units of the median such distance among the training
    points. Of order 1 for a point placed among its neighbors."""
    nn = NearestNeighbors().fit(X_train)
    near = nn.kneighbors(X_new, n_neighbors=1, return_distance=False)[:, 0]
    second = nn.kneighbors(X_train, n_neighbors=2, return_distance=False)[:, 1]
    typical = np.median(np.linalg.norm(y_train[second] - y_train, axis=1))
    return np.linalg.norm(y_new - y_train[near], axis=1) / typical


@pytest.fixture(scope="module")
def sphere_fit():
    X = sphere_with_hole(600)
    X_train, X_new = split(X)
    model = rats.RATS(n_jobs=1, random_state=7).fit(X_train)
    return model, X_train, X_new


@pytest.fixture(scope="module")
def torn_fit():
    X = klein_bottle(1200)
    X_train, X_new = split(X)
    model = rats.RATS(tear=True, n_jobs=1, random_state=7).fit(X_train)
    return model, X_train, X_new


def test_fit_returns_self_and_stores_the_embedding():
    X = sphere_with_hole(600)
    model = rats.RATS(n_jobs=1, random_state=7)
    assert model.fit(X) is model
    np.testing.assert_array_equal(
        model.embedding_, rats.RATS(n_jobs=1, random_state=7).fit_transform(X)
    )


def test_transform_reproduces_the_fit(sphere_fit):
    model, X_train, _ = sphere_fit
    np.testing.assert_allclose(model.transform(X_train), model.embedding_, rtol=0, atol=1e-10)


def test_transform_reproduces_the_fit_with_tearing(torn_fit):
    model, X_train, _ = torn_fit
    np.testing.assert_allclose(model.transform(X_train), model.embedding_, rtol=0, atol=1e-10)


def test_held_out_points_land_among_their_neighbors(sphere_fit):
    model, X_train, X_new = sphere_fit
    j = jumps(model.embedding_, X_train, X_new, model.transform(X_new))
    assert j.max() < 5


def test_held_out_points_near_the_tear_stay_on_one_side(torn_fit):
    model, X_train, X_new = torn_fit
    y_train = model.embedding_
    j = jumps(y_train, X_train, X_new, model.transform(X_new))
    # Averaging the embeddings of the neighbors puts some held-out points in
    # the middle of the tear: the test data do straddle it...
    nn = NearestNeighbors(n_neighbors=model.k).fit(X_train)
    y_mean = y_train[nn.kneighbors(X_new, return_distance=False)].mean(axis=1)
    assert (jumps(y_train, X_train, X_new, y_mean) > 5).sum() > 0
    # ...while transform keeps every one of them next to its neighbors.
    assert j.max() < 5


def test_transform_agrees_with_refitting():
    # A spherical cap, whose embedding barely changes when points are added:
    # the held-out points must land where a refit that includes them puts them.
    X = sphere_with_hole(1400, hole_frac=1.6)
    X_train, X_new = split(X)
    model = rats.RATS(n_jobs=1, random_state=7).fit(X_train)
    y_new = model.transform(X_new)
    refit = rats.RATS(n_jobs=1, random_state=7).fit_transform(np.vstack([X_train, X_new]))
    y_refit_train, y_refit_new = refit[: len(X_train)], refit[len(X_train) :]
    # Align the refit to the fit on the training points (rotation, reflection
    # and translation), then compare the held-out points.
    a_mean, b_mean = model.embedding_.mean(0), y_refit_train.mean(0)
    R, _ = orthogonal_procrustes(y_refit_train - b_mean, model.embedding_ - a_mean)
    placed = (y_refit_new - b_mean) @ R + a_mean
    second = NearestNeighbors().fit(X_train).kneighbors(X_train, n_neighbors=2, return_distance=False)[:, 1]
    typical = np.median(np.linalg.norm(model.embedding_[second] - model.embedding_, axis=1))
    errors = np.linalg.norm(placed - y_new, axis=1) / typical
    assert np.median(errors) < 0.25
    assert errors.max() < 1


def test_points_far_from_the_data_warn(sphere_fit):
    model, X_train, _ = sphere_fit
    far = X_train[:2] + 10.0
    with pytest.warns(UserWarning, match="not among the n_neighbors nearest neighbors"):
        y = model.transform(far)
    assert np.isfinite(y).all()


def test_residual_flags_points_off_the_manifold(sphere_fit):
    model, X_train, X_new = sphere_fit
    _, on = model.transform(X_new, return_residual=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, off = model.transform(X_new * 1.3, return_residual=True)  # pushed out of the sphere
    assert np.median(off) > 3 * np.median(on)


def test_transform_input_checks(sphere_fit):
    model, X_train, _ = sphere_fit
    with pytest.raises(RuntimeError, match="not fitted"):
        rats.RATS().transform(X_train)
    with pytest.raises(ValueError, match="shape"):
        model.transform(X_train[:, :2])


def test_transform_with_kernel_pca():
    X = sphere_with_hole(200)
    model = rats.RATS(kernel="rbf", n_neighbors=12, n_iter=3, n_jobs=1, random_state=7).fit(X)
    np.testing.assert_allclose(model.transform(X), model.embedding_, rtol=0, atol=1e-10)
    with pytest.raises(NotImplementedError, match="kernel=None"):
        model.transform(X, return_residual=True)
