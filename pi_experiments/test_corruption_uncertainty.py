"""Check seed/image pairing instead of treating image corruptions as independent."""

import numpy as np

from .corruption_uncertainty import paired_interval


def test_opposite_case_effects_cancel_for_each_source_image():
    one = np.array([[1., -1., 1., 0.], [0., 1., -1., 1.]])
    cases = np.stack([one, -one], axis=1)
    interval = paired_interval(cases.mean(1), draws=200)
    assert interval['accuracy_delta'] == 0 and interval['ci95'] == [0., 0.]


def test_common_image_and_seed_resampling_matches_manual_reference():
    differences = np.array([[1., 0., -.5], [-1., .5, 0.]])
    interval = paired_interval(differences, draws=40, seed=31)
    rng = np.random.default_rng(31)
    seeds = rng.integers(0, 2, size=(40, 2))
    images = rng.integers(0, 3, size=(40, 3))
    values = np.array([differences[np.ix_(a, b)].mean() for a, b in zip(seeds, images)])
    assert interval['accuracy_delta'] == differences.mean()
    np.testing.assert_allclose(interval['ci95'], np.quantile(values, [.025, .975]))
    opposite = paired_interval(-differences, draws=40, seed=31)
    np.testing.assert_allclose(opposite['ci95'], -np.array(interval['ci95'])[::-1])
