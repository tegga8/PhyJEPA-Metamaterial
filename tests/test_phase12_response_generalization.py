import numpy as np

from scripts.evaluate_phase12_response_generalization import hamming_upper, pairwise_metrics


def test_hamming_upper_uses_unique_pairs_only():
    geometries = np.asarray([[[[0, 0], [0, 0]]], [[[1, 1], [1, 1]]], [[[0, 0], [0, 0]]]], dtype=np.uint8)
    assert np.allclose(hamming_upper(geometries), [1.0, 0.0, 1.0])


def test_good_diversity_is_restricted_to_good_candidates():
    geometries = np.asarray([[[[0, 0], [0, 0]]], [[[1, 1], [1, 1]]], [[[0, 0], [0, 0]]]], dtype=np.uint8)
    latents = np.zeros((3, 64, 8, 8), dtype=np.float32)
    responses = np.zeros((3, 4, 1001), dtype=np.float32)
    result = pairwise_metrics(geometries, latents, responses, np.asarray([.1, .2, .4]), np.asarray([True, True, False]))
    assert result["good_candidate_count"] == 2
    assert result["mean_good_pairwise_geometry_hamming"] == 1.0
