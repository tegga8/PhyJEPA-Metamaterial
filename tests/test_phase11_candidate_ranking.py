import numpy as np

from scripts.rank_phase10_candidates import deduplicate_by_geometry, passes_mode, rank_candidates


def base_row(target_index, candidate_index, response_mse, valid=True, pixel_novelty=0.1, latent_novelty=0.5, geometry_hash=None):
    return {
        "target_index": target_index,
        "candidate_index": candidate_index,
        "target_sample_id": f"sample-{target_index}",
        "complexity_group": "simple",
        "response_mse": response_mse,
        "valid": valid,
        "nearest_train_pixel_hamming": pixel_novelty,
        "nearest_train_latent_mse": latent_novelty,
        "pixel_hamming": 0.2,
        "occupancy_abs_difference": 0.1,
        "geometry_hash": geometry_hash or f"{target_index}-{candidate_index}",
    }


def test_valid_novel_filter_requires_validity_and_both_novelty_metrics():
    assert passes_mode(base_row(0, 0, 0.1), "valid_novel", 0.05, 0.25)
    assert not passes_mode(base_row(0, 0, 0.1, valid=False), "valid_novel", 0.05, 0.25)
    assert not passes_mode(base_row(0, 0, 0.1, pixel_novelty=0.01), "valid_novel", 0.05, 0.25)
    assert not passes_mode(base_row(0, 0, 0.1, latent_novelty=0.01), "valid_novel", 0.05, 0.25)


def test_deduplicate_by_geometry_keeps_lowest_response_error():
    rows = [
        base_row(0, 0, 0.4, geometry_hash="same"),
        base_row(0, 1, 0.2, geometry_hash="same"),
        base_row(0, 2, 0.3, geometry_hash="different"),
    ]
    kept = deduplicate_by_geometry(rows)
    assert len(kept) == 2
    assert min(row["response_mse"] for row in kept if row["geometry_hash"] == "same") == 0.2


def test_rank_candidates_counts_missing_successes_as_failures():
    rows = [
        base_row(0, 0, 0.1),
        base_row(0, 1, 0.2),
        base_row(1, 0, 0.6),
        base_row(1, 1, 0.7),
        base_row(2, 0, 0.1, pixel_novelty=0.01),
        base_row(2, 1, 0.2, pixel_novelty=0.01),
    ]
    report, selected = rank_candidates(rows, min_pixel_novelty=0.05, min_latent_novelty=0.25, response_threshold=0.3)
    assert report["metrics"]["valid"]["multi_solution_success_fraction"] == 2 / 3
    assert report["metrics"]["valid_novel"]["multi_solution_success_fraction"] == 1 / 3
    assert report["metrics"]["valid_novel"]["targets_missing"] == 1
    assert len([row for row in selected if row["ranking_mode"] == "valid"]) == 3
    assert np.isclose(report["metrics"]["valid"]["best_response_mse_mean"], (0.1 + 0.6 + 0.1) / 3)
