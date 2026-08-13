# Archive Candidates

Every candidate, its reason, replacement, reference sites, and whether it is
safe to archive. Default action: **archive > delete**. Only `tests/te.py` and
the two root-level historical reports are actually moved during this cleanup
(see bottom of file). Outputs candidates (git-ignored generated artifacts) are
recorded for future cleanup but are intentionally **left in place** because they
are cheap to regenerate and moving them would only be cosmetic.

## Git-tracked candidates (acted on now)

| Path | Reason | Replacement | Referenced by | Safe to archive |
| --- | --- | --- | --- | --- |
| `tests/te.py` | Temporary `torch.cuda.is_available()` debug check; not a test (name `te.py`, not collected), not imported anywhere | none | none found | YES -> move to `archive/` |
| `Phase_1_SUTD_PRCM_Dataset_Infrastructure.md` | Unique Phase 1 report but belongs in `docs/` for a clean root | `docs/` location | none (no links) | YES -> move to `docs/` |
| `Phase_2_5_Forward_EM_Surrogate_Validation.md` | Unique Phase 2.5 task/report but belongs in `docs/` | `docs/` location; does not duplicate `docs/phase2_5_report.md` (that is the results report) | none (no links) | YES -> move to `docs/` |

## Git-ignored output candidates (recorded, NOT moved in this cleanup)

| Path | Reason | Replacement | Referenced by | Safe to archive |
| --- | --- | --- | --- | --- |
| `outputs/phase2_forward_30k_response_aware/` | Bare `best.pt` only, no report config/history; superseded by `_gpu` run (the one used by Phase 7B) | `outputs/phase2_forward_30k_response_aware_gpu/` | none | YES |
| `outputs/phase2_forward_30k_response_aware_v2/` | Bare `best.pt` only; superseded by `_gpu` run | `outputs/phase2_forward_30k_response_aware_gpu/` | none | YES |
| `outputs/phase2_forward/` | First forward run; superseded by stabilized `phase2_forward_75ep` and 30k runs; README historical reference will be removed by the README rewrite | `outputs/phase2_forward_75ep/`, `outputs/phase2_forward_30k/` | README (historical text only; being rewritten) | YES |
| `outputs/phase2_forward_evaluation/` | Evaluation plots only, for an early/superseded model | re-run `evaluate_forward.py` on the stabilized baseline | none | YES |
| `outputs/phase2_forward_scale_comparison/` | Empty directory | none | `scripts/compare_forward_scales.py` (default output dir, recreates on demand) | YES (delete) |

## Investigated and NOT candidates

| Path | Why kept |
| --- | --- |
| `outputs/physics_jepa/seed42_64d` | Partial (val/test only) but part of the v1 evidence set. Keep. |
| `outputs/phase2_forward_30k_independent/` | Independent held-out evaluation of the 30k forward run; referenced in evaluation workflows. Keep. |
| `outputs/phase6_rcwa/cache`, `outputs/phase6_1/cache` | RCWA solve caches under frozen validation infrastructure. Keep. |
| `docs/phase2_5_task.md` vs `docs/phase2_5_report.md` | Distinct task spec vs results report, not duplicates. Keep both. |

## Already executed archive actions (this cleanup)

1. `tests/te.py` -> `archive/` (moved).
2. `Phase_1_SUTD_PRCM_Dataset_Infrastructure.md` -> `docs/` (moved).
3. `Phase_2_5_Forward_EM_Surrogate_Validation.md` -> `docs/` (moved).

No file was deleted. All archive moves are zero-risk: no code or documented path
references them.