# Physics-JEPA v2 Collapse-Fix Full Run

## Experiment

Label: `physics_jepa_v2_collapse_fix`

The selected A3 formulation was trained on the 30k processed dataset with
latent dimension 32, seed 42, `lambda_variance=0.5`,
`lambda_covariance=0.05`, target centering disabled, and covariance applied to
both `z_online` and `z_geometry`. The model and shuffled-pair control used the
same architecture and training settings.

Both runs used Python 3.14.3, PyTorch 2.10.0+cu126, and CUDA.

| Run | Epochs | Best epoch | Best validation total loss |
|---|---:|---:|---:|
| Correct pairs | 65 | 55 | 0.055665 |
| Shuffled pairs | 75 | 66 | 0.056708 |

## Representation gate

Values below are on the cached test split. `raw` uses unnormalized latent
Euclidean distance; `normalized` uses the fixed L2-normalized distance used by
the representation gate.

| Representation | v1 raw rho | v2 raw rho | v1 normalized rho | v2 normalized rho | v1 rank-1 | v2 rank-1 |
|---|---:|---:|---:|---:|---:|---:|
| `z_target` | -0.0238 | +0.0112 | -0.0405 | -0.0169 | 0.8911 | 0.0378 |
| `z_pred` | +0.2939 | +0.0990 | -0.1716 | -0.1057 | 0.9759 | 0.7136 |

The covariance term therefore fixes the dominant low-rank target collapse, but
does not produce a physics-organized predictor representation. The predictor
remains substantially low-rank and its normalized EM-distance correlation is
still negative.

The correct-vs-shuffled control gives:

```text
correct z_pred normalized rho = -0.1057
shuffled z_pred normalized rho = +0.0050
separation = -0.1107
```

The response probe R2 for `z_pred` decreased from 0.1585 in v1 to 0.0424 in
v2. For `z_target`, it increased from 0.2030 to 0.3418, but this did not
transfer to the geometry-predicted latent. Resonance frequency MAE for
`z_pred` was 3.19/3.09 GHz in v1 (Ty/Rx) and 2.93/2.70 GHz in v2; feature-count
MAE was 0.4224 versus 0.4311.

For the same fixed pair-distance framework, the response-PCA sanity reference
has raw rho approximately +0.9528, +0.9797, +0.9952, and +0.9992 for 4, 8,
16, and 32 components respectively. Its L2-normalized rho is approximately
+0.2177, +0.4442, +0.4364, and +0.4304. Thus the metric remains feasible and
the v2 failure is not a metric artifact.

## Verdict

Outcome C: `physics_jepa_v2_collapse_fix` still fails the central
representation gate. The experiment is complete and should be recorded as a
negative result. Do not add further complexity or proceed to inverse
generation from this formulation without a separately justified hypothesis.

## Artifacts

- Correct-pair run: `outputs/physics_jepa_v2/seed42_32d`
- Shuffled control: `outputs/physics_jepa_v2/shuffled_32d`
- Gate report and plots: `outputs/physics_jepa_v2/evaluation_32d`
- Targeted tests: 18 passed
