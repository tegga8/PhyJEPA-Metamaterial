# Phase 1 --- SUTD-PRCM Dataset & Infrastructure

## Objective

Build a clean, reproducible data pipeline for the first experiment:

> **Given a complete 16×16 electromagnetic metasurface, load its
> geometry and electromagnetic response into PyTorch and verify that the
> dataset is suitable for masked-structure completion.**

At the end of Phase 1, we should be able to:

1.  Download/access SUTD-PRCM.
2.  Understand the exact file structure and data format.
3.  Load one sample correctly.
4.  Visualize random metasurface structures.
5.  Inspect the electromagnetic response.
6.  Create reproducible train/validation/test splits.
7.  Build a PyTorch `Dataset` and `DataLoader`.
8.  Save a small 5,000-sample working subset.
9.  Verify tensor shapes, ranges, and data quality.

------------------------------------------------------------------------

# 1. Dataset

## Primary dataset

**SUTD-PRCM --- SUTD Polarized Reflection of Complex Metasurfaces**

Repository:

https://github.com/veya2ztn/SUTD_PRCM_dataset

Paper:

https://arxiv.org/abs/2203.00002

The dataset contains approximately 260,000 simulated metasurface
samples.

The useful representation for our project is:

``` text
Metasurface geometry
        ↓
   16 × 16 binary pattern
        ↓
Electromagnetic simulation
        ↓
Reflection / EM response
```

This is particularly suitable because the geometry is spatial rather
than merely a small vector of design parameters.

------------------------------------------------------------------------

# 2. Do NOT use the complete dataset initially

For the first experiment use:

``` text
5,000 samples
```

Then scale only after the pipeline works:

``` text
5k → 30k → 100k → 260k
```

The purpose of the 5k subset is debugging, not final benchmarking.

------------------------------------------------------------------------

# 3. Project directory

Create:

``` text
metamaterial-jepa/
│
├── data/
│   ├── raw/
│   │   └── SUTD_PRCM/
│   │
│   ├── processed/
│   │   └── sutd_prcm_5k/
│   │
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
│
├── notebooks/
│   ├── 01_inspect_dataset.ipynb
│   └── 02_visualize_samples.ipynb
│
├── src/
│   ├── dataset.py
│   ├── preprocess.py
│   └── utils.py
│
├── configs/
│   └── phase1.yaml
│
├── scripts/
│   └── inspect_dataset.py
│
├── tests/
│   └── test_dataset.py
│
├── README.md
└── requirements.txt
```

Keep the raw dataset untouched.

------------------------------------------------------------------------

# 4. Environment

Use Python + PyTorch.

Minimum packages:

``` text
python
numpy
pandas
matplotlib
scipy
scikit-learn
torch
torchvision
h5py
jupyter
```

Install only what the actual dataset format requires after inspection.

------------------------------------------------------------------------

# 5. First task --- download and inspect

Do not immediately write a training pipeline.

First determine:

``` text
1. What files exist?
2. Which file contains geometry?
3. Which file contains EM response?
4. What is the geometry shape?
5. What is the response shape?
6. What frequencies are used?
7. What does each geometry value mean?
8. Are there multiple polarizations?
9. Are there multiple response channels?
10. Are there duplicated samples?
```

Run a basic inspection script.

The script should print things like:

``` text
Number of samples:
Geometry shape:
Response shape:
Geometry dtype:
Response dtype:
Geometry min/max:
Response min/max:
Frequency range:
Number of frequency points:
```

Do not assume these values until verified from the actual files.

------------------------------------------------------------------------

# 6. Visualize the geometry

Display at least 100 random samples.

Each sample should show the 16×16 structure.

Example conceptually:

``` text
0 0 0 1 1 1 0 0
0 0 1 1 1 1 1 0
0 1 1 0 0 1 1 0
1 1 0 0 0 0 1 1
...
```

Plot a grid of structures.

Check:

-   Are structures actually binary?
-   Are there repeated patterns?
-   Are they mostly empty?
-   Are they mostly filled?
-   Is there obvious symmetry?
-   Are there disconnected components?
-   Are there pathological samples?

Save:

``` text
reports/geometry_samples.png
```

------------------------------------------------------------------------

# 7. Visualize electromagnetic responses

For the same random structures, plot their response versus frequency.

For example:

``` text
frequency → x-axis
reflection → y-axis
```

Determine exactly what the response represents.

Do not assume it is `S11`; verify from the dataset documentation.

Check:

-   frequency range
-   number of frequency points
-   response channels
-   magnitude/phase vs real/imaginary representation
-   normalization
-   missing values

Save:

``` text
reports/response_samples.png
```

------------------------------------------------------------------------

# 8. Verify the geometry-response pairing

This is critical.

For one sample:

``` text
G_i
```

and:

``` text
S_i
```

must refer to the same physical structure.

Create a simple visualization:

``` text
┌──────────────┐
│ 16×16 design │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ EM response  │
└──────────────┘
```

Pick several samples and confirm that indexing is consistent.

------------------------------------------------------------------------

# 9. Data cleaning

Check for:

``` text
NaN
Inf
missing files
duplicate structures
invalid response values
unexpected geometry values
```

Do not automatically delete unusual samples.

First record:

``` text
number of invalid samples
number of duplicate samples
number of unique geometries
```

Only remove samples after understanding why they are invalid.

------------------------------------------------------------------------

# 10. Train/validation/test split

Use a fixed split.

Recommended initial split:

``` text
Train:      80%
Validation: 10%
Test:       10%
```

For 5,000 samples:

``` text
Train:      4000
Validation: 500
Test:       500
```

Save sample IDs:

``` text
data/splits/train.txt
data/splits/val.txt
data/splits/test.txt
```

Use a fixed random seed.

Example:

``` text
seed = 42
```

Do not regenerate splits every experiment.

------------------------------------------------------------------------

# 11. Important leakage check

Because our future task involves masking structures, **do not create
augmented/masked versions before splitting the original complete
structures**.

Correct:

``` text
Complete dataset
      ↓
train / val / test
      ↓
generate masks independently
```

Incorrect:

``` text
Complete structure
 ↓
100 masked versions
 ↓
random split
```

The incorrect method can put different masks of the same original
structure into both training and testing.

That creates leakage.

------------------------------------------------------------------------

# 12. Create the PyTorch Dataset

The first dataset should return:

``` python
geometry, response
```

Conceptually:

``` python
G.shape = [1, 16, 16]
S.shape = [number_of_response_values]
```

The exact dimensions depend on the actual SUTD-PRCM file format and must
be verified.

Example interface:

``` python
dataset = SUTDPRCMDataset(
    root="data/raw/SUTD_PRCM",
    split="train"
)

G, S = dataset[0]
```

------------------------------------------------------------------------

# 13. Data normalization

### Geometry

If geometry is binary:

``` text
0 → 0
1 → 1
```

Do not unnecessarily normalize it.

### Electromagnetic response

Determine the response representation first.

If it is continuous, calculate training-set statistics:

``` text
mean
std
min
max
```

Normalize using **training-set statistics only**.

Do not calculate normalization statistics using validation/test samples.

------------------------------------------------------------------------

# 14. DataLoader

Create:

``` python
train_loader
val_loader
test_loader
```

Start with a conservative batch size:

``` text
32
```

Then test:

``` text
64
128
```

depending on available RAM/VRAM.

Because the geometry is only 16×16, memory requirements should be
modest.

------------------------------------------------------------------------

# 15. First automated sanity tests

Before training anything, the following must pass:

### Test 1

``` text
len(dataset) == expected number of samples
```

### Test 2

``` text
geometry.shape == expected shape
```

### Test 3

``` text
response.shape == expected shape
```

### Test 4

``` text
geometry contains only valid values
```

### Test 5

``` text
response contains no NaN/Inf
```

### Test 6

Two accesses to the same index return the same sample.

### Test 7

Train/validation/test IDs do not overlap.

### Test 8

The DataLoader produces the expected batch shapes.

------------------------------------------------------------------------

# 16. Create the 5k processed subset

After understanding the raw dataset:

``` text
raw dataset
    ↓
fixed seed
    ↓
5,000 complete structures
    ↓
processed dataset
```

Save it separately:

``` text
data/processed/sutd_prcm_5k/
```

Do not modify the original downloaded data.

------------------------------------------------------------------------

# 17. Phase 1 success criteria

Phase 1 is complete only when all of these are true:

-   [ ] Dataset downloaded.
-   [ ] Dataset documentation read.
-   [ ] Geometry format understood.
-   [ ] EM response format understood.
-   [ ] Frequency points identified.
-   [ ] Geometry samples visualized.
-   [ ] Response samples visualized.
-   [ ] Geometry/response pairing verified.
-   [ ] Invalid samples checked.
-   [ ] Duplicate samples checked.
-   [ ] 80/10/10 split created.
-   [ ] Split leakage checked.
-   [ ] PyTorch Dataset implemented.
-   [ ] DataLoader implemented.
-   [ ] Normalization implemented.
-   [ ] Dataset unit tests pass.
-   [ ] 5,000-sample subset created.
-   [ ] Reproducibility verified with fixed seed.

------------------------------------------------------------------------

# 18. What we should NOT do in Phase 1

Do not implement:

``` text
JEPA
PINN
FDTD
FEM
Diffusion
GAN
Transformer
Inverse design
```

yet.

Phase 1 is purely:

\[ `\boxed{\text{Get trustworthy data into a clean ML pipeline}}`{=tex}
\]

If this phase is wrong, every later experiment is meaningless.

------------------------------------------------------------------------

# 19. Phase 2 preview

Once Phase 1 is finished, Phase 2 will be:

> **Build the forward EM surrogate.**

We will train:

\[ `\boxed{
G \rightarrow S(G)
}`{=tex} \]

using a tiny CNN.

The purpose is to establish:

1.  The dataset is learnable.
2.  A cheap neural approximation to the EM simulator exists.
3.  We can later use the predicted response as a differentiable physics
    objective.

Then Phase 3 will be:

\[ `\boxed{
\text{partial structure}
\rightarrow
\text{complete structure}
}`{=tex} \]

using a simple CNN baseline.

Only after those work will we introduce JEPA.
