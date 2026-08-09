# Phase 2.5 evaluation configuration

The three controlled experiments use the same evaluator and the following
resonance definition. These values are fixed before test evaluation and are not
tuned per experiment:

- frequency range: 2.00-12.00 GHz, 0.01 GHz spacing
- feature detector: `scipy.signal.find_peaks` on magnitude and negative magnitude
- absolute prominence: `0.03`
- minimum feature spacing: `10` samples (`0.10` GHz)
- resonance window: `+-0.10` GHz around every true feature
- matching: nearest predicted extremum of the same peak/dip kind
- frequency-error summary: conditional on a matched predicted feature

The local resonance MAE is computed over the union of the windows around true
features. Samples with no qualifying true feature are retained in per-sample
outputs and excluded from finite resonance-error summaries.
