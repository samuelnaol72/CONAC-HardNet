# Hard-TireNet MATLAB Pipeline

Use from the repository root:

```matlab
addpath(genpath('src-TireNet'))
data = prepareCombinedSlip('Suite','iid');
results = trainCombinedOffline('Suite','iid');
```

Available suites are `iid`, `sparse`, `ood_slip`, `ood_environment`, and
`sparse_ood`. Use `prepareCombinedSlip('Suite','all')` to export all five.
Normalization uses each suite's training rows only. The fixed, load-dependent
friction ellipse follows the preprocessing reference.

Raw CSVs, processed MAT files, manifests, and metadata are saved under
`src-TireNet/data/raw_suites/` and `src-TireNet/data/processed_suites/`.
Training results are saved under `src-TireNet/train/results/`.
