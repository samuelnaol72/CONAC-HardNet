# CONAC-HardNet

## Hard-TireNet MATLAB pipeline

Run from the repository root:

```matlab
addpath(genpath('src-TireNet'))
data = prepareCombinedSlip('Suite','iid');
results = trainCombinedOffline('Suite','iid');
```

Available preprocessing suites are `iid`, `sparse`, `ood_slip`,
`ood_environment`, and `sparse_ood`. Use
`prepareCombinedSlip('Suite','all')` to export all five. Normalization is
fitted on each suite's training split. Raw CSVs, processed MAT files, and
metadata are saved under `src-TireNet/data/`. Training results go to
`src-TireNet/train/results/`.
