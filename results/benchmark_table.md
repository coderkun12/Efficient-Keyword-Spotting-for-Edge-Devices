| Model Variant                 | Size     | Accuracy | CPU Latency | Params  | Sparsity | MACs        |
| ----------------------------- | -------- | -------- | ----------- | ------- | -------- | ----------- |
| baseline_best                 | 959.9 KB | 91.88%   | 16.711 ms   | 242,508 | 0.0%     | 186,222,336 |
| pruned_structured_30-30-30-30 | 101.6 KB | 79.19%   | 4.672 ms    | 22,077  | 0.0%     | 17,019,456  |
| pruned_structured_70-70-70-70 | 486.4 KB | 92.82%   | 14.278 ms   | 120,291 | 0.0%     | 91,472,400  |
| pruned_unstructured_50        | 960.6 KB | 93.99%   | 15.484 ms   | 242,508 | 49.8%    | 186,222,336 |

Use 70-70-70-70 model for Quanitzation.
