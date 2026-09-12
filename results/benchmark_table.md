| Model Variant | Size | Accuracy | CPU Latency | Params | Sparsity | MACs |
|---|---|---|---|---|---|---|
| baseline_best | 959.9 KB | 91.88% | 17.072 ms | 242,508 | 0.0% | 186,222,336 |
| pruned_structured_30-30-30-30 | 101.6 KB | 79.19% | 2.526 ms | 22,077 | 0.0% | 17,019,456 |
| pruned_structured_70-70-70-70 | 486.4 KB | 92.82% | 8.051 ms | 120,291 | 0.0% | 91,472,400 |
| pruned_unstructured_50 | 960.6 KB | 93.99% | 12.585 ms | 242,508 | 49.8% | 186,222,336 |
| quantized_ptq_fbgemm__from_baseline_best | 287.0 KB | 91.61% | 6.102 ms | — | — | — |
| quantized_ptq_fbgemm__from_pruned_structured_70-70-70-70 | 168.4 KB | 92.35% | 4.585 ms | — | — | — |
| quantized_qat_fbgemm__from_baseline_best | 287.0 KB | 93.49% | 5.600 ms | — | — | — |
| quantized_qat_fbgemm__from_pruned_structured_70-70-70-70 | 168.4 KB | 93.41% | 2.903 ms | — | — | — |
