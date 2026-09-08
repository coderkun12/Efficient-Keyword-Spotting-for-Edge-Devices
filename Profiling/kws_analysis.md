# Keyword-Spotting CNN Layer Analysis

**Model:** `KeywordSpottingCNN` (56,844 parameters, 12 classes)  
**Input:** `[1, 1, 40, 101]` (batch, channel, n_mels, time frames) -- one 1-second clip as a 40-bin log-mel spectrogram

## Top 5 Layers by MAC Count

| Rank | Layer Name | MACs | FLOPs (2xMACs) | Parameters | % of Model MACs |
| ---- | ---------- | ---- | -------------- | ---------- | --------------- |
| 1 | `Conv2d (3): 2-4` | 74,723,840 | 149,447,680 | 18,496 | 66.2% |
| 2 | `Conv2d (7): 2-8` | 36,928,000 | 73,856,000 | 36,928 | 32.7% |
| 3 | `Conv2d (0): 2-1` | 1,292,800 | 2,585,600 | 320 | 1.1% |
| 4 | `Linear (classifier): 1-3` | 780 | 1,560 | 780 | 0.0% |
| 5 | `BatchNorm2d (4): 2-5` | 128 | 256 | 128 | 0.0% |

Total MACs across all layers: **112,945,740** (112.95 M) per inference.

---

## Arithmetic Intensity -- Most MAC-Intensive Layer

**Layer:** `Conv2d (3): 2-4`

### Assumptions
- All weights and activations are loaded from DRAM with **no reuse**.
- Data type: **float32** (4 bytes per element).

### Memory Traffic

| Tensor | Shape | Elements | Bytes (x4) |
| ------ | ----- | -------- | ---------- |
| Input activations  | `[1, 32, 40, 101]`  | 129,280  | 517,120  |
| Output activations | `[1, 64, 40, 101]` | 258,560 | 1,034,240 |
| Weights            | --                         | 18,496 | 73,984 |
| **Total**          |                            |                  | **1,625,344** |

### Calculation

```
FLOPs  = 2 x MACs = 2 x 74,723,840 = 149,447,680

DRAM bytes = input + output + weights
           = 517,120 + 1,034,240 + 73,984
           = 1,625,344 bytes

Arithmetic Intensity = FLOPs / DRAM bytes
                     = 149,447,680 / 1,625,344
                     = 91.95 FLOPs/byte
```
