## Benchmark results

Validation set

| Model | Accuracy | Mean n-gram overlap |
|-------|----------|---------------------|
| Baseline | 46.24% | 0.5093 |
| **20% overlap** |
| Chunk Size 100 / Overlap 20 | 33.08% | 0.3881 |
| Chunk Size 200 / Overlap 40 | 47.37% | 0.4970 |
| Chunk Size 400 / Overlap 80 | 54.89% | 0.5516 |
| Chunk Size 800 / Overlap 160 | 62.41% | 0.6137 |
| **40% overlap** |
| Chunk Size 100 / Overlap 40 | 34.59% | 0.4024 |
| Chunk Size 200 / Overlap 80 | 47.37% | 0.5027 |
| Chunk Size 400 / Overlap 160 | 56.39% | 0.5665 |
| Chunk Size 800 / Overlap 320 | 63.91% | 0.6279 |
| **60% overlap** |
| Chunk Size 100 / Overlap 60 | 34.59% | 0.4140 |
| Chunk Size 200 / Overlap 120 | 50.38% | 0.5255 |
| Chunk Size 400 / Overlap 240 | 53.76% | 0.5514 |
| Chunk Size 800 / Overlap 480 | 55.26% | 0.5686 |
| **80% overlap** |
| Chunk Size 100 / Overlap 80 | 30.83% | 0.3734 |
| Chunk Size 200 / Overlap 160 | 45.11% | 0.4819 |
| Chunk Size 400 / Overlap 320 | 57.14% | 0.5738 |
| Chunk Size 800 / Overlap 640 | 56.02% | 0.5575 |
| **Focus search** |
| Chunk Size 500 / Overlap 200 | 56.39% | 0.5751 |
| Chunk Size 600 / Overlap 240 | 57.52% | 0.5838 |
| Chunk Size 500 / Overlap 250 | 57.52% | 0.5753 |
| Chunk Size 600 / Overlap 300 | 60.53% | 0.6074 |
| Chunk Size 500 / Overlap 300 | 57.14% | 0.5797 |
| Chunk Size 600 / Overlap 360 | 62.41% | 0.6216 |
Test set

| Model | Accuracy | Mean n-gram overlap |
|-------|----------|---------------------|
| Baseline | 0.46 % | 0.4706 |

**Baseline**
The baseline. 
