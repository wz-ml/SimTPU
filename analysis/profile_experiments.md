# Profile Experiments

These experiments use the default architecture (`tile=128`, `bundle=128`) and compare the one-instruction baseline against the greedy bundler. The goal is to show both end-to-end speedup and where each kernel spends its schedule budget.

## Bundler Scaling

| Kernel | Shape | One Bundles | Greedy Bundles | One Cycles | Greedy Cycles | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| gemm | 128x128x128 | 1543 | 774 | 197504 | 99072 | 1.99x |
| gemm | 256x256x256 | 10288 | 5160 | 1316864 | 660480 | 1.99x |
| gemm | 384x384x384 | 32409 | 16254 | 4148352 | 2080512 | 1.99x |
| softmax | 64x64 | 2178 | 1281 | 278784 | 163968 | 1.70x |
| softmax | 128x64 | 4354 | 2561 | 557312 | 327808 | 1.70x |
| softmax | 256x64 | 8706 | 5121 | 1114368 | 655488 | 1.70x |

## GEMM Profile (256x256x256, Greedy)

| Unit | Instrs | Dynamic Cycles | Utilization |
| --- | --- | --- | --- |
| mxu | 20 | 2560 | 0.4% |
| dma | 2560 | 163840 | 24.8% |
| vector | 0 | 0 | 0.0% |
| scalar | 7708 | 7708 | 1.2% |
| tile | 0 | 0 | 0.0% |

## Softmax Profile (128x64, Greedy)

| Unit | Instrs | Dynamic Cycles | Utilization |
| --- | --- | --- | --- |
| mxu | 0 | 0 | 0.0% |
| dma | 256 | 16384 | 5.0% |
| vector | 1025 | 3712 | 1.1% |
| scalar | 3073 | 3073 | 0.9% |
| tile | 0 | 0 | 0.0% |

## Short Analysis

- GEMM scales cleanly with problem size and benefits more from greedy bundling than softmax. This matches the fact that GEMM has long MXU and DMA sequences that can be overlapped conservatively.
- The GEMM profile is dominated by scalar instruction count, but MXU and DMA consume most of the expensive schedule slots. This is typical for a tiled accelerator kernel with explicit address generation.
- Softmax shows lower speedup because most vector operations are in a strict dependency chain: reduce max, subtract, exp, reduce sum, reciprocal, multiply. The bundler still helps by hiding scalar and DMA work around those chains.
- If you want stronger DMA/vector-budget sensitivity, the next experiment should use a kernel with more independent vector streams or double-buffered DMA.
