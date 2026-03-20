# Architecture Sweep

This report compares a simple one-instruction bundler against a conservative greedy bundler and then sweeps a few architecture knobs around the greedy baseline.

## Bundler

| Case | Kernel | Bundler | Tile | Bundle | DMA | Vector | Bundles | Cycles | Speedup vs one |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one_baseline | gemm | one | 128 | 128 | 128 | 128 | 10288 | 1316864 | 1.00x |
| one_baseline | softmax | one | 128 | 128 | 128 | 128 | 4354 | 557312 | 1.00x |
| greedy_baseline | gemm | greedy | 128 | 128 | 128 | 128 | 5160 | 660480 | 1.99x |
| greedy_baseline | softmax | greedy | 128 | 128 | 128 | 128 | 2561 | 327808 | 1.70x |

## Tile Size

| Case | Kernel | Bundler | Tile | Bundle | DMA | Vector | Bundles | Cycles | Speedup vs one |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tile_64 | gemm | greedy | 64 | 128 | 128 | 128 | 18720 | 2396160 | 0.55x |
| tile_64 | softmax | greedy | 64 | 128 | 128 | 128 | 2561 | 327808 | 1.70x |
| tile_128 | gemm | greedy | 128 | 128 | 128 | 128 | 5160 | 660480 | 1.99x |
| tile_128 | softmax | greedy | 128 | 128 | 128 | 128 | 2561 | 327808 | 1.70x |

## Dma Budget

| Case | Kernel | Bundler | Tile | Bundle | DMA | Vector | Bundles | Cycles | Speedup vs one |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dma_64 | gemm | greedy | 128 | 128 | 64 | 128 | 5160 | 660480 | 1.99x |
| dma_64 | softmax | greedy | 128 | 128 | 64 | 128 | 2561 | 327808 | 1.70x |
| dma_128 | gemm | greedy | 128 | 128 | 128 | 128 | 5160 | 660480 | 1.99x |
| dma_128 | softmax | greedy | 128 | 128 | 128 | 128 | 2561 | 327808 | 1.70x |
| dma_256 | gemm | greedy | 128 | 128 | 256 | 128 | 5160 | 660480 | 1.99x |
| dma_256 | softmax | greedy | 128 | 128 | 256 | 128 | 2561 | 327808 | 1.70x |

## Vector Budget

| Case | Kernel | Bundler | Tile | Bundle | DMA | Vector | Bundles | Cycles | Speedup vs one |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vector_64 | gemm | greedy | 128 | 128 | 128 | 64 | 5160 | 660480 | 1.99x |
| vector_64 | softmax | greedy | 128 | 128 | 128 | 64 | 2561 | 327808 | 1.70x |
| vector_128 | gemm | greedy | 128 | 128 | 128 | 128 | 5160 | 660480 | 1.99x |
| vector_128 | softmax | greedy | 128 | 128 | 128 | 128 | 2561 | 327808 | 1.70x |
| vector_256 | gemm | greedy | 128 | 128 | 128 | 256 | 5160 | 660480 | 1.99x |
| vector_256 | softmax | greedy | 128 | 128 | 128 | 256 | 2561 | 327808 | 1.70x |

## Bundle Window

| Case | Kernel | Bundler | Tile | Bundle | DMA | Vector | Bundles | Cycles | Speedup vs one |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bundle_128 | gemm | greedy | 128 | 128 | 128 | 128 | 5160 | 660480 | 1.99x |
| bundle_128 | softmax | greedy | 128 | 128 | 128 | 128 | 2561 | 327808 | 1.70x |
| bundle_256 | gemm | greedy | 128 | 256 | 256 | 256 | 5160 | 1320960 | 1.00x |
| bundle_256 | softmax | greedy | 128 | 256 | 256 | 256 | 2561 | 655616 | 0.85x |

## Short Analysis

- The greedy bundler is the only change that clearly improves both kernels. It nearly halves GEMM bundle count and cuts softmax bundle count by about a third.
- Tile size matters a lot for GEMM. On the fixed `256 x 256 x 256` problem, `128 x 128` tiles are much better than `64 x 64` because the smaller tile forces more DMA traffic and more MXU launches.
- DMA and vector slot budget sweeps are flat on these kernels. That is not a bug in the sweep; it means the current programs are dominated by true dependencies and register reuse, so the bundler does not find extra independent DMA or vector work to co-issue.
- The wider bundle window also stays flat in bundle count. In this simulator, that means a larger bundle window just increases the cost per bundle unless the code also exposes more instruction-level parallelism.

## Notes

- `bundle_window` scales the full per-bundle issue window, not just the DMA/vector slots.
- GEMM uses a fixed `256 x 256 x 256` problem. Softmax uses `128 x 64`.
- Softmax is mostly dependency-chained, so vector budget changes are expected to matter less than DMA or bundling changes.
