# GEMM Profile Benchmark

Shape: `256 x 256 x 256`

| Kernel | Bundler | Instructions | Bundles | Cycles | Plot |
| --- | --- | ---: | ---: | ---: | --- |
| baseline | OneBundleOneInstructionBundler | 10288 | 10288 | 1316864 | `analysis/gemm_profiles/baseline_OneBundleOneInstructionBundler.png` |
| all_regs | OneBundleOneInstructionBundler | 7958 | 7958 | 1018624 | `analysis/gemm_profiles/all_regs_OneBundleOneInstructionBundler.png` |
| pipe_128 | OneBundleOneInstructionBundler | 8348 | 8348 | 1068544 | `analysis/gemm_profiles/pipe_128_OneBundleOneInstructionBundler.png` |
| baseline | GreedyBundler | 10288 | 5160 | 660480 | `analysis/gemm_profiles/baseline_GreedyBundler.png` |
| all_regs | GreedyBundler | 7958 | 2497 | 319616 | `analysis/gemm_profiles/all_regs_GreedyBundler.png` |
| pipe_128 | GreedyBundler | 8348 | 2305 | 295040 | `analysis/gemm_profiles/pipe_128_GreedyBundler.png` |
