# GEMM Optimization Comparison

Shape: `256 x 256 x 256`

| Configuration | Instructions | Bundles | Cycles | Speedup vs Baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 10288 | 10288 | 1316864 | 1.00x |
| baseline + greedy | 10288 | 5160 | 660480 | 1.99x |
| pipe32 | 7840 | 7840 | 1003520 | 1.31x |
| pipe128 | 8332 | 8332 | 1066496 | 1.23x |
| pipe128 + greedy | 8332 | 2301 | 294528 | 4.47x |