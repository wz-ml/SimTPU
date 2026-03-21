from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from source.builder import ProgramBuilder
from source.bundlers import GreedyBundler, OneBundleOneInstructionBundler
from source.constants import BUNDLE_CYCLES, INST_TYPES, NUM_REGS
from source.programs.matmul import (
    get_results,
    matmul_kernel,
    matmul_kernel_pipelined_32regs,
    matmul_kernel_pipelined_128regs,
    setup,
)
from source.sim import SimTPU


OUT_PATH = Path("analysis") / "gemm_profiles" / "utilization_comparison.png"
SHAPE = (256, 256, 256)


def mean_utilization(profiler):
    util = {unit: 0.0 for unit in INST_TYPES}
    if not profiler.bundle_data:
        return util

    for bundle in profiler.bundle_data:
        for unit in INST_TYPES:
            util[unit] += bundle["usage"].get(unit, 0) / BUNDLE_CYCLES

    scale = len(profiler.bundle_data)
    return {unit: 100.0 * value / scale for unit, value in util.items()}


def run_case(label, kernel_fn, bundler):
    torch.manual_seed(0)
    tpu = SimTPU(use_device=False)
    builder = ProgramBuilder()
    m, n, k = SHAPE
    a_ptr, b_ptr, c_ptr, expected = setup(tpu, m, n, k)
    kernel_fn(builder, a_ptr, b_ptr, c_ptr, m, n, k)
    bundles = builder.build(bundler)
    tpu.run(bundles)
    actual = get_results(tpu, c_ptr, m, n)
    assert torch.allclose(actual.float(), expected.float(), atol=3e-1, rtol=1e-1)
    return {
        "label": label,
        "util": mean_utilization(tpu.profiler),
    }


def plot(results):
    units = ["mxu", "dma", "scalar", "vector", "tile"]
    x = np.arange(len(units))
    width = 0.18
    colors = ["#4c78a8", "#f58518", "#54a24b", "#72b7b2", "#e45756"]
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, result in enumerate(results):
        offsets = x + (idx - 2) * width
        heights = [result["util"][unit] for unit in units]
        ax.bar(offsets, heights, width=width, label=result["label"], color=colors[idx])

    ax.set_ylabel("Average Utilization Per Bundle (%)")
    ax.set_title("GEMM Functional Unit Utilization Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([unit.upper() for unit in units])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    assert NUM_REGS >= 126, f"Run with SIMTPU_NUM_REGS=128; found {NUM_REGS}"

    results = [
        run_case("baseline + no bundling", matmul_kernel, OneBundleOneInstructionBundler()),
        run_case("baseline + greedy", matmul_kernel, GreedyBundler()),
        run_case("pipe32 + no bundling", matmul_kernel_pipelined_32regs, OneBundleOneInstructionBundler()),
        run_case("pipe128 + no bundling", matmul_kernel_pipelined_128regs, OneBundleOneInstructionBundler()),
        run_case("pipe128 + greedy", matmul_kernel_pipelined_128regs, GreedyBundler()),
    ]
    plot(results)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
