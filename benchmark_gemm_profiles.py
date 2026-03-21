from pathlib import Path

import matplotlib.pyplot as plt
import torch

from source.builder import ProgramBuilder
from source.bundlers import GreedyBundler, OneBundleOneInstructionBundler
from source.programs.matmul import (
    get_results,
    matmul_kernel,
    matmul_kernel_all_regs,
    matmul_kernel_pipelined_128regs,
    setup,
)
from source.sim import SimTPU


OUT_DIR = Path("analysis") / "gemm_profiles"


def run_case(name, kernel_fn, bundler, shape):
    torch.manual_seed(0)
    tpu = SimTPU(use_device=False)
    builder = ProgramBuilder()
    M, N, K = shape
    a_ptr, b_ptr, c_ptr, expected = setup(tpu, M, N, K)
    kernel_fn(builder, a_ptr, b_ptr, c_ptr, M, N, K)
    bundles = builder.build(bundler)
    cycles = tpu.run(bundles)
    actual = get_results(tpu, c_ptr, M, N)
    assert torch.allclose(actual.float(), expected.float(), atol=3e-1, rtol=1e-1)

    plot_path = OUT_DIR / f"{name}_{type(bundler).__name__}.png"
    tpu.profiler.plot_utilization_distribution(
        title=f"{name} ({type(bundler).__name__})",
        filename=str(plot_path),
    )

    return {
        "name": name,
        "bundler": type(bundler).__name__,
        "instructions": len(builder.instructions),
        "bundles": len(bundles),
        "cycles": cycles,
        "plot": plot_path,
    }


def write_report(results, shape):
    rows = [
        "| Kernel | Bundler | Instructions | Bundles | Cycles | Plot |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        rows.append(
            f"| {result['name']} | {result['bundler']} | {result['instructions']} | "
            f"{result['bundles']} | {result['cycles']} | `{result['plot']}` |"
        )

    report = [
        "# GEMM Profile Benchmark",
        "",
        f"Shape: `{shape[0]} x {shape[1]} x {shape[2]}`",
        "",
        *rows,
        "",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(report))


def write_performance_plot(results):
    labels = [f"{r['name']}\n{r['bundler'].replace('Bundler', '')}" for r in results]
    cycles = [r["cycles"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, cycles, color=[
        "#6c8ebf" if "OneBundle" in r["bundler"] else "#93c47d"
        for r in results
    ])
    ax.set_ylabel("Cycles")
    ax.set_title("GEMM Performance Comparison")
    ax.tick_params(axis="x", labelrotation=20)

    for bar, value in zip(bars, cycles):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "performance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shape = (256, 256, 256)
    bundlers = [OneBundleOneInstructionBundler(), GreedyBundler()]
    kernels = [
        ("baseline", matmul_kernel),
        ("all_regs", matmul_kernel_all_regs),
        ("pipe_128", matmul_kernel_pipelined_128regs),
    ]

    results = []
    for bundler in bundlers:
        for name, kernel_fn in kernels:
            results.append(run_case(name, kernel_fn, bundler, shape))

    write_report(results, shape)
    write_performance_plot(results)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
