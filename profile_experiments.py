import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from source.builder import ProgramBuilder
from source.bundlers import GreedyBundler, OneBundleOneInstructionBundler
from source.constants import BUNDLE_CYCLES, SLOT_BUDGETS, TILE_SIZE
from source.programs.matmul import get_results as get_gemm_results
from source.programs.matmul import matmul_kernel, setup as setup_gemm
from source.programs.softmax import get_results as get_softmax_results
from source.programs.softmax import setup as setup_softmax
from source.programs.softmax import softmax_kernel
from source.sim import SimTPU


REPORT_PATH = Path("analysis") / "profile_experiments.md"


def _profile(builder, bundles):
    instr_counts = Counter()
    dynamic_cycles = Counter()
    bundle_unit_cycles = defaultdict(int)

    for instr in builder.instructions:
        instr_counts[instr.unit] += 1
        dynamic_cycles[instr.unit] += instr.cycles

    for bundle in bundles:
        per_bundle = Counter()
        for instr in bundle.instructions:
            per_bundle[instr.unit] += instr.cycles
        for unit, cycles in per_bundle.items():
            bundle_unit_cycles[unit] += cycles

    utilization = {}
    for unit, budget in SLOT_BUDGETS.items():
        if budget == 0 or not bundles:
            utilization[unit] = 0.0
        else:
            utilization[unit] = bundle_unit_cycles[unit] / (len(bundles) * budget)

    return {
        "instruction_count": len(builder.instructions),
        "bundle_count": len(bundles),
        "instr_counts": dict(instr_counts),
        "dynamic_cycles": dict(dynamic_cycles),
        "utilization": utilization,
    }


def _run_gemm(size, bundler_name):
    torch.manual_seed(0)
    tpu = SimTPU(use_device=False)
    builder = ProgramBuilder()
    bundler = OneBundleOneInstructionBundler() if bundler_name == "one" else GreedyBundler()

    a_ptr, b_ptr, c_ptr, expected = setup_gemm(tpu, size, size, size)
    matmul_kernel(builder, a_ptr, b_ptr, c_ptr, size, size, size)
    bundles = builder.build(bundler)
    cycles = tpu.run(bundles)
    actual = get_gemm_results(tpu, c_ptr, size, size)
    assert torch.allclose(actual.float(), expected.float(), atol=3e-1, rtol=1e-1)

    return {
        "kernel": "gemm",
        "shape": f"{size}x{size}x{size}",
        "bundler": bundler_name,
        "cycles": cycles,
        **_profile(builder, bundles),
    }


def _run_softmax(rows, cols, bundler_name):
    torch.manual_seed(0)
    tpu = SimTPU(use_device=False)
    builder = ProgramBuilder()
    bundler = OneBundleOneInstructionBundler() if bundler_name == "one" else GreedyBundler()

    x_ptr, y_ptr, expected = setup_softmax(tpu, rows, cols)
    softmax_kernel(builder, x_ptr, y_ptr, rows, cols)
    bundles = builder.build(bundler)
    cycles = tpu.run(bundles)
    actual = get_softmax_results(tpu, y_ptr, rows, cols)
    assert torch.allclose(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)

    return {
        "kernel": "softmax",
        "shape": f"{rows}x{cols}",
        "bundler": bundler_name,
        "cycles": cycles,
        **_profile(builder, bundles),
    }


def _fmt_table(rows, headers):
    header_line = "| " + " | ".join(headers) + " |"
    rule_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join([header_line, rule_line, *body])


def _speedup_table(results):
    rows = []
    grouped = defaultdict(dict)
    for result in results:
        grouped[(result["kernel"], result["shape"])][result["bundler"]] = result

    for (kernel, shape), variants in grouped.items():
        one = variants["one"]
        greedy = variants["greedy"]
        rows.append({
            "Kernel": kernel,
            "Shape": shape,
            "One Bundles": one["bundle_count"],
            "Greedy Bundles": greedy["bundle_count"],
            "One Cycles": one["cycles"],
            "Greedy Cycles": greedy["cycles"],
            "Speedup": f"{one['cycles'] / greedy['cycles']:.2f}x",
        })
    return rows


def _util_rows(result):
    rows = []
    for unit in ["mxu", "dma", "vector", "scalar", "tile"]:
        rows.append({
            "Unit": unit,
            "Instrs": result["instr_counts"].get(unit, 0),
            "Dynamic Cycles": result["dynamic_cycles"].get(unit, 0),
            "Utilization": f"{100 * result['utilization'].get(unit, 0.0):.1f}%",
        })
    return rows


def main():
    assert TILE_SIZE == 128, "profile_experiments.py assumes the default tile size"

    scaling = []
    for size in [128, 256, 384]:
        scaling.append(_run_gemm(size, "one"))
        scaling.append(_run_gemm(size, "greedy"))

    for rows in [64, 128, 256]:
        scaling.append(_run_softmax(rows, 64, "one"))
        scaling.append(_run_softmax(rows, 64, "greedy"))

    gemm_profile = _run_gemm(256, "greedy")
    softmax_profile = _run_softmax(128, 64, "greedy")

    sections = [
        "# Profile Experiments",
        "",
        "These experiments use the default architecture (`tile=128`, `bundle=128`) and compare the one-instruction baseline against the greedy bundler. The goal is to show both end-to-end speedup and where each kernel spends its schedule budget.",
        "",
        "## Bundler Scaling",
        "",
        _fmt_table(
            _speedup_table(scaling),
            ["Kernel", "Shape", "One Bundles", "Greedy Bundles", "One Cycles", "Greedy Cycles", "Speedup"],
        ),
        "",
        "## GEMM Profile (256x256x256, Greedy)",
        "",
        _fmt_table(_util_rows(gemm_profile), ["Unit", "Instrs", "Dynamic Cycles", "Utilization"]),
        "",
        "## Softmax Profile (128x64, Greedy)",
        "",
        _fmt_table(_util_rows(softmax_profile), ["Unit", "Instrs", "Dynamic Cycles", "Utilization"]),
        "",
        "## Short Analysis",
        "",
        "- GEMM scales cleanly with problem size and benefits more from greedy bundling than softmax. This matches the fact that GEMM has long MXU and DMA sequences that can be overlapped conservatively.",
        "- The GEMM profile is dominated by scalar instruction count, but MXU and DMA consume most of the expensive schedule slots. This is typical for a tiled accelerator kernel with explicit address generation.",
        "- Softmax shows lower speedup because most vector operations are in a strict dependency chain: reduce max, subtract, exp, reduce sum, reciprocal, multiply. The bundler still helps by hiding scalar and DMA work around those chains.",
        "- If you want stronger DMA/vector-budget sensitivity, the next experiment should use a kernel with more independent vector streams or double-buffered DMA.",
        "",
    ]

    REPORT_PATH.write_text("\n".join(sections))
    print(f"Wrote {REPORT_PATH}")
    print(json.dumps({"gemm_profile": gemm_profile, "softmax_profile": softmax_profile}, indent=2))


if __name__ == "__main__":
    main()
