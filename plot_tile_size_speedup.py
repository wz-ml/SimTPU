from pathlib import Path

import matplotlib.pyplot as plt


OUT_PATH = Path("analysis") / "tile_size_speedup.png"


def main():
    tile_sizes = [64, 128]
    gemm_cycles = [2396160, 660480]
    softmax_cycles = [327808, 327808]
    gemm_speedup = [gemm_cycles[0] / c for c in gemm_cycles]
    softmax_speedup = [softmax_cycles[0] / c for c in softmax_cycles]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(tile_sizes, gemm_speedup, marker="o", linewidth=2, color="#4c78a8", label="GEMM")
    ax.plot(tile_sizes, softmax_speedup, marker="o", linewidth=2, color="#f58518", label="Softmax")

    for x, y in zip(tile_sizes, gemm_speedup):
        ax.text(x, y + 0.05, f"{y:.2f}x", ha="center", color="#4c78a8")
    for x, y in zip(tile_sizes, softmax_speedup):
        ax.text(x, y - 0.12, f"{y:.2f}x", ha="center", color="#f58518")

    ax.set_title("Tile Size vs Performance")
    ax.set_xlabel("Tile Size (Systolic Array Dimension)")
    ax.set_ylabel("Speedup Relative to Tile Size 64")
    ax.set_xticks(tile_sizes)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
