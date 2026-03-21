from __future__ import annotations
from collections import defaultdict, Counter
from typing import TYPE_CHECKING
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from source.constants import BUNDLE_CYCLES, SLOT_BUDGETS, INST_TYPES

if TYPE_CHECKING:
    from source.sim import Bundle

class Profiler:
    def __init__(self):
        self.bundle_data = []
        self.total_instructions = 0
    def update_bundle(self, bundle: Bundle):
        # Collect data on the distribution of cycles per bundle
        usage, ops = Counter(), Counter()
        for i in bundle.instructions:
            usage[i.unit] += i.cycles
            ops[i.unit] += 1
        self.bundle_data.append({
            "usage": dict(usage),
            "ops": dict(ops),
            "n_instrs": len(bundle.instructions),
        })
    def plot_utilization_distribution(self, title="", filename=None):
        units = INST_TYPES
        colors = dict(zip(units, sns.color_palette("tab10", len(units))))

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for idx, unit in enumerate(units):
            ax = axes[idx]
            vals = np.array([d["usage"].get(unit, 0) / BUNDLE_CYCLES * 100
                                for d in self.bundle_data])
            ax.hist(vals, bins=20, color=colors[unit], edgecolor='black', alpha=0.7)
            ax.set_xlabel("Utilization (%)")
            ax.set_ylabel("Count")
            ax.set_title(f"{unit.upper()} Utilization Distribution")
            ax.set_xlim(0, 100)

        for idx in range(len(units), len(axes)):
            axes[idx].axis('off')

        fig.suptitle(title or "Functional Unit Utilization Distribution", fontsize=12)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()