from source.sim import Bundler, Bundle, Instr

# Bundlers pack instructions into bundles (where within each bundle, 
# instructions are run concurrently).

# Must resolve hazards such that there are no dependencies within a bundle.

class OneBundleOneInstructionBundler(Bundler):
    # minimal bundler for correctness
    def __init__(self):
        super().__init__()
    def __call__(self, instructions: list[Instr]) -> list[Bundle]:
        return [Bundle(instructions=inst) for inst in instructions]

# always packs adjacent instructions into a bundle, checking for hazards
class GreedyBundler(Bundler):
    def __init__(self):
        super().__init__()
    def __call__(self, instructions: list[Instr]) -> list[Bundle]:
        pass # TODO