from source.sim import Bundler, Bundle, Instr

class GreedyBundler(Bundler):
    def __init__(self):
        super().__init__()
    def __call__(self, instructions: list[Instr]) -> list[Bundle]:
        pass # TODO