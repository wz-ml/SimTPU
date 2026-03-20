from collections import Counter

from source.constants import TILE_ELEMS, TILE_SIZE, SLOT_BUDGETS
from source.sim import Bundler, Bundle, Instr

# Bundlers pack instructions into bundles (where within each bundle, 
# instructions are run concurrently).

# Must resolve hazards such that there are no dependencies within a bundle.

class OneBundleOneInstructionBundler(Bundler):
    # minimal bundler for correctness
    def __init__(self):
        super().__init__()
    def __call__(self, instructions: list[Instr]) -> list[Bundle]:
        return [Bundle(instructions=[inst]) for inst in instructions]

# always packs adjacent instructions into a bundle, checking for hazards
class GreedyBundler(Bundler):
    def __init__(self):
        super().__init__()

    def __call__(self, instructions: list[Instr]) -> list[Bundle]:
        bundles = []
        current = []
        current_meta = []
        reg_state = {0: 0}
        vector_length = TILE_SIZE

        for instr in instructions:
            meta = _analyze_instr(instr, reg_state, vector_length)
            if current and not _can_pack(current_meta, meta):
                bundles.append(Bundle(instructions=current))
                reg_state, vector_length = _advance_state(current, reg_state, vector_length)
                current = []
                current_meta = []
                meta = _analyze_instr(instr, reg_state, vector_length)
            current.append(instr)
            current_meta.append(meta)

        if current:
            bundles.append(Bundle(instructions=current))

        return bundles


def _read_set(*regs):
    return {reg for reg in regs if reg is not None}


def _range(start, size):
    if start is None or size is None:
        return None
    return (start, start + size)


def _vector_length_range(reg_state, vector_length):
    return vector_length if vector_length is not None else None


def _lookup(reg_state, reg):
    return reg_state.get(reg)


def _analyze_instr(instr, reg_state, vector_length):
    reads = set()
    writes = set()
    specials_read = set()
    specials_write = set()
    mem = []

    def scratch_read(addr, size):
        mem.append(("scratch", "r", _range(addr, size)))

    def scratch_write(addr, size):
        mem.append(("scratch", "w", _range(addr, size)))

    def hbm_read(addr, size):
        mem.append(("hbm", "r", _range(addr, size)))

    def hbm_write(addr, size):
        mem.append(("hbm", "w", _range(addr, size)))

    args = instr.args

    if instr.unit == "scalar":
        if instr.op == "load_imm":
            writes.add(args[0])
        elif instr.op == "load":
            writes.add(args[0])
            reads.add(args[1])
            scratch_read(_lookup(reg_state, args[1]), 1)
        elif instr.op == "store":
            reads |= _read_set(args[0], args[1])
            scratch_write(_lookup(reg_state, args[0]), 1)
        elif instr.op == "logical_not":
            writes.add(args[0])
            reads.add(args[1])
        else:
            writes.add(args[0])
            reads |= _read_set(args[1], args[2])

    elif instr.unit == "dma":
        reads |= _read_set(*args)
        src = _lookup(reg_state, args[0])
        dst = _lookup(reg_state, args[1])
        size_reg = _lookup(reg_state, args[2])
        size = min(size_reg, TILE_ELEMS) if size_reg is not None else None
        if instr.op == "load":
            hbm_read(src, size)
            scratch_write(dst, size)
        else:
            scratch_read(src, size)
            hbm_write(dst, size)

    elif instr.unit == "mxu":
        reads |= _read_set(*args)
        addr = _lookup(reg_state, args[0])
        if instr.op == "load_weights":
            scratch_read(addr, TILE_ELEMS)
            specials_write.add("mxu_weights")
        elif instr.op == "matmul":
            scratch_read(addr, TILE_ELEMS)
            specials_read |= {"mxu_weights", "mxu_acc"}
            specials_write.add("mxu_acc")
        else:
            scratch_write(addr, TILE_ELEMS)
            specials_read.add("mxu_acc")

    elif instr.unit == "tile":
        reads |= _read_set(*args)
        addr = _lookup(reg_state, args[0])
        scratch_read(addr, TILE_ELEMS)
        scratch_write(addr, TILE_ELEMS)

    elif instr.unit == "vector":
        if instr.op == "set_length":
            reads.add(args[0])
            specials_write.add("vl")
        else:
            specials_read.add("vl")
            width = _vector_length_range(reg_state, vector_length)
            if instr.op in {"reduce_sum", "reduce_max", "reduce_or", "reduce_and"}:
                reads |= _read_set(args[0], args[1])
                scratch_write(_lookup(reg_state, args[0]), 1)
                scratch_read(_lookup(reg_state, args[1]), width)
            elif instr.op in {"exp", "log", "sqrt", "logical_not", "reciprocal", "neg", "abs", "copy", "vbroadcast"}:
                reads |= _read_set(args[0], args[1])
                scratch_write(_lookup(reg_state, args[0]), width)
                src_width = 1 if instr.op == "vbroadcast" else width
                scratch_read(_lookup(reg_state, args[1]), src_width)
            elif instr.op == "where":
                reads |= _read_set(*args)
                scratch_write(_lookup(reg_state, args[0]), width)
                scratch_read(_lookup(reg_state, args[1]), width)
                scratch_read(_lookup(reg_state, args[2]), width)
                scratch_read(_lookup(reg_state, args[3]), width)
            else:
                reads |= _read_set(*args)
                out_addr = _lookup(reg_state, args[0])
                lhs_addr = _lookup(reg_state, args[1])
                rhs_addr = _lookup(reg_state, args[2])
                scratch_write(out_addr, 1 if instr.op == "dot" else width)
                scratch_read(lhs_addr, width)
                scratch_read(rhs_addr, width)

    return {
        "instr": instr,
        "reads": reads,
        "writes": writes,
        "specials_read": specials_read,
        "specials_write": specials_write,
        "mem": mem,
    }


def _can_pack(current_meta, meta):
    unit_cycles = Counter(item["instr"].unit for item in current_meta)
    used_cycles = Counter()
    for item in current_meta:
        used_cycles[item["instr"].unit] += item["instr"].cycles
    if used_cycles[meta["instr"].unit] + meta["instr"].cycles > SLOT_BUDGETS[meta["instr"].unit]:
        return False

    current_reads = set().union(*(item["reads"] for item in current_meta)) if current_meta else set()
    current_writes = set().union(*(item["writes"] for item in current_meta)) if current_meta else set()
    current_special_reads = set().union(*(item["specials_read"] for item in current_meta)) if current_meta else set()
    current_special_writes = set().union(*(item["specials_write"] for item in current_meta)) if current_meta else set()

    if meta["reads"] & current_writes:
        return False
    if meta["writes"] & (current_reads | current_writes):
        return False
    if meta["specials_read"] & current_special_writes:
        return False
    if meta["specials_write"] & (current_special_reads | current_special_writes):
        return False

    for existing in current_meta:
        for lhs in existing["mem"]:
            for rhs in meta["mem"]:
                if _memory_conflict(lhs, rhs):
                    return False

    return True


def _memory_conflict(lhs, rhs):
    lhs_space, lhs_mode, lhs_range = lhs
    rhs_space, rhs_mode, rhs_range = rhs
    if lhs_space != rhs_space:
        return False
    if lhs_range is None or rhs_range is None:
        return True
    if lhs_range[1] <= rhs_range[0] or rhs_range[1] <= lhs_range[0]:
        return False
    return "w" in {lhs_mode, rhs_mode}


def _advance_state(bundle, reg_state, vector_length):
    reg_state = dict(reg_state)
    for instr in bundle:
        if instr.unit == "scalar":
            if instr.op == "load_imm":
                reg_state[instr.args[0]] = instr.args[1]
            elif instr.op != "store":
                reg_state[instr.args[0]] = None
        elif instr.unit == "vector" and instr.op == "set_length":
            width = reg_state.get(instr.args[0])
            vector_length = min(width, TILE_SIZE) if width is not None else None
    reg_state[0] = 0
    return reg_state, vector_length
