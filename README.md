# SimTPU: A small, all-python VLIW TPU simulator.
### (Toilet paper unit)

## ISA:

```
registers:
r0 - r15: scalar int32 registers
r0 always holds 0.

scratchpad: 256 KB, instruction-controlled (byte addressable), bfloat16
```

```
MXU:

mfma.load_weights r1 # load scratchpad[r1:r1+2**14] into the systolic arr's weight registers. This loads a 128x128 array in row-major order.
# Takes 128 cycles.

mfma.matmul r2, r3 # stream activation tile from scratchpad[r2:r2+2**14] into systolic array and performs matmul.
# r3 is a boolean for if the matmul accumulates onto previously computed results or overwrites them (1 for acc, 0 for overwrite).
# This also takes 128 cycles.

mfma.store r4 # drain accumulated results to scratchpad[r3:r3+2**14]. Also takes 128 cycles.
```

```
Elementwise unit:

ewu.transpose r1 # load scratchpad[r1:r1+2**14] and tranposes it. 128 cycles.
```

```
Memory:

For dma instructions, the register order is `from`, `to`, `size`.
dma.load r1, r2, r3 # loads mem[r1:r1+min(2**14, r3)] into scratchpad[r2:r2+min(2**14, r3)]. Takes 64 cycles.
dma.store r1, r2, r3 # stores scratchpad[r1:r1+min(2**14, r3)] into mem[r2:r2+min(2**14, r3)]. Takes 64 cycles.
```

```
Scalar:

s.load r1, r2 # loads scalar from scratchpad[r2] into r1. 1 cycle.
s.store r1, r2 # stores scalar in r2 into scratchpad[r1]. 1 cycle.
s.load_imm r1, val # loads immediate value
s.mult r1, r2, r3 # r1 = r2 * r3. 1 cycle.

# Arithmetic comparison ops
s.gre r1, r2, r3 # r1 = (r2 >= r3)
s.g
s.eq
s.l
s.le

s.div s1, s2, s3 # r1 = s2 // s3
s.add s1, s2, s3 # r1 = r2 + r3
s.mod s1, s2, s3 # r1 = r2 % r3

# Branch instructions
s.beq r1, label # the label here is the bundle name to jump to. Read the bundling section for more.
s.bge r1, label
s.bg r1, label
s.bl r1, label
b.ble r1, label
b.bneq r1, label

s.jmp label # unconditional jump
```

```
Vector:

r1 = scratchpad address destination, r2 = source op scratchpad addr 1, r3 = source op scratchpad addr 2
r4 = vector length (max 128)

v.mult r1, r2, r3, r4 # hadamard multiplication
v.div r1, r2, r3, r4
v.add r1, r2, r3, r4
v.mod r1, r2, r3, r4

v.gre r1, r2, r3, r4 # comparison operators produce binary mask
v.g r1, r2, r3, r4
v.eq r1, r2, r3, r4
v.l r1, r2, r3, r4
v.le r1, r2, r3, r4

# Reduction operations store their scalar results in the first element of the destination vector
v.dot r1, r2, r3, r4 # scratchpad[r1] = dot(scratchpad[r2:r2+r4], scratchpad[r3:r3+r4]

# Unary reduction operations
v.reduce_sum r1, r2 # scratchpad[r1] = reduce(scratchpad[r1+r2]) # Don't forget that r2 is, at max, 128
v.reduce_max r1, r2
v.reduce_or r1, r2 # on boolean masks
v.reduce_and r1, r2
```

## Bundling

Each bundle of instructions 
