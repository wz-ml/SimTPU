# SimTPU: A small, all-python VLIW TPU simulator.
### (Toilet paper unit)

## ISA:

```
registers:
r0 - r15: scalar int32 registers
r0 always holds 0.

scratchpad: 256 KB, instruction-controlled (2-byte addressable), bfloat16.
Note that all addresses, both HBM and scratchpad, are element-addressed, not byte-addressed (all dtypes assumed bf16).
```

```
MXU:

mfma.load_weights r1 # load scratchpad[r1:r1+2**14] into the systolic arr's weight registers. This loads a 128x128 array in row-major order.
# Takes 128 cycles.

mfma.matmul r2, r3 # stream activation tile from scratchpad[r2:r2+2**14] into systolic array and performs matmul.
# r3 is a boolean for if the matmul accumulates onto previously computed results or overwrites them (1 for acc, 0 for overwrite).
# This also takes 128 cycles.

mfma.store r4 # drain accumulated results to scratchpad[r4:r4+2**14]. Also takes 128 cycles.
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

# Arithmetic ops
s.mult r1, r2, r3 # r1 = r2 * r3. 1 cycle.
s.div r1, r2, r3 # r1 = r2 // r3
s.add r1, r2, r3 # r1 = r2 + r3
s.sub r1, r2, r3 # r1 = r2 - r3
s.mod r1, r2, r3 # r1 = r2 % r3

# Unary ops
s.not r1, r2 # r1 = bitwise NOT of r2

# Arithmetic comparison ops
s.gre r1, r2, r3 # r1 = (r2 >= r3)
s.gt r1, r2, r3 # r1 = (r2 > r3)
s.eq r1, r2, r3 # r1 = (r1 == r3)
s.lt r1, r2, r3 # r1 = (r2 < r3) 
s.le r1, r2, r3 # r1 = (r2 <= r3)
s.neq r1, r2, r3 # r1 = (r2 != r3)

# Branch instructions
s.beq r1, label # the label here is the bundle name to jump to. Read the bundling section for more.
s.bge r1, label
s.bgt r1, label
s.blt r1, label
s.ble r1, label
s.bneq r1, label

s.jmp label # unconditional jump
```

```
Vector:

r1 = scratchpad address destination, r2 = source op scratchpad addr 1, r3 = source op scratchpad addr 2
v.set_length r4 # sets vector length to r4 (max 128).
# Set length always executes before any vector ops in a given bundle.

# Implicitly, what this does is hold an internal register for the vector length. All following instructions will
# operate on vectors of this length.

v.mult r1, r2, r3 # hadamard multiplication (scratchpad[r1:r1+vector_length] = scratchpad[r2+r2+vector_length] + ...)
v.div r1, r2, r3
v.add r1, r2, r3
v.sub r1, r2, r3
v.mod r1, r2, r3
v.exp r1, r2, r3 # scratchpad[r1:r1+vector_length] = scratchpad[r2:r2+vector_length] ^ scratchpad[r3:r3+vector_length]

v.eq r1, r2, r3 # comparison operators produce binary mask
v.ge r1, r2, r3 
v.gt r1, r2, r3
v.lt r1, r2, r3
v.le r1, r2, r3
v.neq r1, r2, r3

# Unary elementwise ops
v.not r1, r2 # elementwise not
v.reciprocal r1, r2 # scratchpad[r1:r1+vector_length] = 1 / scratchpad[r2:r2+vector_length] # useful for softmax
v.neg r1, r2 # scratchpad[r1:r1+vector_length] = -scratchpad[r2:r2+vector_length]
v.abs r1, r2 # scratchpad[r1:r1+vector_length] = |scratchpad[r2:r2+vector_length]|
v.copy r1, r2 # scratchpad[r1:r1+vector_length] = scratchpad[r2:r2+vector_length]

# Reduction operations store their scalar results in the first element of the destination vector
v.dot r1, r2, r3 # scratchpad[r1] = dot(scratchpad[r2:r2+vector_length], scratchpad[r3:r3+vector_length]

# Unary reduction operations
v.reduce_sum r1, r2 # scratchpad[r1] = reduce(scratchpad[r2:r2+vector_length]) - only the first element of scratchpad[r1] is filled in!
v.reduce_max r1, r2
v.reduce_or r1, r2 # on boolean masks
v.reduce_and r1, r2

# Broadcast
v.vbroadcast r1, r2 # scratchpad[r1:r1+vector_length] = scratchpad[r2] # to load imm into scratchpad, s.store and then vbroadcast

# Select ops
v.where r1, r2, r3, r4
r1 = destination, r2 = mask, r3 = select if true, r4 = select if false

# Vector-scalar operations
# Semantics: r1 = vector destination, r2 = scalar operand, r3 = vector operand (if exists)
v.exp r1, r2 # exp scratchpad[r1:r1+vector_length] by r2 (scalar register).
v.max r1, r2, r3 # produces elementwise mask of r2 and scratchpad[r3:r3+vector_length], store in scratchpad[r1]
```

## Bundling

Each bundle of instructions 
