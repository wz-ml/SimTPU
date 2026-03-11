# SimTPU: A small, all-python VLIW TPU simulator.
### (Toilet paper unit)

## ISA:

```
registers:
r0 - r15: scalar int32 registers
v0 - v7: fp vector registers, width 128, bfloat16

scratchpad: 128 KB, instruction-controlled (byte addressable), bfloat16
```

```
matmul:
mfma.load_weights r0, r1 # load scratchpad[s0:s0+2**14] into the systolic arr's weight registers. This loads 
mfma.matmul r0, 

```