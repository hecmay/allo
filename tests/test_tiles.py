import allo
from allo.ir.types import int8, int16, int32, float32, index

M = K = N = 64

# Hard fix in allo.
# by default, memref is used in allo to represent. Using tensor slice requires
# tensor dialect support, which is less stable
# as a workaround, we use decorator to revert it back


@allo.as_tile(  )
def single_tile[MT: int32, NT: int32](
        A: int8[M, K], B: int8[K, N], C: int16[M, N], i: index, j: index):


    # Loading memory from L3 to L1
    A_local: int8[MT, K] = A[i*MT:(i+1)*MT, :]
    B_local: int8[K, NT] = B[:, j*NT:(j+1)*NT]

    # Compute: expanded to loops. if there are multiple ops, they are fused
    # into a single loop nest, and used to generate input for aries
    C_local: int16[MT, NT] = A_local @ B_local

    # Storing (partial) outputs from L1 to L3
    C[i*MT:(i+1)*MT, j*NT:(j+1)*NT] = C_local


# NB: fn type parameter syntax only supported with py >= 3.12
def matmul[MT: int32, NT: int32](A: int8[M, K], B: int8[K, N], C: int16[M, N]):
    dim = (M / MT, N / NT)
    for i, j in allo.grid(dim, name="PEs"):
        single_tile[MT, NT](A, B, C, i, j)


# mlir input and compilation configs to aries-opt
print(single_tile.source)
import sys; sys.exit(0)

# instantiate = [...] specifies the values for function type parameters
s0 = allo.customize(single_tile.source, instantiate=[8, 8])
s1 = allo.customize(matmul, instantiate=[8, 8])
s1.compose(s0) 

tiles = s1.unfold("PE", axis=[0, 1])
# can be used to construct inter-PE data resue for systolic array on FPGA targets
# s.to(tiles[:, :-1].A, tiles[:, 1:].A, depth = 2)
# s.to(tiles[:-1, :].B, tiles[1:, :].B, depth = 2)

# for AIE targets, automatic placement is used
s1.to(tiles, target="NPU[:4, :4]")


# Emit code for ADF optimizer and compilation configs
code, config = s1.build(target="vck")
