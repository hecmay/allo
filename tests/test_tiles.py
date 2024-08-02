import allo
from allo.ir.types import int8, int16, int32, float32, index

M = K = N = 64

# Hard fix in allo.
# by default, memref is used in allo to represent. Using tensor slice requires
# tensor dialect support, which is less stable
# as a workaround, we use decorator to revert it back


@allo.as_tile()
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


# Mlir input and compilation configs to aries-opt:
# module {
#   func.func @single_tile(%A: memref<MxKxint8>, %B: memref<KxNxint8>, %C: memref<MxNxint16>) {
#     affine.for %arg3 = 0 to MT {
#       affine.for %arg4 = 0 to NT {
#         affine.for %arg5 = 0 to K {
#         %0 = affine.load %arg1[%arg3, %arg5] : memref<MTxKxf32>
#         %1 = affine.load %arg2[%arg5, %arg4] : memref<KxNTxf32>
#         %2 = arith.mulf %0, %1 : f32
#         %3 = affine.load %arg0[%arg3, %arg4] : memref<MTxNTxf32>
#         %4 = arith.addf %3, %2 : f32
#         affine.store %4, %arg0[%arg3, %arg4] : memref<MTxNTxf32>
#         }
#       }
#     }
#   }
# }
# NB: {MT, NT} needs to be stretched to {M, N}. and be replaced with actual numbers
print(single_tile.source)

# instantiate = [...] specifies the values for function type parameters
s0 = allo.customize(single_tile.source, instantiate=[8, 8])

# schedule on single core
s0.vectorize("i", 8)

# compose the single_tile with matmul
s1 = allo.customize(matmul, instantiate=[8, 8])
s1.compose(s0)

# get tile inside the grid to manipulate the tile placement
tiles = s1.unfold("PE", axis=[0, 1])
# can be used to construct inter-PE data resue for systolic array on FPGA targets
# s.to(tiles[:, :-1].A, tiles[:, 1:].A, depth = 2)
# s.to(tiles[:-1, :].B, tiles[1:, :].B, depth = 2)

# for AIE targets, automatic placement is used
s1.to(tiles, target="NPU[:4, :4]")


# Emit code for ADF optimizer and compilation configs
code, config = s1.build(target="vck")
