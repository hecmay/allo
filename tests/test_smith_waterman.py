
import os
import numpy as np
import pytest
import allo
from allo.ir.types import int4, int8, int16, int32, int64, int128, index, UInt
from allo.ir.utils import MockBuffer
from allo.utils import get_np_struct_type

def smith_waterman(seq1, seq2):
    match = 2
    mismatch = -1
    gap = -1

    matrix = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]
    
    # Fill in the score matrix
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            match_score = match if seq1[i - 1] == seq2[j - 1] else mismatch
            score = max(matrix[i-1][j-1] + match_score,
                        matrix[i-1][j] + gap,
                        matrix[i][j-1] + gap,
                        0)
            matrix[i][j] = score
    return matrix


M = N = 128
def sw_sa(seq1: int8[M], seq2: int8[N], O: int8[M, N]):

    # score matrix
    score : int8[M, N] = 0

    # inter-PE FIFOs 
    fifo_h: int8[M, N+1]
    fifo_v: int8[N, M+1]
    fifo_d: int8[M+1, N+1] 

    # I/O network loader
    for m in range(1, M):
      fifo_h[m, 0] = 0
      fifo_d[m, 0] = 0

    for n in range(1, N):
      fifo_v[0, n] = 0
      fifo_d[0, n] = 0

    fifo_h[0, 0] = 0
    fifo_v[0, 0] = 0
    fifo_d[0, 0] = 0
      
    # dataflow region - MxN PE array 
    for i, j in allo.grid(M, N):
      PE(
        fifo_h[i,j],   fifo_v[i,j],   fifo_d[i,j],
        fifo_h[i+1,j], fifo_v[i,j+1], fifo_d[i+1,j+1], 
        score, i, j
      )

    # I/O network drainer
    drain_h: int8[M]
    drain_v: int8[N]
    drain_d: int8[M+N-1]

    for m in range(M-1):
      drain_h[m] = fifo_h[m, N]
      drain_d[m] = fifo_d[m, N]
    for n in range(N-1):
      drain_v[n] = fifo_v[n, M]
      drain_d[n+M-1] = fifo_d[M, n]

    drain_h[M-1] = fifo_h[M-1,N]
    drain_v[N-1] = fifo_v[N,M-1]
    drain_d[N+M-2] = fifo_d[M, N]    


def PE(
  h_inp: int8, v_inp: int8, d_inp: int8,
  h_out: int8, v_out: int8, d_out: int8,
  score: int8[M, N], i: index, j: index
  ):

# FIXME: weight stationary value in each PE
#   a = seq1[i]
#   b = seq2[j]
  a: int8 = 0
  b: int8 = 1

  match_score: int8 = -1
  if a == b:
    match_score = 2
  
  v: int8
  v = d_inp + match_score
  if h_inp - 1 > v:
     v = h_inp -1
  
  if v_inp - 1 > v:
     v = v_inp - 1
  
  if v < 0:
     v = 0
  
  score[i,j] = v

  h_out = v
  v_out = v
  d_out = v


# allo customization passes
s = allo.customize(sw_sa)
s.partition(s.O, dim=0)  
s.partition(s.seq1, dim=1)
s.partition(s.seq2, dim=2)
pe = s.unfold("PE", [0, 1]) 

s.to(s.fifo_h, pe, axis=0, depth=1)
s.to(s.fifo_v, pe, axis=1, depth=1)
s.to(s.fifo_d, pe, axis=2, depth=1)

code = s.build("vhls")
assert "#pragma HLS dataflow" in str(code)
if os.system("which vivado_hls >> /dev/null") == 0:
    hls_mod = s.build(
        target="vivado_hls", mode="debug", project="systolic_stream.prj"
    )
    print(hls_mod)
    hls_mod()