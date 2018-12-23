//
// (c) December 20, 2018 Saman Biookaghazadeh @ Arizona State University
//

#ifdef INT_PRECISION
#define DTYPE int
#elif SINGLE_PRECISION
#define DTYPE float
#elif DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define DTYPE double
#endif

#include "funcs.h"

channel DTYPE c0;

__kernel void doitgen_k1 (__global DTYPE* restrict A,
                          __global DTYPE* retrict sum,
                          const int lllR,
                          const int lllQ,
                          const int lllP)
{

#ifdef GPU


#endif


#ifdef FPGA_SINGLE

       const DTYPE alpha = 2.5;

       for (int r = 0; r < lllR; r++) {
           for (int q = 0; q < lllQ; q++) {
               DTYPE tempA = 0.0;
               for (int s = 0; s < lllP/BLOCK_SIZE; s++) {
                   DTYPE A_local[BLOCK_SIZE];
                   #pragma unroll
                   for (int z = 0; z < BLOCK_SIZE; z++) {
                       A_local[z] = A[r*lllQ*lllP+q*lllP+s*BLOCK_SIZE+z];
                   }
                   #pragma unroll
                   for (int z = 0; z < BLOCK_SIZE; z++) {
                       tempA += A_local[z];
                   }
               }
               for (int p = 0; p < lllP; p++) {
                   write_channel_altera(c0, tempA*alpha);
               }
           }
       }

#endif

}

__kernel void doitgen_k2 (__global DTYPE* restrict A,
                          __global DTYPE* restrict sum,
                          const int lllR,
                          const int lllQ,
                          const int lllP)

{

#ifdef GPU
       
#endif

#ifdef FPGA_SINGLE
       for (int r = 0; r < lllR; r++) {
           for (int q = 0; q < lllQ; q++) {
               for (int p = 0; p < lllP/BLOCK_SIZE; p++) {
                   DTYPE A_local[BLOCK_SIZE];

                   for (int z = 0; z < BLOCK_SIZE; z++) {
                       A_local[z] = read_channel_altera(c0);
                   }

                   #pragma unroll 
                   for (int z = 0; z < BLOCK_SIZE; z++) {
                       A[r*lllQ*lllP+q*lllP+p*BLOCK_SIZE+z] = A_local[z];
                   }
               }
           }
       }
#endif

}