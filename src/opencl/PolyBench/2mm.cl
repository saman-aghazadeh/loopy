//
// (c) December 19, 2018 Saman Biookaghazadeh @ Arizona State University
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

#ifdef FPGA_NDRANGE
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((num_simd_work_items(16)))
__attribute__((num_compute_units(NUM_COMPUTE_UNITS)))
#endif

__kernel void 2mm_k1 (__global const DTYPE* restrict A,
                   __global const DTYPE* restrict B,
                   __global const DTYPE* restrict C,
                   __global DTYPE* restrict D,
                   const DTYPE alpha,
                   const DTYPE beta
#ifdef FPGA_SINGLE
                   ,const int lll)
#else
                   )
#endif
{

#ifdef GPU

#endif

#ifdef FPGA_SINGLE

       for (int i = 0; i < lll; i++) {
           for (int j = 0; j < lll; j++) {
               DTYPE temp = 0.0;
               for (int k = 0; k < lll; k++) {
                   temp += A[i][k] * B[j][k] * alpha;
               }
               write_channel_altera (c0, temp);
           }
       }

#endif

}

__kernel void 2mm_k2 (__global const DTYPE* restrict A,
                      __global const DTYPE* restrict B,
                      __global const DTYPE* restrict C,
                      __global DTYPE* restrict D,
                      const DTYPE alpha,
                      const DTYPE beta
#ifdef FPGA_SINGLE
                      ,const int lll)
#else
                      )
#endif
{

#ifdef GPU

#endif


#ifdef FPGA_SINGLE

       for (int i = 0; i < lll; i++) {
           for (int j = 0; j < lll; j++) {
               DTYPE temp = read_channel_altera(c0);
               for (int k = 0; k < lll; k++) {
                   D[i][k] += temp * C[j][k];
               }
           }
       }
#endif
  
}