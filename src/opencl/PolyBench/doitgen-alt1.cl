//
// (c) January 2, 2018 Saman Biookaghazadeh @ Arizona State University
//

#ifdef INT_PRECISION
#define DTYPE int
#elif SINGLE_PRECISION
#define DTYPE float
#elif DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define DTYPE double
#endif

#include "../TSVC/funcs.h"

channel DTYPE c0;

__kernel void doitgen1 (__global const DTYPE* restrict A,
                        __global const DTYPE* restrict C4,
												__global DTYPE* restrict sum,
                        const int lllX,
                        const int lllY) {

#ifdef GPU

#endif


#ifdef FPGA_SINGLE

	for (int i = 0; i < lllX; i++) {
		for (int j = 0; j < lllY/BLOCK_SIZE; j++) {
      DTYPE A_temp[BLOCK_SIZE];

      #pragma unroll
      for (int k = 0; k < BLOCK_SIZE; k++) {
        A_temp[k] = A[i*lllY + j*BLOCK_SIZE + k];
      }

      for (int k = 0; k < lllY; k++) {
        DTYPE C4_temp[BLOCK_SIZE];

        #pragma unroll
        for (int z = 0; z < BLOCK_SIZE; z++) {
        	C4_temp[z] = C4[k*lllY + j*BLOCK_SIZE + z];
        }

        #pragma unroll
        for (int z = 0; z < BLOCK_SIZE; z++) {
          sum[i*lllY + k] += C4_temp[z] * A_temp[k];
        }
      }

    }

    for (int j = 0; j < lllY; j++)
			write_channel_intel(c0, sum[i*lllY+j]);
  }

#endif

}


__kernel void doitgen2 (__global DTYPE* restrict AA,
                        const int lllX,
                        const int lllY) {

#ifdef GPU

#endif


#ifdef FPGA_SINGLE

	for (int i = 0; i < lllX; i++) {
    for (int j = 0; j < lllY; j++) {
      DTYPE temp = read_channel_intel(c0);

#if INTENSITY1
			megaBfunctionNoAcc (AA[i*lllY + j], temp, j);
#elif INTENSITY2
			megaBfunctionNoAcc2 (AA[i*lllY + j], temp, j);
#elif INTENSITY3
			megaBfunctionNoAcc3 (AA[i*lllY + j], temp, j);
#elif INTENSITY4
			megaBfunctionNoAcc4 (AA[i*lllY + j], temp, j);
#elif INTENSITY5
			megaBfunctionNoAcc5 (AA[i*lllY + j], temp, j);
#endif

    }
  }

#endif

}
