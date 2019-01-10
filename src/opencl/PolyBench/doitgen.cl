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
    for (int j = 0; j < lllY; j++) {
      ddsum[i][j] = 0;
      DTYPE temp = 0;
      for (int k = 0; k < lllY/BLOCK_SIZE; k++) {
        DTYPE A_temp[BLOCK_SIZE];
				DTYPE C4_temp[BLOCK_SIZE];
        DTYPE temp_temp = 0;

        #pragma unroll
        for (int z = 0; z < BLOCK_SIZE; z++) {
          A_temp[z] = A[i*lllY + k*BLOCK_SIZE + z];
        }

       	for (int z = 0; z < BLOCK_SIZE; z++) {
          C4_temp[z] = C4[j*lllY + k*BLOCK_SIZE + z];
        }

				for (int z = 0; z < BLOCK_SIZE; z++) {

#if INTENSITY1
          megaBfunction (temp_temp, A_temp[z], C4_temp[z]);
#elif INTENSITY2
          megaBfunction2 (temp_temp, A_temp[z], C4_temp[z]);
#elif INTENSITY3
          megaBfunction3 (temp_temp, A_temp[z], C4_temp[z]);
#elif INTENSITY4
          megaBfunction4 (temp_temp, A_temp[z], C4_temp[z]);
#elif INTENSITY5
          megaBfunction5 (temp_temp, A_temp[z], C4_temp[z]);
#endif

        }

        temp += temp_temp;
      }

      sum[i*lllY + j] = temp;
      write_channel_intel (c0, temp);
    }
  }

#endif

}


__kernel void doitgen2 (__global const DTYPE* restrict A,
                        __global const DTYPE* restrict C4,
                        __global DTYPE* restrict sum,
                        const int lllX,
                        const int lllY) {

#ifdef GPU

#endif


#ifdef FPGA_SINGLE

	for (int i = 0; i < lllX; i++) {
    for (int j = 0; j < lllY; j++) {
      DTYPE temp = read_channel_intel(c0);

#if INTENSITY1
			megaBfunction (A[i][j], temp, j);
#elif INTENSITY2
			megaBfunction2 (A[i][j], temp, j);
#elif INTENSITY3
			megaBfunction3 (A[i][j], temp, j);
#elif INTENSITY4
			megaBfunction4 (A[i][j], temp, j);
#elif INTENSITY5
			megaBfunction5 (A[i][j], temp, j);
#endif

    }
  }

#endif

}
