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

	int i = get_global_id(0);
  int j = get_global_id(1);

  float sum = 0.0;

  for (int k = 0; k < lllY; k++) {
#if INTENSITY1
		megaBfunction (sum, A[j*lllY + k], C4[i*lllY + k]);
#elif INTENSITY2
		megaBfunction2 (sum, A[j*lllY + k], C4[i*lllY + k]);
#elif INTENSITY3
		megaBfunction3 (sum, A[j*lllY + k], C4[i*lllY + k]);
#elif INTENSITY4
		megaBfunction4 (sum, A[j*lllY + k], C4[i*lllY + k]);
#elif INTENSITY5
		megaBfunction5 (sum, A[j*lllY + k], C4[i*lllY + k]);
#endif
  }


#endif


#ifdef FPGA_SINGLE

	for (int i = 0; i < lllX; i++) {
    for (int j = 0; j < lllY; j++) {

      DTYPE temp_copies[4];
			DTYPE temp = 0.0;

      #pragma unroll
      for (int k = 0; k < 4; k++)
				temp_copies[k] = 0.1;

      for (int k = 0; k < lllY/BLOCK_SIZE; k++) {
        DTYPE A_temp[BLOCK_SIZE];
				DTYPE C4_temp[BLOCK_SIZE];
        DTYPE temp_temp = 0.1;

        #pragma unroll
        for (int z = 0; z < BLOCK_SIZE; z++) {
          A_temp[z] = A[i*lllY + k*BLOCK_SIZE + z];
        }

        #pragma unroll
       	for (int z = 0; z < BLOCK_SIZE; z++) {
          C4_temp[z] = C4[j*lllY + k*BLOCK_SIZE + z];
        }

        #pragma unroll
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

				DTYPE cur = temp_copies[3] + temp_temp;

        #pragma unroll
        for (int z = 3; z > 0; z--)
          temp_copies[z] = temp_copies[z-1];

				temp_copies[0] = cur;
      }

      #pragma unroll
			for (int i = 0; i < 4; i++)
				temp += temp_copies[i];

      write_channel_intel (c0, temp);
    }
  }

#endif

}


__kernel void doitgen2 (__global DTYPE* restrict AA,
												__global DTYPE* restrict sum,
                        const int lllX,
                        const int lllY) {

#ifdef GPU

  int i = get_global_id(0);
  int j = get_global_id(1);

#if INTENSITY1
	megaBfunction (AA[j*lllY + i], sum[j*lllY + i], j);
#elif INTENSITY2
	megaBfunction2 (AA[j*lllY + i], sum[j*lllY + i], j);
#elif INTENSITY3
	megaBfunction3 (AA[j*lllY + i], sum[j*lllY + i], j);
#elif INTENSITY4
	megaBfunction4 (AA[j*lllY + i], sum[j*lllY + i], j);
#elif INTENSITY5
	megaBfunction5 (AA[j*lllY + i], sum[j*lllY + i], i);
#endif

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
