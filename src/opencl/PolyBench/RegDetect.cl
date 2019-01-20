//
// (c) January 18, 2019 Saman Biookaghazadeh @ Arizona State University
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

__kernel void floydWarshall (__global const DTYPE* restrict path,
                             __global const DTYPE* restrict mean,
                             const int lll) {

#ifdef GPU

#endif

#ifdef FPGA_SINGLE

	int exit = lll / BLOCK_SIZE;
  int i = 0;

  while (i < exit) {

    DTYPE path_buffer[BLOCK_SIZE+1];

		#pragma unroll
    for (int j = 0; j <= BLOCK_SIZE; j++) {
			buffer[j] = path[i*BLOCK_SIZE+j];
    }

    for (int k = 1; k < lll; k++) {

			DTYPE mean_buffer[BLOCK_SIZE+1];

      #pragma unroll
      for (int j = 0; j < BLOCK_SIZE; j++) {
        mean_buffer[j+1] = mean[k*lll + i*BLOCK_SIZE + j];
      }

      #pragma unroll
      for (int j = BLOCK_SIZE; j >= 1; j--) {
        buffer[j] = buffer[j-1];
      }

      buffer[0] = path[k * lll + i * BLOCK_SIZE];

      #pragma ivdep
      #pragma unroll UNROLL_FACTOR
      for (int j = 1; j <= BLOCK_SIZE; j++) {
#if INTENSITY1
				megaBfunction (buffer[j], mean_buffer[j], j);
#elif INTENSITY2
				megaBfunction2 (buffer[j], mean_buffer[j], j);
#elif INTENSITY3
				megaBfunction3 (buffer[j], mean_buffer[j], j);
#elif INTENSITY4
				megaBfunction4 (buffer[j], mean_buffer[j], j);
#elif INTENSITY5
				megaBfunction5 (buffer[j], mean_buffer[j], j);
#endif
      }

      #pragma unroll
      for (int j = 1; j < BLOCK_SIZE; j++) {
        path[k*lll + i*BLOCK_SIZE + j] = buffer[j]
      }

    }

    i++;

  }

#endif

}
