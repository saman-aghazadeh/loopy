//
// (c) August 20, 2018 Saman Biookaghazadeh @ Arizona State University
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

#ifdef FPGA_NDRANGE
__attribute__((reqd_work_group_size(16, 16, 1)))
__attribute__((num_simd_work_items(16)))
__attribute__((num_compute_units(NUM_COMPUTE_UNITS)))
#endif


__kernel void S119 (__global DTYPE* restrict AA,
										__global DTYPE* restrict BB
#ifdef FPGA_SINGLE
										,const int lllX
                    ,const int lllY)
#else
																	)
#endif
{

#ifdef FPGA_SINGLE


	int exit = lllX / BLOCK_SIZE;
  int i = 0;

	while (i < exit) {

		DTYPE buffer[BLOCK_SIZE+1];

		#pragma unroll
		for (int j = 0; j <= BLOCK_SIZE; j++) {
			buffer[j] = AA[i*BLOCK_SIZE+j];
		}

		for (int k = 1; k < lllY; k++) {
	
			#pragma unroll
			for (int j = BLOCK_SIZE; j >= 1; j--) {
				buffer[j] = buffer[j-1];
			}

			buffer[0] = AA[k*lllX];

			#pragma ivdep
   		#pragma unroll UNROLL_FACTOR
			for (int j = 1; j <= BLOCK_SIZE; j++) {
				buffer[j] = buffer[j] + BB[k*lllX + i*BLOCK_SIZE + j];
			}	

			#pragma unroll
			for (int j = 1; j < BLOCK_SIZE; j++) {
				AA[k*lllX + i*BLOCK_SIZE + j] = buffer[j];
			}

		}

		i++;
	}
#endif

}
