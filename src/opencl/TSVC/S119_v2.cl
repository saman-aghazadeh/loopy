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

#ifdef GPU
	const int gidX = get_global_id(0);
  const int gidY = get_global_id(1);

	const int sizeX = get_global_size(0);
	const int sizeY = get_global_size(1);

	if (gidX == 1 || gidY == 1) {
    int i = gidX;
    int j = gidY;
    DTYPE holder = AA[(i-1)*sizeY+(j-1)];
    while (i < sizeX && j < sizeY) {
#if INTENSITY1
			megaBfunction (holder, holder, BB[i*sizeY+j]);
      AA[i*sizeY+j] = holder;
#elif INTENSITY2
			megaBfunction2 (holder, holder, BB[i*sizeY+j]);
      AA[i*sizeY+j] = holder;
#elif INTENSITY3
			megaBfunction3 (holder, holder, BB[i*sizeY+j]);
      AA[i*sizeY+j] = holder;
#elif INTENSITY4
			megaBfunction4 (holder, holder, BB[i*sizeY+j]);
      AA[i*sizeY+j] = holder;
#elif INTENSITY5
			megaBfunction5 (holder, holder, BB[i*sizeY+j]);
      AA[i*sizeY+j] = holder;
#endif
      i++;
      j++;
		}
	}
	
#endif

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
#if INTENSITY1
				megaBfunction (buffer[j], buffer[j], BB[k*llX + i*BLOCK_SIZE + j]);
#elif INTENSITY2
				megaBfunction2 (buffer[j], buffer[j], BB[k*llX + i*BLOCK_SIZE + j]);
#elif INTENSITY3
				megaBfunction3 (buffer[j], buffer[j], BB[k*llX + i*BLOCK_SIZE + j]);
#elif INTENSITY4
				megaBfunction4 (buffer[j], buffer[j], BB[k*llX + i*BLOCK_SIZE + j]);
#elif INTENSITY5
				megaBfunction5 (buffer[j], buffer[j], BB[k*llX + i*BLOCK_SIZE + j]);
#endif
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
