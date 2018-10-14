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
										, const int startX
                    , const int startY)
{

#ifdef GPU
	const int gidX = get_global_id(0);
  	const int gidY = get_global_id(1);

	const int sizeX = get_global_size(0);
  	const int sizeY = get_global_size(1);

	if ( gidX < startX && gidX > startY && gidY < startX && gidY > startY) {

		DTYPE top = AA[gidX * sizeY + (gidY-1)];
      		DTYPE left = AA[(gidX-1) * sizeY + gidY];
      		DTYPE topleft = AA[(gidX-1) * sizeY + (gidY-1)];

#if INTENSITY1
		Dfunction(AA[gidX * sizeY + gidY], top, left, topleft, BB[gidX * sizeY + gidY]);
#elif INTENSITY2
		Dfunction2(AA[gidX * sizeY + gidY], top, left, topleft, BB[gidX * sizeY + gidY]);
#elif INTENSITY3
		Dfunction3(AA[gidX * sizeY + gidY], top, left, topleft, BB[gidX * sizeY + gidY]);
#elif INTENSITY4
		Dfunction4(AA[gidX * sizeY + gidY], top, left, topleft, BB[gidX * sizeY + gidY]);
#elif INTENSITY5
		Dfunction5(AA[gidX * sizeY + gidY], top, left, topleft, BB[gidX * sizeY + gidY]);
#endif
	}
#endif

}
