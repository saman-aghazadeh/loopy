//
// (c) August 16, 2018 Saman Biookaghazadeh @ Arizona State University
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
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((num_simd_work_items(16)))
__attribute__((num_compute_units(NUM_COMPUTE_UNITS)))
#endif


__kernel void S1119 (__global DTYPE* restrict AA,
										__global const DTYPE* restrict BB,
                    const int lllX
#ifdef FPGA_SINGLE
                    ,const int lllY)
#else
																	)
#endif
{

#ifdef GPU

	const int gid = get_global_id(0);
	const int size = get_global_size(0);

	for (int i = 1; i < lllX; i++) {

#if INTENSITY1
		Bfunction(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY2
		Bfunction2(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY3
		Bfunction3(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY4
		Bfunction4(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY5
		Bfunction5(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY6
		Bfunction6(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY7
		Bfunction7(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY8
		Bfunction8(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#endif
	}	
#endif


#ifdef FPGA_NDRANGE
	const int gid = get_global_id(0);
	const int size = get_global_size(0);

	#pragma unroll UNROLL_FACTOR
	for (int i = 1; i < lllX; i++) {
		AA[i*size+gid] = AA[(i-1)*size+gid] + BB[i*size+gid];
	}
#endif

#ifdef FPGA_SINGLE

  for (int i = 1; i < lllX; i++) {
  	#pragma ivdep
    #pragma unroll UNROLL_FACTOR
  	for (int j = 0; j < lllY; j++) {
			AA[i*lllY+j] = AA[(i-1)*lllY+j] + BB[i*lllY+j];
		}
  }

#endif

}
