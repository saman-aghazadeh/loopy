//
// (c) September 26, 2018 Saman Biookaghazadeh @ Arizona State University
//


#ifdef INT_PRECISION
#define DTYPE int
#elif SINGLE_PRECISION
#define DTYPE float
#elif DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define DTYPE double
#endif

#ifdef FPGA_NDRANGE
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((num_simd_work_items(16)))
__attribute__((num_compute_units(NUM_COMPUTE_UNITS)))
#endif

#include "funcs.h"

__kernel void S221 (__global DTYPE* restrict AA,
										__global DTYPE* restrict BB,
                    __global DTYPE* restrict CC
#ifdef FPGA_SINGLE
										,const int lll)
#else
																)
#endif
{

#ifdef GPU
	DTYPE multiplier = 1.5;

	int gid = get_global_id(0);

#if INTENSITY1
	Bfunction (AA[gid], CC[gid], multiplier);
#elif INTENSITY2
	Bfunction2 (AA[gid], CC[gid], multiplier);
#elif INTENSITY3
	Bfunction3 (AA[gid], CC[gid], multiplier);
#elif INTENSITY4
	Bfunction4 (AA[gid], CC[gid], multiplier);
#elif INTENSITY5
	Bfunction5 (AA[gid], CC[gid], multiplier);
#endif

#endif

#ifdef FPGA_SINGLE

	DTYPE multiplier = 1.5;

	#pragma unroll UNROLL_FACTOR1
	for (int i = 1; i < lll; i++) {
#if INTENSITY1
		Bfunction (AA[i], CC[i], multiplier);
#elif INTENSITY2
		Bfunction2 (AA[i], CC[i], multiplier);
#elif INTENSITY3
		Bfunction3 (AA[i], CC[i], multiplier);
#elif INTENSITY4
		Bfunction4 (AA[i], CC[i], multiplier);
#elif INTENSITY5
		Bfunction5 (AA[i], CC[i], multiplier);
#endif
	}

	#pragma unroll UNROLL_FACTOR2
	for (int i = 1; i < lll; i++) {
#if INTENSITY1
		Cfunction (BB[i], BB[i-1], multiplier, AA[i]);
#elif INTENSITY2
		Cfunction2 (BB[i], BB[i-1], multiplier, AA[i]);
#elif INTENSITY3
		Cfunction3 (BB[i], BB[i-1], multiplier, AA[i]);
#elif INTENSITY4
		Cfunction4 (BB[i], BB[i-1], multiplier, AA[i]);
#elif INTENSITY5
		Cfunction5 (BB[i], BB[i-1], multiplier, AA[i]);
#endif
	}

#endif


}
