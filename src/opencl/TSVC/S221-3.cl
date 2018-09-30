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

	DTYPE multiplier = 1.5;

	#pragma unroll UNROLL_FACTOR
	for (int i = 1; i < lll; i++) {
#if INTENSITY1
		Bfunction (A[i], CC[i], multiplier);
#elif INTENSITY2
		Bfunction2 (A[i], CC[i], multiplier);
#elif INTENSITY3
		Bfunction3 (A[i], CC[i], multiplier);
#elif INTENSITY4
		Bfunction4 (A[i], CC[i], multiplier);
#elif INTENSITY5
		Bfunction5 (A[i], CC[i], multiplier);
#endif
	}


	for (int i = 1; i < lll; i++) {
#if INTENSITY1
		Cfunction (BB[i], BB[i-1], multiplier, A[i]);
#elif INTENSITY2
		Cfunction2 (BB[i], BB[i-1], multiplier, A[i]);
#elif INTENSITY3
		Cfunction3 (BB[i], BB[i-1], multiplier, A[i]);
#elif INTENSITY4
		Cfunction4 (BB[i], BB[i-1], multiplier, A[i]);
#elif INTENSITY5
		Cfunction5 (BB[i], BB[i-1], multiplier, A[i]);
#endif
	}

}
