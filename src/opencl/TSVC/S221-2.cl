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

channel DTYPE c0;

__kernel void S221K1 (__global DTYPE* restrict AA,
										__global DTYPE* restrict BB,
                    __global DTYPE* restrict CC
#ifdef FPGA_SINGLE
										,const int lll)
#else
																)
#endif
{

#ifdef FPGA_SINGLE

	DTYPE multiplier = 1.5;

	for (int i = 1; i < lll; i++) {
  	DTYPE tempA;
#if INTENSITY1
		Bfunction (tempA, CC[i], multiplier);
#elif INTENSITY2
		Bfunction2 (tempA, CC[i], multiplier);
#elif INTENSITY3
		Bfunction3 (tempA, CC[i], multiplier);
#elif INTENSITY4
		Bfunction4 (tempA, CC[i], multiplier);
#elif INTENSITY5
		Bfunction5 (tempA, CC[i], multiplier);
#endif

		write_channel_altera (c0, tempA);
	}

#endif
}

__kernel void S221K2 (__global DTYPE* restrict AA,
										__global DTYPE* restrict BB,
                    __global DTYPE* restrict CC
#ifdef FPGA_SINGLE
										,const int lll)
#else
																)
#endif
{


	for (int i = 1; i < lll; i++) {
  	DTYPE tempA = read_channel_altera(c0);
#if INTENSITY1
		Cfunction (BB[i], BB[i-1], multiplier, tempA);
#elif INTENSITY2
		Cfunction2 (BB[i], BB[i-1], multiplier, tempA);
#elif INTENSITY3
		Cfunction3 (BB[i], BB[i-1], multiplier, tempA);
#elif INTENSITY4
		Cfunction4 (BB[i], BB[i-1], multiplier, tempA);
#elif INTENSITY5
		Cfunction5 (BB[i], BB[i-1], multiplier, tempA);
#endif
	}

#endif


}
