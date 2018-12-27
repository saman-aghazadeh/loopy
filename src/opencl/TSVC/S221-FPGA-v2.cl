//
// (c) Septembre 13, 2018 Saman Biookaghazadeh @ Arizona State University
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

	DTYPE multiplier = 1.5;

	for (int i = 1; i < lll; i++) {
		DTYPE temp;
#if INTENSITY1
		megaBfunction (temp, CC[i], multiplier);
#elif INTENSITY2
		megaBfunction2 (temp, CC[i], multiplier);
#elif INTENSITY3
		megaBfunction3 (temp, CC[i], multiplier);
#elif INTENSITY4
		megaBfunction4 (temp, CC[i], multiplier);
#elif INTENSITY5
		megaBfunction5 (temp, CC[i], multiplier);
#endif

		write_channel_altera (c0, temp);
	}

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

	DTYPE multiplier = 1.5;

	int sum = 0.0;

	for (int i = 1; i < lll; i++) {
		DTYPE temp = read_channel_altera(c0);
    DTYPE temp2 = 0;
#if INTENSITY1
		megaCfunction (temp2, BB[i-1], multiplier, temp);
#elif INTENSITY2
		megaCfunction (temp2, BB[i-1], multiplier, temp);
#elif INTENSITY3
		megaCfunction (temp2, BB[i-1], multiplier, temp);
#elif INTENSITY4
		megaCfunction (temp2, BB[i-1], multiplier, temp);
#elif INTENSITY5
		megaCfunction (temp2, BB[i-1], multiplier, temp);
#endif

		sum += temp2;
	}

	B[0] = sum;

}