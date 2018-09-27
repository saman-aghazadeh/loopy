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

	DTYPE multiplier = 1.5;

	for (int i = 1; i < lll; i++) {
  	DTYPE temp;
#if INTENSITY1
		Bfunction (temp, CC[i], multiplier);
#elif INTENSITY2
		Bfunction2 (temp, CC[i], multiplier);
#elif INTENSITY3
		Bfunction3 (temp, CC[i], multiplier);
#elif INTENSITY4
		Bfunction4 (temp, CC[i], multiplier);
#elif INTENSITY5
		Bfunction5 (temp, CC[i], multiplier);
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

	for (int j = 0; j < lll/BS; j++) {

    DTYPE tempArray[BS];
    tempArray[0] = BB[i*BS];
    #pragma unroll
		for (int i = 1; i < BS; i++) {
  		DTYPE temp = read_channel_altera(c0);
#if INTENSITY1
			Cfunction (tempArray[i], tempArray[i-1], multiplier, temp);
#elif INTENSITY2
			Cfunction2 (tempArray[i], tempArray[i-1], multiplier, temp);
#elif INTENSITY3
			Cfunction3 (tempArray[i], tempArray[i-1], multiplier, temp);
#elif INTENSITY4
			Cfunction4 (tempArray[i], tempArray[i-1], multiplier, temp);
#elif INTENSITY5
			Cfunction5 (tempArray[i], tempArray[i-1], multiplier, temp);
#endif
		}

		#pragma unroll
		for (int i = 0; i < BS; i++) {
			BB[i + j*BS] = tempArray[i]; 
		}
	}

}
