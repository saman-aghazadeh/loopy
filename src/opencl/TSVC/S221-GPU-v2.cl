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

	const int gid = get_global_id(0);

#if INTENSITY1
	megaBfunction (AA[gid], CC[gid], multiplier);
#elif INTENSITY2
	megaBfunction2 (AA[gid], CC[gid], multiplier);
#elif INTENSITY3
	megaBfunction3 (AA[gid], CC[gid], multiplier);
#elif INTENSITY4
	megaBfunction4 (AA[gid], CC[gid], multiplier);
#elif INTENSITY5
	megaBfunction5 (AA[gid], CC[gid], multiplier);
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


}