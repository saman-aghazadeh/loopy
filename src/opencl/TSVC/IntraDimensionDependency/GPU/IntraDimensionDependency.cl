#include "constants.h"

#ifdef INT_PRECISION
#define DTYPE int
#elif SINGLE_PRECISION
#define DTYPE float
#elif DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define DTYPE double
#endif


__kernel void IntraDimensionDependency (__global DTYPE* restrict input,
					__global DTYPE* restrict weight,
					__global DTYPE* restrict output,
					const int lllX) {

	const int gid = get_global_id(0);
	const int size = get_global_size(0);

	int temp;
	temp = input[gid];

	for (int i = 0; i < lllX; i++) {
		temp += temp * weight[i*size+gid];
	}

	output[i] = temp;

} 	
