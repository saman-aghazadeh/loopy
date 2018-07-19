//
// (c) July 18, 2018 Saman Biookaghazadeh @ Arizona State University
//

#ifdef INT_PRECISION
#define DTYPE int
#elif SINGLE_PRECISION
#define DTYPE float
#elif DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define DTYPE double
#endif


__kernel void Stream( __global DTYPE* restrict A,
											__global const DTYPE* restrict B,
                      __global const DTYPE* restrict C,
                     	const DTYPE alpha)
 
{

	const int gid = get_global_id(0);

	A[gid] = B[gid] + alpha * C[gid];

}