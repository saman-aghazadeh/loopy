//
// (c) August 23, 2018 Saman Biookaghazadeh @ Arizona State University
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


__kernel void S211K1 (__global DTYPE* restrict A,
										__global const DTYPE* restrict B,
                    __global DTYPE* restrict BPrime,
                    __global const DTYPE* restrict C,
                    __global const DTYPE* restrict D,
                    __global const DTYPE* restrict E
#if FPGA_SINGLE
										,const int lll)
#else
										)
#endif

{

#ifdef GPU
	const int gid = get_global_id(0);
	const int size = get_global_size(0);
  const int index = gid+1;

#if INTENSITY1
	Cfunction(BPrime[index], B[index+1], E[index], D[index]);	
#elif INTENSITY2
	Cfunction2(BPrime[index], B[index+1], E[index], D[index]);
#elif INTENSITY3
	Cfunction3(BPrime[index], B[index+1], E[index], D[index]);
#elif INTENSITY4
	Cfunction4(BPrime[index], B[index+1], E[index], D[index]);
#elif INTENSITY5
	Cfunction5(BPrime[index], B[index+1], E[index], D[index]);
#elif INTENSITY6
	Cfunction6(BPrime[index], B[index+1], E[index], D[index]);
#elif INTENSITY7
	Cfunction7(BPrime[index], B[index+1], E[index], D[index]);
#elif INTENSITY8
	Cfunction8(BPrime[index], B[index+1], E[index], D[index]);
#endif
	
#endif

#ifdef FPGA_NDRANGE
	const int gid = get_global_id(0);
  cosnt int size = get_global_size(0);
	const int index = gid+1;

#if INTENSITY1
	Cfunction(BPrime[index], B[index+1], E[index], D[index]);
#elif INTENSITY2
	Cfunction2(BPrime[index], B[index+1], E[index], D[index]);
#elif INTENSITY3
	Cfunction3(BPrime[index], B[index+1], E[index], D[index]);
#elif INTENSITY4
	Cfunction4(BPrime[index], B[index+1], E[index], D[index]);
#elif INTENSITY5
	Cfunction5(BPrime[index], B[index+1], E[index], D[index]);
#elif INTENSITY6
	Cfunction6(BPrime[index], B[index+1], E[index], D[index]);
#elif INTENSITY7
	Cfunction7(BPrime[index], B[index+1], E[index], D[index]);
#elif INTENSITY8
	Cfunction8(BPrime[index], B[index+1], E[index], D[index]);
#endif

#endif

#ifdef FPGA_SINGLE

#endif

}

__kernel void S211K2 (__global DTYPE* restrict A,
										__global const DTYPE* restrict B,
                    __global DTYPE* restrict BPrime,
                    __global const DTYPE* restrict C,
                    __global const DTYPE* restrict D,
                    __global const DTYPE* restrict E
#if FPGA_SINGLE
										,const int lll)
#else
										)
#endif

{

#ifdef GPU
	const int gid = get_global_id(0);
	const int size = get_global_size(0);
  const int index = gid+1;

#if INTENSITY1
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#elif INTENSITY2
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#elif INTENSITY3
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#elif INTENSITY4
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#elif INTENSITY5
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#elif INTENSITY6
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#elif INTENSITY7
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#elif INTENSITY8
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#endif

#endif

#ifdef FPGA_NDRANGE
	const int gid = get_global_id(0);
  cosnt int size = get_global_size(0);
	const int index = gid+1;

#if INTENSITY1
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#elif INTENSITY2
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#elif INTENSITY3
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#elif INTENSITY4
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#elif INTENSITY5
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#elif INTENSITY6
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#elif INTENSITY7
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#elif INTENSITY8
	Cfunction(A[index], BPrime[index-11], E[index], D[index]);
#endif

#endif

#ifdef FPGA_SINGLE

#endif

}
