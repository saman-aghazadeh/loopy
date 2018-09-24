//
// (c) September 4, 2018 Saman Biookaghazadeh @ Arizona State University
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


__kernel void S124 (__global DTYPE* restrict A,
										__global const DTYPE* restrict B,
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
  const int gidM = gid%4;
	DTYPE multiplier = 1.5;
  DTYPE additive = 2.5;


	if (gidM == 0) {
#if INTENSITY1
		CCfunction(A[gid], B[gid], multiplier, additive);
#elif INTENSITY2
		CCfunction2(A[gid], B[gid], multiplier, additive);
#elif INTENSITY3
		CCfunction3(A[gid], B[gid], multiplier, additive);
#elif INTENSITY4
		CCfunction4(A[gid], B[gid], multiplier, additive);
#elif INTENSITY5
		CCfunction5(A[gid], B[gid], multiplier, additive);
#elif INTENSITY6
		CCfunction6(A[gid], B[gid], multiplier, additive);
#endif

	} else if (gidM == 1) {
#if INTENSITY1
		CCfunction(A[gid], B[gid], multiplier, -additive);
#elif INTENSITY2
		CCfunction2(A[gid], B[gid], multiplier, -additive);
#elif INTENSITY3
		CCfunction3(A[gid], B[gid], multiplier, -additive);
#elif INTENSITY4
		CCfunction4(A[gid], B[gid], multiplier, -additive);
#elif INTENSITY5
		CCfunction5(A[gid], B[gid], multiplier, -additive);
#elif INTENSITY6
		CCfunction6(A[gid], B[gid], multiplier, -additive);
#endif

	} else if (gidM == 2) {
#if INTENSITY1
		CCfunction(A[gid], C[gid], multiplier, additive);
#elif INTENSITY2
		CCfunction2(A[gid], C[gid], multiplier, additive);
#elif INTENSITY3
		CCfunction3(A[gid], C[gid], multiplier, additive);
#elif INTENSITY4
		CCfunction4(A[gid], C[gid], multiplier, additive);
#elif INTENSITY5
		CCfunction5(A[gid], C[gid], multiplier, additive);
#elif INTENSITY6
		CCfunction6(A[gid], C[gid], multiplier, additive);
#endif

	} else if (gidM == 3) {
#if INTENSITY1
		CCfunction(A[gid], C[gid], multiplier, -additive);
#elif INTENSITY2
		CCfunction2(A[gid], C[gid], multiplier, -additive);
#elif INTENSITY3
		CCfunction3(A[gid], C[gid], multiplier, -additive);
#elif INTENSITY4
		CCfunction4(A[gid], C[gid], multiplier, -additive);
#elif INTENSITY5
		CCfunction5(A[gid], C[gid], multiplier, -additive);
#elif INTENSITY6
		CCfunction6(A[gid], C[gid], multiplier, -additive);
#endif

	}

#endif

#ifdef FPGA_NDRANGE

#endif

#ifdef FPGA_SINGLE

	DTYPE multiplier = 1.5;
  DTYPE additive = 2.5;

	int j = -1;
  #pragma unroll UNROLL_FACTOR
	for (int i = 0; i < lll; i++) {

		DTYPE B_local = B[i];
 		DTYPE C_local = C[i];
	  DTYPE A_local = 0;

		int iM = i % 4;

	if (iM == 0) {
#if INTENSITY1
		CCfunction(A_local, B_local, multiplier, additive);
#elif INTENSITY2
		CCfunction2(A_local, B_local, multiplier, additive);
#elif INTENSITY3
		CCfunction3(A_local, B_local, multiplier, additive);
#elif INTENSITY4
		CCfunction4(A_local, B_local, multiplier, additive);
#elif INTENSITY5
		CCfunction5(A_local, B_local, multiplier, additive);
#elif INTENSITY6
		CCfunction6(A_local, B_local, multiplier, additive);
#endif

	} else if (iM == 1) {
#if INTENSITY1
		CCfunction(A_local, B_local, multiplier, -additive);
#elif INTENSITY2
		CCfunction2(A_local, B_local, multiplier, -additive);
#elif INTENSITY3
		CCfunction3(A_local, B_local, multiplier, -additive);
#elif INTENSITY4
		CCfunction4(A_local, B_local, multiplier, -additive);
#elif INTENSITY5
		CCfunction5(A_local, B_local, multiplier, -additive);
#elif INTENSITY6
		CCfunction6(A_local, B_local, multiplier, -additive);
#endif

	} else if (iM == 2) {
#if INTENSITY1
		CCfunction(A_local, C_local, multiplier, additive);
#elif INTENSITY2
		CCfunction2(A_local, C_local, multiplier, additive);
#elif INTENSITY3
		CCfunction3(A_local, C_local, multiplier, additive);
#elif INTENSITY4
		CCfunction4(A_local, C_local, multiplier, additive);
#elif INTENSITY5
		CCfunction5(A_local, C_local, multiplier, additive);
#elif INTENSITY6
		CCfunction6(A_local, C_local, multiplier, additive);
#endif

	} else if (iM == 3) {
#if INTENSITY1
		CCfunction(A_local, C_local, multiplier, -additive);
#elif INTENSITY2
		CCfunction2(A_local, C_local, multiplier, -additive);
#elif INTENSITY3
		CCfunction3(A_local, C_local, multiplier, -additive);
#elif INTENSITY4
		CCfunction4(A_local, C_local, multiplier, -additive);
#elif INTENSITY5
		CCfunction5(A_local, C_local, multiplier, -additive);
#elif INTENSITY6
		CCfunction6(A_local, C_local, multiplier, -additive);
#endif

	}

	A[i] = A_local;

#endif
}
