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

	if (B[gid] > 10) {
#if INTENSITY1
		Cfunction(A[gid], B[gid], D[gid], (-E[gid]));
#elif INTENSITY2
		Cfunction2(A[gid], B[gid], D[gid], (-E[gid]));
#elif INTENSITY3
		Cfunction3(A[gid], B[gid], D[gid], (-E[gid]));
#elif INTENSITY4
		Cfunction4(A[gid], B[gid], D[gid], (-E[gid]));
#elif INTENSITY5
		Cfunction5(A[gid], B[gid], D[gid], (-E[gid]));
#elif INTENSITY6
		Cfunction6(A[gid], B[gid], D[gid], (-E[gid]));
#endif
	} else if (B[gid] > 0){
#if INTENSITY1
		Cfunction(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY2
		Cfunction2(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY3
		Cfunction3(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY4
		Cfunction4(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY5
		Cfunction5(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY6
		Cfunction6(A[gid], C[gid], D[gid], E[gid]);
#endif
	} else if (B[gid] < 0) {
#if INTENSITY1
		Cfunction(A[gid], C[gid], D[gid], (-E[gid]));
#elif INTENSITY2
		Cfunction2(A[gid], C[gid], D[gid], (-E[gid]));
#elif INTENSITY3
		Cfunction3(A[gid], C[gid], D[gid], (-E[gid]));
#elif INTENSITY4
		Cfunction4(A[gid], C[gid], D[gid], (-E[gid]));
#elif INTENSITY5
		Cfunction5(A[gid], C[gid], D[gid], (-E[gid]));
#elif INTENSITY6
		Cfunction6(A[gid], C[gid], D[gid], (-E[gid]));
#endif
	} else if (B[gid] < -10) {
#if INTENSITY1
		Cfunction(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY2
		Cfunction2(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY3
		Cfunction3(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY4
		Cfunction4(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY5
		Cfunction5(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY6
		Cfunction6(A[gid], C[gid], D[gid], E[gid]);
#endif
	}	

#endif

#ifdef FPGA_NDRANGE
	const int gid = get_global_id(0);

	if (B[gid] > 10) {
#if INTENSITY1
		Cfunction(A[gid], B[gid], D[gid], (-E[gid]));
#elif INTENSITY2
		Cfunction2(A[gid], B[gid], D[gid], (-E[gid]));
#elif INTENSITY3
		Cfunction3(A[gid], B[gid], D[gid], (-E[gid]));
#elif INTENSITY4
		Cfunction4(A[gid], B[gid], D[gid], (-E[gid]));
#elif INTENSITY5
		Cfunction5(A[gid], B[gid], D[gid], (-E[gid]));
#elif INTENSITY6
		Cfunction6(A[gid], B[gid], D[gid], (-E[gid]));
#endif
	} else if (B[gid] > 0){
#if INTENSITY1
		Cfunction(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY2
		Cfunction2(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY3
		Cfunction3(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY4
		Cfunction4(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY5
		Cfunction5(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY6
		Cfunction6(A[gid], C[gid], D[gid], E[gid]);
#endif
	} else if (B[gid] < 0) {
#if INTENSITY1
		Cfunction(A[gid], C[gid], D[gid], (-E[gid]));
#elif INTENSITY2
		Cfunction2(A[gid], C[gid], D[gid], (-E[gid]));
#elif INTENSITY3
		Cfunction3(A[gid], C[gid], D[gid], (-E[gid]));
#elif INTENSITY4
		Cfunction4(A[gid], C[gid], D[gid], (-E[gid]));
#elif INTENSITY5
		Cfunction5(A[gid], C[gid], D[gid], (-E[gid]));
#elif INTENSITY6
		Cfunction6(A[gid], C[gid], D[gid], (-E[gid]));
#endif
	} else if (B[gid] < -10) {
#if INTENSITY1
		Cfunction(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY2
		Cfunction2(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY3
		Cfunction3(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY4
		Cfunction4(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY5
		Cfunction5(A[gid], C[gid], D[gid], E[gid]);
#elif INTENSITY6
		Cfunction6(A[gid], C[gid], D[gid], E[gid]);
#endif
	}

#endif

#ifdef FPGA_SINGLE

	int j = -1;
  #pragma unroll UNROLL_FACTOR
	for (int i = 0; i < lll; i++) {
		if (B[i] > 10) {
#if INTENSITY1
			Cfunction(A[i], B[i], D[i], (-E[i]));
#elif INTENSITY2
			Cfunction2(A[i], B[i], D[i], (-E[i]));
#elif INTENSITY3
			Cfunction3(A[i], B[i], D[i], (-E[i]));
#elif INTENSITY4
			Cfunction4(A[i], B[i], D[i], (-E[i]));
#elif INTENSITY5
			Cfunction5(A[i], B[i], D[i], (-E[i]));
#elif INTENSITY6
			Cfunction6(A[i], B[i], D[i], (-E[i]));
#endif
		} else if (B[i] > 0){
#if INTENSITY1
			Cfunction(A[i], C[i], D[i], E[i]);
#elif INTENSITY2
			Cfunction2(A[i], C[i], D[i], E[i]);
#elif INTENSITY3
			Cfunction3(A[i], C[i], D[i], E[i]);
#elif INTENSITY4
			Cfunction4(A[i], C[i], D[i], E[i]);
#elif INTENSITY5
			Cfunction5(A[i], C[i], D[i], E[i]);
#elif INTENSITY6
			Cfunction6(A[i], C[i], D[i], E[i]);
#endif
		} else if (B[i] < 0) {
#if INTENSITY1
			Cfunction(A[i], C[i], D[i], (-E[i]));
#elif INTENSITY2
			Cfunction2(A[i], C[i], D[i], (-E[i]));
#elif INTENSITY3
			Cfunction3(A[i], C[i], D[i], (-E[i]));
#elif INTENSITY4
			Cfunction4(A[i], C[i], D[i], (-E[i]));
#elif INTENSITY5
			Cfunction5(A[i], C[i], D[i], (-E[i]));
#elif INTENSITY6
			Cfunction6(A[i], C[i], D[i], (-E[i]));
#endif
		} else if (B[i] < -10) {
#if INTENSITY1
			Cfunction(A[i], C[i], D[i], E[i]);
#elif INTENSITY2
			Cfunction2(A[i], C[i], D[i], E[i]);
#elif INTENSITY3
			Cfunction3(A[i], C[i], D[i], E[i]);
#elif INTENSITY4
			Cfunction4(A[i], C[i], D[i], E[i]);
#elif INTENSITY5
			Cfunction5(A[i], C[i], D[i], E[i]);
#elif INTENSITY6
			Cfunction6(A[i], C[i], D[i], E[i]);
#endif
		}
	}
#endif
*/
}