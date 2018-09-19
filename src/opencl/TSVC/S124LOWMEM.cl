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

	if (gidM == 0) {
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
	} else if (gidM == 1){
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
	} else if (gidM == 2) {
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
	} else if (gidM == 3) {
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

	DTYPE B_local = B[gid];
  DTYPE C_local = C[gid];
  DTYPE D_local = D[gid];
  DTYPE E_local = E[gid];
  DTYPE A_local = 0;

	if (B_local > 10) {
#if INTENSITY1
		Cfunction(A_local, B_local, D_local, (-E_local));
#elif INTENSITY2
		Cfunction2(A_local, B_local, D_local, (-E_local));
#elif INTENSITY3
		Cfunction3(A_local, B_local, D_local, (-E_local));
#elif INTENSITY4
		Cfunction4(A_local, B_local, D_local, (-E_local));
#elif INTENSITY5
		Cfunction5(A_local, B_local, D_local, (-E_local));
#elif INTENSITY6
		Cfunction6(A_local, B_local, D_local, (-E_local));
#endif
	} else if (B_local > 0){
#if INTENSITY1
		Cfunction(A_local, C_local, D_local, E_local);
#elif INTENSITY2
		Cfunction2(A[gid], C_local, D_local, E_local);
#elif INTENSITY3
		Cfunction3(A_local, C_local, D_local, E_local);
#elif INTENSITY4
		Cfunction4(A_local, C_local, D_local, E_local);
#elif INTENSITY5
		Cfunction5(A_local, C_local, D_local, E_local);
#elif INTENSITY6
		Cfunction6(A_local, C_local, D_local, E_local);
#endif
	} else if (B_local < 0) {
#if INTENSITY1
		Cfunction(A_local, C_local, D_local, (-E_local));
#elif INTENSITY2
		Cfunction2(A_local, C_local, D_local, (-E_local));
#elif INTENSITY3
		Cfunction3(A_local, C_local, D_local, (-E_local));
#elif INTENSITY4
		Cfunction4(A_local, C_local, D_local, (-E_local));
#elif INTENSITY5
		Cfunction5(A_local, C_local, D_local, (-E_local));
#elif INTENSITY6
		Cfunction6(A_local, C_local, D_local, (-E_local));
#endif
	} else if (B_local < -10) {
#if INTENSITY1
		Cfunction(A_local, C_local, D_local, E_local);
#elif INTENSITY2
		Cfunction2(A_local, C_local, D_local, E_local);
#elif INTENSITY3
		Cfunction3(A_local, C_local, D_local, E_local);
#elif INTENSITY4
		Cfunction4(A_local, C_local, D_local, E_local);
#elif INTENSITY5
		Cfunction5(A_local, C_local, D_local, E_local);
#elif INTENSITY6
		Cfunction6(A_local, C_local, D_local, E_local);
#endif
	}

	A[gid] = A_local;

#endif

#ifdef FPGA_SINGLE

	int j = -1;
  #pragma unroll UNROLL_FACTOR
	for (int i = 0; i < lll; i++) {

		DTYPE B_local = B[i];
 		DTYPE C_local = C[i];
	  DTYPE D_local = D[i];
	  DTYPE E_local = E[i];
	  DTYPE A_local = 0;

		if (B_local > 10) {
#if INTENSITY1
			Cfunction(A_local, B_local, D_local, (-E_local));
#elif INTENSITY2
			Cfunction2(A_local, B_local, D_local, (-E_local));
#elif INTENSITY3
			Cfunction3(A_local, B_local, D_local, (-E_local));
#elif INTENSITY4
			Cfunction4(A_local, B_local, D_local, (-E_local));
#elif INTENSITY5
			Cfunction5(A_local, B_local, D_local, (-E_local));
#elif INTENSITY6
			Cfunction6(A_local, B_local, D_local, (-E_local));
#endif
		} else if (B_local > 0){
#if INTENSITY1
			Cfunction(A_local, C_local, D_local, E_local);
#elif INTENSITY2
			Cfunction2(A_local, C_local, D_local, E_local);
#elif INTENSITY3
			Cfunction3(A_local, C_local, D_local, E_local);
#elif INTENSITY4
			Cfunction4(A_local, C_local, D_local, E_local);
#elif INTENSITY5
			Cfunction5(A_local, C_local, D_local, E_local);
#elif INTENSITY6
			Cfunction6(A_local, C_local, D_local, E_local);
#endif
		} else if (B_local < 0) {
#if INTENSITY1
			Cfunction(A_local, C_local, D_local, (-E_local));
#elif INTENSITY2
			Cfunction2(A_local, C_local, D_local, (-E_local));
#elif INTENSITY3
			Cfunction3(A_local, C_local, D_local, (-E_local));
#elif INTENSITY4
			Cfunction4(A_local, C_local, D_local, (-E_local));
#elif INTENSITY5
			Cfunction5(A_local, C_local, D_local, (-E_local));
#elif INTENSITY6
			Cfunction6(A_local, C_local, D_local, (-E_local));
#endif
		} else if (B_local < -10) {
#if INTENSITY1
			Cfunction(A_local, C_local, D_local, E_local);
#elif INTENSITY2
			Cfunction2(A_local, C_local, D_local, E_local);
#elif INTENSITY3
			Cfunction3(A_local, C_local, D_local, E_local);
#elif INTENSITY4
			Cfunction4(A_local, C_local, D_local, E_local);
#elif INTENSITY5
			Cfunction5(A_local, C_local, D_local, E_local);
#elif INTENSITY6
			Cfunction6(A_local, C_local, D_local, E_local);
#endif
		}
	}
#endif
}
