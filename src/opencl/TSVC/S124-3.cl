//
// (c) August 4, 2018 Saman Biookaghazadeh @ Arizona State University
//

#ifdef INT_PRECISION
#define DTYPE int
#elif SINGLE_PRECISION
#define DTYPE float
#elif DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define DTYPE double
#endif

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


	A[gid] = B[gid] + D[gid] - E[gid];


#endif

#ifdef FPGA_NDRANGE
	const int gid = get_global_id(0);

	DTYPE B_local = B[gid];
  DTYPE C_local = C[gid];
  DTYPE D_local = D[gid];
  DTYPE E_local = E[gid];
  DTYPE A_local = 0;

	if ( B_local > 10) {
		A_local = B_local + D_local - E_local;
	} else if (B_local > 0) {
  	A_local = B_local + D_local + E_local;
	} else if (B_local < -10){
		A_local = C_local + D_local - E_local;
	} else if (B_local < 0) {
		A_local = C_local + D_local + E_local;
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

		A_local = B_local + D_local - E_local;

		A[i] = A_local;
	}

#endif

}
