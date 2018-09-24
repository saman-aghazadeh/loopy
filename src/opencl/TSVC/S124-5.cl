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


__kernel void S124 (__global volatile DTYPE* restrict A,
					__global volatile const DTYPE* restrict B,
                    __global volatile const DTYPE* restrict C,
                    __global volatile const DTYPE* restrict D,
                    __global volatile const DTYPE* restrict E
#if FPGA_SINGLE
										,const int lll)
#else
										)
#endif

{

#ifdef GPU
	const int gid = get_global_id(0);

	const int gidM = gid%2;
  const DTYPE multiplier = 1.5;
  const DTYPE additive = 2.5;

	if (gidM == 0) {
		A[gid] = (B[gid] + additive) * multiplier;
	} else if (gidM == 1){
		A[gid] = (C[gid] + additive) * multiplier;
	}

#endif

#ifdef FPGA_NDRANGE
#endif

#ifdef FPGA_SINGLE

	int j = -1;
  const DTYPE multiplier = 1.5;
  const DTYPE additive = 2.5;
  
  #pragma unroll UNROLL_FACTOR
	for (int i = 0; i < lll; i++) {

		DTYPE B_local = B[i];
  	DTYPE C_local = C[i];
  	DTYPE A_local = 0;

		int iM = i%2;
		if (iM == 0) {
			A_local = (B_local + additive) * multiplier;
		} else if (iM == 1){
			A_local = (C_local + additive) * multiplier;
		}


		A[i] = A_local;
	}

#endif

}
