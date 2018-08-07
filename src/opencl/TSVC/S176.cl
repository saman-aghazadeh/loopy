//
// (c) August 5, 2018 Saman Biookaghazadeh @ Arizona State University
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


__kernel void S176 (__global DTYPE* restrict A,
										__global const DTYPE* restrict B,
                    __global const DTYPE* restrict C
#if FPGA_SINGLE
										,const int lll)
#else
										)
#endif

{

#ifdef GPU
	const int gid = get_global_id(0);
	const int size = get_global_size(0);

	for (int j = 0; j < size; j++) {
		A[gid] += B[gid+size-j-1] * C[j];
	}
	
#endif

#ifdef FPGA_NDRANGE
	const int gid = get_global_id(0);
  cosnt int size = get_global_size(0);

	for (int j = 0; j < size; j++) {
		A[gid] += B[gid+size-j-1] * C[j];
	}
#endif

#ifdef FPGA_SINGLE

	int m = lll/2;

	for (int j = 0; j < m; j++ ) {
  	#pragma unroll UNROLL_FACTOR
		for (int i = 0; i < m; i++) {
			A[i] += B[i+m-j-1] * C[j];
		}
	}

#endif

}
