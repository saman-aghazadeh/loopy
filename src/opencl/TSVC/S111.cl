//
// (c) July 26, 2018 Saman Biookaghazadeh @ Arizona State University
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

__kernel void S111 (__global const DTYPE* restrict A,
										__global DTYPE* restrict B,
                    const DTYPE alpha
#ifdef FPGA_SINGLE
										,const int lll)
#else
																	)
#endif
{

#ifdef GPU
	const int gid = 2*get_global_id(0);
#endif

#ifdef FPGA_NDRANGE
	const int gid = 2*get_global_id(0);
#endif

#ifdef FPGA_SINGLE
	#pragma unroll UNROLL_FACTOR
	#pragma ivdep array(B)
  for (int gid = 0; gid < lll; gid += 2) {
#endif

	B[gid] = B[gid-1] + A[gid];

#ifdef FPGA_SINGLE
	}
#endif

}
