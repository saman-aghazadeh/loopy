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

#ifdef FPGA_NDRANGE
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((num_simd_work_items(16)))
__attribute__((num_compute_units(NUM_COMPUTE_UNITS)))
#endif

__kernel void Reduction( __global const DTYPE* restrict A,
					__global DTYPE* restrict B
#ifdef FPGA_SINGLE
		      ,const int numIterations)
#else
					)
#endif
 
{
#ifdef GPU
	const int gid = get_global_id(0);
  const int groupid = get_group_id(0);
  const int lsize = get_local_size(0);
  const int lid = get_local_id(0);
  const int start = lsize * groupid * PACK + lid;
  DTYPE sum[256] = {0};
#endif

#ifdef FPGA_NDRANGE
	const int gid = get_global_id(0);
#endif

#ifdef FPGA_SINGLE
	#pragma unroll UNROLL_FACTOR
	for (int gid = 0; gid < numIterations; gid++) {
#endif

	#pragma unroll PACK
	for (int i = 0; i < PACK; i++) 
		sum[lid] = sum[lid] + A[start + lsize * i];

	barrier (CLK_GLOBAL_MEM_FENCE);
  if (lid == 0) {
		for (int i = 1; i < 256; i++)
    	sum[0] += sum[i];
	}

	B[groupid] = sum[0];

#ifdef FPGA_SINGLE
	}
#endif

}
