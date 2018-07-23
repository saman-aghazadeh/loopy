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

__kernel void Stream( __global DTYPE* restrict A,
		      __global const DTYPE* restrict B,
                      __global const DTYPE* restrict C,
                      const DTYPE alpha
#ifdef FPGA_SINGLE
		      ,const int numIterations)
#else
					)
#endif
 
{
#ifdef GPU
	const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  const int lSize = get_local_size(0);

	__local DTYPE localB[256];
  __local DTYPE localC[256];
	__local DTYPE localA[256];

	localB[lid] = B[gid];
  localC[lid] = C[gid];
#endif

#ifdef FPGA_NDRANGE
	const int gid = get_global_id(0);
#endif

#ifdef FPGA_SINGLE
	#pragma unroll UNROLL_FACTOR
	for (int gid = 0; gid < numIterations; gid++) {
#endif

	localA[lid] = localB[lid] + alpha * localC[lid];
  A[gid] = localA[lid];

#ifdef FPGA_SINGLE
	}
#endif

}
