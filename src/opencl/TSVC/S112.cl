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

__kernel void S112 (__global const DTYPE* restrict A,
										__global DTYPE* restrict B,
                    const DTYPE alpha
#ifdef FPGA_SINGLE
										,const int lll)
#else
																	)
#endif
{

#ifdef GPU
	const int groupId = get_group_id(0) * 257;
  const int lid = get_local_id(0);
  const int start = groupId;
 	DTYPE value;
#endif

#ifdef FPGA_NDRANGE
	const int groupId = get_group_id(0) * 257;
  const int lid = get_local_id(0);
	const int start = groupId;
#endif

#ifdef FPGA_SINGLE
	#pragma unroll UNROLL_FACTOR
  for (int gid = lll-2; gid >= 0; gid--) {
		B[gid+1] = B[gid] + A[gid];
	}
#else
	value = B[start+lid] + A[start+lid];
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	B[start+lid+1] = value;
#endif

}
