//
// (c) August 11, 2018 Saman Biookaghazadeh @ Arizona State University
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


__kernel void S1119 (__global DTYPE* restrict AA,
										__global const DTYPE* restrict BB,
                    const int lllX
#ifdef FPGA_SINGLE
 ,const int lllY)
#else
																	)
#endif
{

#ifdef GPU

	const int gid = get_global_id(0);
	const int size = get_global_size(0);

	for (int i = 1; i < lllX; i++) {
  	AA[gid*lllX+i] = AA[gid*lllX+(i-1)] + BB[gid*lllX+i];
	}
	
#endif

#ifdef FPGA_NDRANGE
	const int gid = get_global_id(0);
	const int size = get_global_size(0);

	#pragma unroll UNROLL_FACTOR
	for (int i = 1; i < lllX; i++) {
		AA[gid*lllX+i] = AA[gid*lllX+(i-1)] + BB[gid*lllX+i];
	}
#endif

#ifdef FPGA_SINGLE

  for (int i = 1; i < lllX; i++) {
  	#pragma ivdep
    #pragma unroll UNROLL_FACTOR
  	for (int j = 0; j < lllY; j++) {
			AA[i*lllY+j] = AA[(i-1)*lllY+j] + BB[i*lllY+j];
		}
  }

	for (int i = 1; i < lllX; i++) {
  	#pragma ivdep
    #pragma unroll UNROLL_FACTOR
		for (int j = 0; j < lllY; j++) {
			AA[j*lllX+i] = AA[j*lllX+(i-1)] + BB[j*lllX+i];
		}
	}

#endif

}
