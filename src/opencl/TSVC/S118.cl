//
// (c) July 30, 2018 Saman Biookaghazadeh @ Arizona State University
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
__attribute__((reqd_work_group_size(128, 1, 1)))
__attribute__((num_simd_work_items(16)))
__attribute__((num_compute_units(NUM_COMPUTE_UNITS)))
#endif

__kernel void S113 (__global DTYPE* restrict A,
										__global DTYPE* restrict BB
#ifdef FPGA_SINGLE
										,const int lll)
#else
																	)
#endif
{

#ifdef GPU
	const int gid = get_global_id(0);
#endif

#ifdef FPGA_NDRANGE
	const int gid = get_global_id(0);
#endif

#ifdef FPGA_SINGLE
  for (int i = 1; i < lll; i++) {
  	#pragma unroll UNROLL_FACTOR
  	for (int j = 0; j <= i - 1; j++ ) {
    	a[i] += b[j*lll+i] * a[i-j-1];
    }
  }
#else 


#endif

}
