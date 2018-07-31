//
// (c) July 31, 2018 Saman Biookaghazadeh @ Arizona State University
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


__kernel void S113 (__global DTYPE* restrict AA,
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
  	for (int j = 1; j < lll; j++) {
			AA[i*lll+j] = AA[(i-1)*lll+(j-1)] + BB[i*lll+j];
		}
  }
#else


#endif

}
