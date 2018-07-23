//
// (c) July 19, 2018 Saman Biookaghazadeh @ Arizona State University
//
// Considerations: 
// This code works on a two dimensional dataset
// The host code needs to make sure that the horizontal
// and vertical size of the data is similar. That means
// get_global_size(0) and get_global_size(1) both return
// the same value.
//
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
__attribute__((reqd_work_group_size(256,1,1)))
__attribute__((num_simd_work_items(16)))
__attribute__((num_compute_units(NUM_COMPUTE_UNITS)))
#endif

__kernel void Random (__global DTYPE* restrict A,
												 __global const DTYPE* restrict B,
                         __global const unsigned int* restrict C
#ifdef FPGA_SINGLE                      
                         ,const int lengthX)
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
#endif 

#ifdef FPGA_NDRANGE
		const int gidX = get_global_id(0);
#endif

#ifdef FPGA_SINGLE
	for (int gidX = 0; gidX < lengthX; gidX++) {
#endif

	#pragma unroll PACK
	for (int i = 0; i < PACK; i++)
		A[start + lsize * i] = B[C[start + lsize * i]];

#ifdef FPGA_SINGLE
	}
#endif

}