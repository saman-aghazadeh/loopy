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
__attribute__((reqd_work_group_size(16,16,1)))
__attribute__((num_simd_work_items(16)))
__attribute__((num_compute_units(NUM_COMPUTE_UNITS)))
#endif

__kernel void Transpose (__global const DTYPE* restrict A,
												 __global DTYPE* restrict B
#ifdef FPGA_SINGLE                      
                         ,const int numIterationsX,
                         const int numIterationsY
                         const int lengthX)
#else
												 )
#endif
{

#ifdef GPU
		const int gidX = get_global_id(0);
    const int gidY = get_gloabl_id(1);
		const int lengthX = get_global_size(0);
#endif 

#ifdef FPGA_NDRANGE
		const int gidX = get_global_id(0);
    const int gidY = get_global_id(1);
    const int lengthX = get_global_size(0);
#endif

#ifdef FPGA_SINGLE
	for (int gidX = 0; gidX < numIterationsX; gidX++) {
  	for (int gidY = 0; gidY < numIterationsY; gidY++) {
#endif

			B[gidY*lengthX + gidX] = A[gidX*lengthX + gidY];

#ifdef FPGA_SINGLE
		}
	}
#endif

}