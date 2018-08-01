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
__attribute__((reqd_work_group_size(16, 16, 1)))
__attribute__((num_simd_work_items(16)))
__attribute__((num_compute_units(NUM_COMPUTE_UNITS)))
#endif


__kernel void S119 (__global DTYPE* restrict AA,
										__global DTYPE* restrict BB
#ifdef FPGA_SINGLE
										,const int llX
                    ,const int llY)
#else
																	)
#endif
{

#ifdef GPU
	const int gidX = get_global_id(0);
  const int gidY = get_global_id(1);

	const int sizeX = get_global_size(0);
	const int sizeY = get_global_size(1);

	if (gidX == 0 || gidY == 0) {
		if (gidX != sizeX-1 && gidY != sizeY-1) {
    	int i = 1;
      int j = 1;
			for (i = 1, j = 1; i < sizeX && j < sizeY; i++, j++) {
				AA[i*sizeY+j] = AA[(i-1)*sizeY+(j-1)] + BB[i*sizeY+j];
			}
		}
	}
	
#endif

#ifdef FPGA_NDRANGE
	const int gidX = get_global_id(0);
  const int gidY = get_global_id(1);

	const int sizeX = get_global_size(0);
  const int sizeY = get_global_size(1);

	if (gidX == 0 || gidY == 0) {
		if (gidX != sizeX-1 && gidY != sizeY-1 ) {
    	int i = 1;
      int j = 1;
      for (i = 1, j = 1; i < sizeX && j < sizeY; i++, j++) {
				AA[i*sizeY+j] = AA[(i-1)*sizeY+(j-1)] + BB[i*sizeY+j];
			}
		}
	}
#endif

#ifdef FPGA_SINGLE

  for (int i = 1; i < lllX; i++) {
  	#pragma ivdep
    #pragma unroll UNROLL_FACTOR
  	for (int j = 1; j < lllY; j++) {
			AA[i*lllY+j] = AA[(i-1)*lllY+(j-1)] + BB[i*lllY+j];
		}
  }
#else


#endif

}
