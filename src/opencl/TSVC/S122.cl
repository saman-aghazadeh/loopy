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


__kernel void S122 (__global DTYPE* restrict AA,
										__global const DTYPE* restrict BB,
                    const int stride,
                    const int start,
                   	const int elems,
										const int lll)
                    
{

#ifdef GPU
	const int gid = get_global_id(0);
	const int size = get_global_size(0);

	if (gid < elems) {
		const int index = start + gid * stride;
    AA[index] += BB[lll-(gid+1)];
	}
	
#endif

#ifdef FPGA_NDRANGE
	const int gid = get_global_id(0);
  const int size = get_global_size(0);

	if (gid < elems) {
		const int index = start + gid * stride;
    AA[index] += BB[lll-(gid+1)];
	}
#endif

#ifdef FPGA_SINGLE

	int j = 1;
  int k = 0;
	for (int i = start; i < lll; i += stride) {
		k += j;
    AA[i] += BB[lll-k];
	}

#else


#endif

}
