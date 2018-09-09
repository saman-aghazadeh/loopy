//
// (c) August 1, 2018 Saman Biookaghazadeh @ Arizona State University
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
  	AA[i*size+gid] = AA[(i-1)*size+gid] + BB[i*size+gid];
	}
	
#endif

#ifdef FPGA_NDRANGE
	const int gid = get_global_id(0);
	const int size = get_global_size(0);

	#pragma unroll UNROLL_FACTOR
	for (int i = 1; i < lllX; i++) {
		AA[i*size+gid] = AA[(i-1)*size+gid] + BB[i*size+gid];
	}
#endif

#ifdef FPGA_SINGLE

  for (int i = 1; i < lllX; i++) {

		int exit = (lllY % UNROLL_FACTOR == 0) ? (lllY / UNROLL_FACTOR) : (lllY / UNROLL_FACTOR) + 1;

		#pragma ivdep
    for (int j = 0; j < exit; j++) {

			float a[UNROLL_FACTOR];

			#pragma unroll
      for (int k = 0; k < UNROLL_FACTOR; k++) {
				int j_real = j * UNROLL_FACTOR;
        a[k] = AA[(i-1)*lllY + j_real];
			}

			#pragma unroll
      for (int k = 0; k < UNROLL_FACTOR; k++) {
				int j_real = j * UNROLL_FACTOR + k;
       	if (j_real < lllY) {
					AA[i*lllY+j_real] = a[k] + BB[i*lllY+j_real];
				}
			}
		}
	}
#endif

}
