//
// (c) Aug 7, 2018 Saman Biookaghazadeh @ Arizona State University
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
										,const int lllX
                    ,const int lllY)
#else
										)
#endif
{

#ifdef GPU

	const int gidX = get_global_id(0);
  const int gidY = get_global_id(1);
  const int lidX = get_local_id(0);
  const int lidY = get_local_id(1);

	const int sizeX = get_global_size(0);
  const int sizeY = get_global_size(1);

	DTYPE temp1 = gidX + gidY;
  DTYPE temp2 = lidX + lidY;

	if (gidX == 0 || gidY == 0) {
  	if (gidX != sizeX-1 && gidY != sizeY-1) {
			int i = 1;
    	int j = 1;

			while (i < sizeX && j < sizeY) {
				temp1 = temp1 + temp2;
        i++; j++;
			}

			AA[gidX * sizeY + gidY] = temp1;
    }
	}

#endif

#ifdef FPGA_NDRANGE

	const int gidX = get_global_id(0);
  const int gidY = get_global_id(1);
	const int lidX = get_local_id(0);
  const int lidY = get_local_id(1);

	const int sizeX = get_global_size(0);
	const int sizeY = get_global_size(1);

	DTYPE temp1 = gidX + gidY;
  DTYPE temp2 = lidX + lidY;

	if (gidX == 0 || gidY == 0) {
		if (gidX != sizeX-1 && gidY != sizeY-1) {
			int i = 1;
      int j = 1;

			while (i < sizeX && j < sizeY) {
				temp1 = temp1 + temp2;
        i++; j++;
			}

			AA[gidX * sizeY + gidY] = temp1;
		}
	}

#endif

#ifdef FPGA_SINGLE

	DTYPE temp1 = lllX;
  DTYPE temp2 = lllY;

	for (int i = 1; i < lllX; i++) {
  	//#pragma ivdep
    #pragma unroll UNROLL_FACTOR
		for (int j = 1; j < lllY; j++) {
			temp1 = temp1 + temp2;
		}

		AA[lllX+lllY] = temp1;
		temp2++;
	}

#endif

}
