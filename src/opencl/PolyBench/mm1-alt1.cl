//
// (c) January 9, 2019 Saman Biookaghazadeh @ Arizona State University
//

#ifdef INT_PRECISION
#define DTYPE int
#elif SINGLE_PRECISION
#define DTYPE float
#elif DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define DTYPE double
#endif

#include "../TSVC/funcs.h"


__kernel
#ifdef FPGA_NDRANGE
__attribute((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
__attribute((num_simd_work_items(SIMD_WORK_ITEMS)))
#endif
void mm (__global const DTYPE* restrict A,
                  __global const DTYPE* restrict B,
                  __global DTYPE* restrict C,
                  int lllX,
                  int lllY)

{

#ifdef GPU

#endif

#ifdef FPGA_NDRANGE

	__local DTYPE A_local[BLOCK_SIZE][BLOCK_SIZE];
  __local DTYPE B_local[BLOCK_SIZE][BLOCK_SIZE];

  // Block index
  int block_x = get_group_id(0);
  int block_y = get_group_id(1);

  // local ID index (offset withing a block)
  int local_x = get_local_id(0);
  int local_y = get_local_id(1);

  // Compute loop bounds
  int a_start = lllX * BLOCK_SIZE * block_y;
  int a_end = a_start + lllX - 1;
  int b_start = BLOCK_SIZE * block_x;

  DTYPE running_sum = 0;

	for (int a = a_start, b = b_start; a <= a_end; a += BLOCK_SIZE, b += (BLOCK_SIZE * lllY)) {

		A_local[local_y][local_x] = A[a + lllX * local_y + local_x];
    B_local[local_x][local_y] = B[b + lllY * local_y + local_x];

    barrier (CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      running_sum += A_local[local_y][k] * B_local[local_x][k];
    }

		barrier (CLK_LOCAL_MEM_FENCE);
  }

  C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = running_sum;

#endif

}
