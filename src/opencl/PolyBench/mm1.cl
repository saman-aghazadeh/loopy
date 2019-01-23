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

__kernel void mm (__global const DTYPE* restrict A,
                  __global const DTYPE* restrict B,
                  __global DTYPE* restrict C,
                  const DTYPE alpha,
                  const DTYPE beta,
                  const int lllX,
                  const int lllY)
{

#ifdef GPU

  const int row = get_local_id(0);
  const int col = get_local_id(1);
	const int globalRow = 32 * get_group_id(0) + row;
  const int globalCol = 32 * get_group_id(1) + col;

  // Local memory to fit a tile of 32*32 elements of A and B
  __local float Asub[32][32];
  __local float Bsub[32][32];

  // Initialize the accumulation register
  float acc = 0.0f;

  // Loop over all tiles
  const int numTiles = lllY/32;
	for (int t = 0; t < numTiles; t++) {

    // Load one tile of A and B into local memory
    const int tiledRow = 32*t + row;
    const int tiledCol = 32*t + col;
    Asub[row][col] = A[globalRow*lllY + tiledCol];
    Bsub[row][col] = B[globalRow*lllY + tiledCol];

    // Synchronize to make sure the tile loaded
    barrier (CLK_LOCAL_MEM_FENCE);

    // Perform the computation for a single tile
    for (int k = 0; k < 32; k++) {
#if INTENSITY1
			megaBfunction(acc, Asub[row][k], Bsub[k][col]);
#elif INTENSITY2
			megaBfunction2(acc, Asub[row][k], Bsub[k][col]);
#elif INTENSITY3
			megaBfunction3(acc, Asub[row][k], Bsub[k][col]);
#elif INTENSITY4
			megaBfunction4(acc, Asub[row][k], Bsub[k][col]);
#elif INTENSITY5
			megaBfunction5(acc, Asub[row][k], Bsub[k][col]);
#endif

    }

    barrier (CLK_LOCAL_MEM_FENCE);

  }

  C[globalRow*lllX + globalCol] = acc;

#endif


#ifdef FPGA_SINGLE

	for (int i = 0; i < lllX; i++) {
    int iIndex = i*lllY;
		for (int j = 0; j < lllX; j++) {
      int jIndex = j*lllY;
      DTYPE temp = 0.0f;
      #pragma ivdep
      for (int z = 0; z < lllY/BLOCK_SIZE; z++) {
        int zIndex = z * BLOCK_SIZE;
        DTYPE A_local[BLOCK_SIZE];
        DTYPE B_local[BLOCK_SIZE];
        DTYPE local_temp = 0.0f;

        // Coalescing memory read from the memory section "A"

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
          A_local[k] = A[iIndex + zIndex + k];
        }

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
          B_local[k] = B[jIndex + zIndex + k];
        }

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
#if INTENSITY1
					megaCfunction(local_temp, A_local[k], B_local[k], alpha);
#elif INTENSITY2
					megaCfunction2(local_temp, A_local[k], B_local[k], alpha);
#elif INTENSITY3
					megaCfunction3(local_temp, A_local[k], B_local[k], alpha);
#elif INTENSITY4
					megaCfunction4(local_temp, A_local[k], B_local[k], alpha);
#elif INTENSITY5
					megaCfunction5(local_temp, A_local[k], B_local[k], alpha);
#endif
        }

        temp += local_temp;
      }

      C[i*lllX+j] = temp;
    }

  }

#endif

}
