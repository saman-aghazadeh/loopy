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

channel DTYPE c0;

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
  const int gloBalCol = 32 * get_group_id(1) + col;

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
    Asub[col][row] = A[tiledCol*M + globalRow];
    Bsub[col][row] = B[globalCol*K + tiledRow];

    // Synchronize to make sure the tile loaded
    barrier (CLK_LOCAL_MEM_FENCE);

    // Perform the computation for a single tile
    for (int k = 0; k < 32; k++) {
#if INTENSITY1
			megaBfunction1(acc, Asub[k][row], Bsub[col][k]);
#elif INTENSITY2
			megaBfunction2(acc, Asub[k][row], Bsub[col][k]);
#elif INTENSITY3
			megaBfunction3(acc, Asub[k][row], Bsub[col][k]);
#elif INTENSITY4
			megaBfunction4(acc, Asub[k][row], Bsub[col][k]);
#elif INTENSITY5
			megaBfunction5(acc, Asub[k][row], Bsub[col][k]);
#endif

    }

    barrier (CLK_LOCAL_MEM_FENCE);

  }

  C[globalCol*M + globalRow] = acc;

#endif


#ifdef FPGA_SINGLE

	for (int i = 0; i < lllX; i++) {
		for (int j = 0; j < lllX; j++) {
      DTYPE temp = 0.0f;
      #pragma ivdep
      for (int z = 0; z < lllY/BLOCK_SIZE; z++) {
        DTYPE A_local[BLOCK_SIZE];
        DTYPE B_local[BLOCK_SIZE];
        DTYPE local_temp = 0.0f;

        // Coalescing memory read from the memory section "A"

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
          A_local[k] = A[i*lllY+z*BLOCK_SIZE+k];
        }

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
          B_local[k] = B[j*lllY+z*BLOCK_SIZE+k];
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

      C[i*BLOCK_SIZE+j] = temp;
    }

  }

#endif

}
