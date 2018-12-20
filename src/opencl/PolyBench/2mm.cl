//
// (c) December 19, 2018 Saman Biookaghazadeh @ Arizona State University
//

#ifdef INT_PRECISION
#define DTYPE int
#elif SINGLE_PRECISION
#define DTYPE float
#elif DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define DTYPE double
#endif

#include "funcs.h"

channel DTYPE c0;

__kernel void mm_k1 (__global const DTYPE* restrict A,
                   __global const DTYPE* restrict B,
                   __global const DTYPE* restrict C,
                   __global DTYPE* restrict D,
                   const DTYPE alpha,
                   const DTYPE beta
#ifdef FPGA_SINGLE
                   ,const int lll)
#else
                   ,const int lll
                   ,__global DTYPE* restrict temp)
#endif
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
      const int numTiles = lll/32;
      for (int t = 0; t < numTiles; t++) {

          // Load one tile of A and B into local memory
          const int tiledRow = 32*t + row;
          const int tiledCol = 32*t + col;
          Asub[col][row] = A[tiledCol*M + globalRow];
          Bsub[col][row] = B[globalCol*K + tiledRow];

          // Synchronize to make sure the tile is loaded
          barrier(CLK_LOCAL_MEM_FENCE);

          // Perform the computation for a single tile
          for (int k = 0; k < 32; k++) {
              acc += Asub[k][row] * Bsub[col][k];
          }

          // Synchronization before loading the next tile
          barrier(CLK_LOCAL_MEM_FENCE);
      }

      // Store the final result in C
      temp[globalCol*M + globalRow] = acc;

#endif

#ifdef FPGA_SINGLE

       for (int i = 0; i < lll; i++) {
           for (int j = 0; j < lll; j++) {
			   DTYPE temp = 0.0;
			   for (int z = 0; z < lll/BLOCK_SIZE; z++) {
			   		DTYPE A_local[BLOCK_SIZE];
			   		DTYPE B_local[BLOCK_SIZE];
					DTYPE local_temp = 0.0;
					
					// Coalescing memory read from the memory section "A"
					#pragma unroll
					for (int k = 0; k < BLOCK_SIZE; k++) {
						A_local[k] = A[i*lll+z*BLOCK_SIZE+k];
					}

					// Coalescing memory read from the memory section "B"
					#pragma unroll
					for (int k = 0; k < BLOCK_SIZE; k++) {
						B_local[k] = B[i*lll+z*BLOCK_SIZE+k];
					}

					// Accumulating the result of multiplications
					#pragma unroll
               		for (int k = 0; k < BLOCK_SIZE; k++) {
                   		local_temp += A[k] * B[k] * alpha;
               		}
		
					// final accumulation
					temp += local_temp;
			   }
               write_channel_altera (c0, temp);
           }
       }

#endif

}

__kernel void mm_k2 (__global const DTYPE* restrict A,
                      __global const DTYPE* restrict B,
                      __global const DTYPE* restrict C,
                      __global DTYPE* restrict D,
                      const DTYPE alpha,
                      const DTYPE beta
#ifdef FPGA_SINGLE
                      ,const int lll)
#else
                      ,const int lll
                      ,__global DTYPE* restrict temp)
#endif
{

#ifdef GPU

       const int row = get_local_id(0);
       const int col = get_local_id(1);
       const int globalRow = 32 * get_group_id(0) + row;
       const int globalCol = 32 * get_group_id(1) + col;

       // Local memory to fit a tile of 32*32 elements of A and B
       __local Tsub[32][32];
       __local Csub[32][32];

       // Initialize the accumulation register
       float acc = 0.0f;

       // Loop over all tiles
       const int numTiles = lll/32;
       for (int t = 0; t < numTiles; t++) {

           // Load one tile of A and B into local memory
           const int tiledRow = 32*t + row;
           const int tiledCol = 32*t + col;
           Tsub[col][row] = temp[tiledCol*M + globalRow];
           Csub[col][row] = C[globalCol*K + tiledRow];

           // Synchronize to make sure the tile is loaded
           barrier (CLK_LOCAL_MEM_FENCE);

           // Perform the computation for a single tile
           for (int k = 0; k < 32; k++) {
               acc += Asub[k][row] * Bsub[col][k];
           }

           // Synchronization before loading the next tile
           barrier (CLK_LOCAL_MEM_FENCE);
       }

       D[globalCol*M + globalRow] = acc;

#endif

#ifdef FPGA_SINGLE
       for (int i = 0; i < lll; i++) {
           for (int j = 0; j < lll; j++) {
               DTYPE temp = read_channel_altera(c0);

			         #pragma ivdep	
			         for (int z = 0; z < lll/BLOCK_SIZE; z++) {
					         DTYPE C_local[BLOCK_SIZE];
					         DTYPE D_local[BLOCK_SIZE];
				
                   // Coalescing memory read from the memory section "A"
					         #pragma unroll
					         for (int k = 0; k < BLOCK_SIZE; k++) {
						           C_local[k] = C[j*lll+z*BLOCK_SIZE+k];
					         }
					
                   // Initializing the memory section "D"
					         #pragma unroll
					         for (int k = 0; k < BLOCK_SIZE; k++) {
						           D_local[k] = 0.0;
					         }

					         // Accumulating the result of multiplications
					         #pragma unroll
               		 for (int k = 0; k < BLOCK_SIZE; k++) {
                       D_local[k] += temp * C_local[k];
               		 }

					         // final accumulation
					         #pragma unroll
					         for (int k = 0; k < BLOCK_SIZE; k++) {
						           D[j*lll+z*BLOCK_SIZE+k] += D_local[k];
					         }

			         }
				
           }
       }
#endif
  
}
