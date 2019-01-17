//
// (c) January 14, 2019 Saman Biookaghazadeh @ Arizona State University
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

__kernel void floydWarshall (__global const DTYPE* restrict M,
                             __global const DTYPE* restrict M_T,
                             __global DTYPE* restrict M_out,
                             const int lll) {

#ifdef GPU

#endif

#ifdef FPGA_SINGLE

	for (int i = 0; i < lll; i++) {
    for (int j1 = 0; j1 < lll/BLOCK_SIZE; j1++) {

      DTYPE items[BLOCK_SIZE];

      #pragma unroll
      for (int j2 = 0; j2 < BLOCK_SIZE; j2++) {
				items[j2] = M[i*lll + j1*BLOCK_SIZE + j2];
      }

      for (int j2 = 0; j2 < BLOCK_SIZE; j2++) {

      	int j = j1 * BLOCK_SIZE + j2;

      	for (int k = 0; k < lll/BLOCK_SIZE; k++) {
        	DTYPE M_temp[BLOCK_SIZE];
        	DTYPE M_T_temp[BLOCK_SIZE];
        	DTYPE M_out_temp[BLOCK_SIZE];

        	#pragma unroll
        	for (int p = 0; p < BLOCK_SIZE; p++) {
          	M_temp[p] = M[i*lll + k*BLOCK_SIZE + p];
        	}

        	#pragma unroll
        	for (int p = 0; p < BLOCK_SIZE; p++) {
          	M_T_temp[p] = M_T[j*lll + k*BLOCK_SIZE + p];
        	}

        	for (int p = 0; p < BLOCK_SIZE; p++) {

						DTYPE dist = 0.1;

#if INTENSITY1
						megaBfunction (dist, M_temp[p], M_T_temp[p]);
#elif INTENSITY2
						megaBfunction2 (dist, M_temp[p], M_T_temp[p]);
#elif INTENSITY3
						megaBfunction3 (dist, M_temp[p], M_T_temp[p]);
#elif INTENSITY4
						megaBfunction4 (dist, M_temp[p], M_T_temp[p]);
#elif INTENSITY5
						megaBfunction5 (dist, M_temp[p], M_T_temp[p]);
#endif

          	if (dist > items[j2]) {
            	M_out_temp[p] = items[j2];
          	} else {
            	M_out_temp[p] = dist;
          	}
        	}

        	#pragma unroll
        	for (int p = 0; p < BLOCK_SIZE; p++) {
          	M_out[i*lll + k*BLOCK_SIZE + p] = M_out_temp[p];
        	}
        }

      }
    }
  }

#endif

}
