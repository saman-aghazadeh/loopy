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


#include "funcs.h"

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

	int temp;
	temp = AA[gid];
	for (int i = 1; i < lllX; i++) {
#if INTENSITY1
		megaBfunction(temp, temp, BB[i*size+gid]);
    AA[i*size+gid] = temp;
#elif INTENSITY2
		megaBfunction2(temp, temp, BB[i*size+gid]);
    AA[i*size+gid] = temp;
#elif INTENSITY3
		megaBfunction3(temp, temp, BB[i*size+gid]);
    AA[i*size+gid] = temp;
#elif INTENSITY4
		megaBfunction4(temp, temp, BB[i*size+gid]);
    AA[i*size+gid] = temp;
#elif INTENSITY5
		megaBfunction5(temp, temp, BB[i*size+gid]);
#endif
	}
	
#endif

#ifdef FPGA_SINGLE


	int exit = lllY / BLOCK_SIZE;
	int i = 0;

	while (i < exit) {

		int i_real[2];

		i_real[0] = i*BLOCK_SIZE;
		i_real[1] = (i+1)*BLOCK_SIZE;


		// start processing
    	for (int j = 1; j < lllX; j++) {

			DTYPE BB_SR[2][BLOCK_SIZE];
			DTYPE AA_SR[2][BLOCK_SIZE];

			if (j == 1) {
				#pragma unroll
				#pragma loop_coalesce
				for (int ii = 0; ii < 2; ii++) {
					#pragma unroll
					for (int k = 0; k < BLOCK_SIZE; k++)
						AA_SR[ii][k] = AA[i_real[ii]+k];
				}

			}


			#pragma unroll
			#pragma ivdep
			for (int ii = 0; ii < 2; ii++){
	
				#pragma ivdep
				#pragma unroll
				for (int k = 0; k < BLOCK_SIZE; k++) {
					BB_SR[ii][k] = BB[j*lllY+k+i_real[ii]];
				}
		
    			#pragma ivdep
      			#pragma unroll UNROLL_FACTOR
				for (int k = 0; k < BLOCK_SIZE; k++) {
					//AA_SR[k][1] = AA_SR[k][0] * BB_SR[k];
					//AA_SR_INTER[k] = AA_SR[k][0] * BB_SR[k];
#if INTENSITY1
					megaBfunction(AA_SR[ii][k], AA_SR[ii][k], BB_SR[ii][k]);
#elif INTENSITY2
					megaBfunction2(AA_SR[ii][k], AA_SR[ii][k], BB_SR[ii][k]);
#elif INTENSITY3
					megaBfunction3(AA_SR[ii][k], AA_SR[ii][k], BB_SR[ii][k]);
#elif INTENSITY4
					megaBfunction4(AA_SR[ii][k], AA_SR[ii][k], BB_SR[ii][k]);
#elif INTENSITY5
					megaBfunction5(AA_SR[ii][k], AA_SR[ii][k], BB_SR[ii][k]);
#endif
				}

				#pragma unroll
				for (int k = 0; k < BLOCK_SIZE; k++) {
					AA[j*lllY+k+i_real[ii]] = AA_SR[ii][k];
				}
			}
		}
		i+=2;
	
	}


#endif

}
