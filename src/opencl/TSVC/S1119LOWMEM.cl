//
// (c) August 16, 2018 Saman Biookaghazadeh @ Arizona State University
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

#if INTENSITY1
		Bfunction(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY2
		Bfunction2(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY3
		Bfunction3(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY4
		Bfunction4(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY5
		Bfunction5(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY6
		Bfunction6(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY7
		Bfunction7(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY8
		Bfunction8(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#endif
	}	
#endif


#ifdef FPGA_NDRANGE
	const int gid = get_global_id(0);
	const int size = get_global_size(0);

	#pragma unroll UNROLL_FACTOR
	for (int i = 1; i < lllX; i++) {

#if INTENSITY1
		Bfunction(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY2
		Bfunction2(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY3
		Bfunction3(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY4
		Bfunction4(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY5
		Bfunction5(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY6
		Bfunction6(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY7
		Bfunction7(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#elif INTENSITY8
		Bfunction8(AA[i*size+gid], AA[(i-1)*size+gid], BB[i*size+gid]);
#endif
	}
#endif

#ifdef FPGA_SINGLE

	int exit = lllY / BLOCK_SIZE;

	for (int i = 0; i < exit; i++) {

		int i_real = i * BLOCK_SIZE;

		float AA_SR[BLOCK_SIZE][2];

		// initialize shift registers
		#pragma unroll
   	for (int j = 0; j < BLOCK_SIZE; j++) {
			for (int k = 0; k < 2; k++) {
				AA_SR[j][k] = 0.0f;
			}
		}

		#pragma unroll
   	for (int j = 0; j < BLOCK_SIZE; j++) {
			AA_SR[j][1] = AA[i_real+j];
		}

		// start processing
    for (int j = 1; j < lllX; j++) {

			#pragma unroll
     	for (int k = 0; k < BLOCK_SIZE; k++) {
				AA_SR[j][0] = AA_SR[j][0];
			}

			#pragma ivdep
     	#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; k++) {
#if INTENSITY1
				Bfunction(AA_SR[k][1], AA_SR[k][0], BB[j*lllY+k+i_real]);
#elif INTENSITY2
				Bfunction2(AA_SR[k][1], AA_SR[k][0], BB[j*lllY+k+i_real]);
#elif INTENSITY3
				Bfunction3(AA_SR[k][1], AA_SR[k][0], BB[j*lllY+k+i_real]);
#elif INTENSITY4
				Bfunction4(AA_SR[k][1], AA_SR[k][0], BB[j*lllY+k+i_real]);
#elif INTENSITY5
				Bfunction5(AA_SR[k][1], AA_SR[k][0], BB[j*lllY+k+i_real]);
#elif INTENSITY6
				Bfunction6(AA_SR[k][1], AA_SR[k][0], BB[j*lllY+k+i_real]);
#elif INTENSITY7
				Bfunction7(AA_SR[k][1], AA_SR[k][0], BB[j*lllY+k+i_real]);
#elif INTENSITY8
				Bfunction8(AA_SR[k][1], AA_SR[k][0], BB[j*lllY+k+i_real]);
#endif
			}

			#pragma unroll
	    for (int k = 0; k < BLOCK_SIZE; k++) {
				AA[j*lllY+k+i_real] = AA_SR[k][1];
			}
		}
	}

#endif

}
