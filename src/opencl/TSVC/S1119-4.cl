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

	/*
	int exit = lllY / BLOCK_SIZE;

	for (int i = 0; i < exit; i++) {

		int i_real = i*BLOCK_SIZE;

		//DTYPE AA_SR[BLOCK_SIZE][2];
		DTYPE AA_SR[BLOCK_SIZE];
		// initialize shift registers
  		#pragma unroll
  		for (int j = 0; j < BLOCK_SIZE; j++) {
//    		for (int k = 0; k < 2; k++) {
//				AA_SR[j][k] = 0.0f;
//      	}
			AA_SR[j] = 0.0f;
		}

		for (int j = 0; j < BLOCK_SIZE; j++) {
			//AA_SR[j][1] = AA[i_real+j];
			AA_SR[j] = AA[i_real+j];
		}

		// start processing
    	for (int j = 1; j < lllX; j++) {

		
			DTYPE BB_SR[BLOCK_SIZE];
			DTYPE AA_SR_INTER[BLOCK_SIZE];

//			#pragma unroll
//			for (int k = 0; k < BLOCK_SIZE; k++) {
//				AA_SR[k][0] = AA_SR[k][1];	
//			}
	
			#pragma ivdep
			#pragma unroll
			for (int k = 0; k < BLOCK_SIZE; k++) {
				BB_SR[k] = BB[j*lllY+k+i_real];
			}
		
    		#pragma ivdep
      		#pragma unroll
			for (int k = 0; k < BLOCK_SIZE; k++) {
				//AA_SR[k][1] = AA_SR[k][0] * BB_SR[k];
				//AA_SR_INTER[k] = AA_SR[k][0] * BB_SR[k];
				AA_SR_INTER[k] = AA_SR[k] * BB_SR[k];
			}

			#pragma unroll
			for (int k = 0; k < BLOCK_SIZE; k++) {
				//AA_SR[k][1] = AA_SR_INTER[k];
				AA_SR[k] = AA_SR_INTER[k];
			}

			#pragma unroll
			for (int k = 0; k < BLOCK_SIZE; k++) {
				AA[j*lllY+k+i_real] = AA_SR_INTER[k];
			}
		}
	
	}
	*/

	int exit = lllY / BLOCK_SIZE;

	for (int i = 0; i < exit; i+=2) {

		int i_real[2];

		i_real[0] = i*BLOCK_SIZE;
		i_real[1] = (i+1)*BLOCK_SIZE;
		//i_real[2] = (i+2)*BLOCK_SIZE;
		//i_real[3] = (i+3)*BLOCK_SIZE;

//		DTYPE AA_SR[BLOCK_SIZE][2];
		// initialize shift registers
//  		#pragma unroll
//  		for (int j = 0; j < BLOCK_SIZE; j++) {
//    		for (int ii = 0; ii < 2; ii++) {
//				AA_SR[j][ii] = 0.0f;
//      		}
//		}

		//#pragma unroll
//		for (int j = 0; j < BLOCK_SIZE; j++) {
//			for (int ii = 0; ii < 2; ii++) {
//				AA_SR[j][ii] = AA[i_real[ii]+j];
//			}
//		}

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


			//#pragma unroll
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
					AA_SR[ii][k] = AA_SR[ii][k] * BB_SR[ii][k];
				}

	//			#pragma unroll
	//			for (int k = 0; k < BLOCK_SIZE; k++) {
	//				//AA_SR[k][1] = AA_SR_INTER[k][ii];
	//				AA_SR[k][ii] = AA_SR_INTER[k][ii];
	//			}

				#pragma unroll
				for (int k = 0; k < BLOCK_SIZE; k++) {
					AA[j*lllY+k+i_real[ii]] = AA_SR[ii][k];
				}
			}
		}
	
	}


#endif

}
