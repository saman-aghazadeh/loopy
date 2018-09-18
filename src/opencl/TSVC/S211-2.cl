//
// (c) August 23, 2018 Saman Biookaghazadeh @ Arizona State University
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

#ifdef FPGA_SINGLE

#if UNROLL_FACTOR == 1
channel float c0 __attribute__((depth(4)));
#elif UNROLL_FACTOR == 2
channel float c0 __attribute__((depth(4)));
channel float c1 __attribute__((depth(4)));
#elif UNROLL_FACTOR == 4
channel float c0 __attribute__((depth(4)));
channel float c1 __attribute__((depth(4)));
channel float c2 __attribute__((depth(4)));
channel float c3 __attribute__((depth(4)));
#elif UNROLL_FACTOR == 8
channel float c0 __attribute__((depth(4)));
channel float c1 __attribute__((depth(4)));
channel float c2 __attribute__((depth(4)));
channel float c3 __attribute__((depth(4)));
channel float c4 __attribute__((depth(4)));
channel float c5 __attribute__((depth(4)));
channel float c6 __attribute__((depth(4)));
channel float c7 __attribute__((depth(4)));
#endif

#endif


__kernel void S211K1 (__global DTYPE* restrict A,
#ifdef FPGA_SINGLE
					__global DTYPE* restrict B,
#else
					__global const DTYPE* restrict B,
#endif
                    __global DTYPE* restrict BPrime,
                    __global const DTYPE* restrict C,
                    __global const DTYPE* restrict D,
                    __global const DTYPE* restrict E
#if FPGA_SINGLE
										,const int lll)
#else
										)
#endif

{

#ifdef GPU
	const int gid = get_global_id(0);
	const int size = get_global_size(0);
  const int index = gid+1;

	BPrime[index] = B[index+1] - E[index] * D[index];
	
#endif

#ifdef FPGA_NDRANGE
	const int gid = get_global_id(0);
  cosnt int size = get_global_size(0);
	const int index = gid+1;

	BPrime[index] = B[index+1] - E[index] * D[index];

#endif

#ifdef FPGA_SINGLE

#if UNROLL_FACTOR == 1
#pragma ivdep
for (int i = 1; i < (lll-1); i++) {
	DTYPE temp = B[i+1] - E[i] * D[i];
	write_channel_altera (c0, temp);
}
#elif UNROLL_FACTOR == 2
#pragma ivdep
for (int i = 1; i < (lll-1); i+=2) {
	DTYPE temp0 = B[i+1] - E[i] * D[i];
  	DTYPE temp1 = B[i+2] - E[i+1] * D[i+1];

	write_channel_altera (c0, temp0);
  	write_channel_altera (c1, temp1);
}
#elif UNROLL_FACTOR == 4
#pragma ivdep
for (int i = 1; i < (lll-1); i+=4) {
	DTYPE temp0 = B[i+1] - E[i] * D[i];
  	DTYPE temp1 = B[i+2] - E[i+1] * D[i+1];
	DTYPE temp2 = B[i+3] - E[i+2] * D[i+2];
 	DTYPE temp3 = B[i+4] - E[i+3] * D[i+3];

	write_channel_altera (c0, temp0);
  	write_channel_altera (c1, temp1);
  	write_channel_altera (c2, temp2);
  	write_channel_altera (c3, temp3);
}

#elif UNROLL_FACTOR == 8
#pragma ivdep
for (int i = 1; i < (lll-1); i+=8) {
	DTYPE temp0 = B[i+1] - E[i] * D[i];
  	DTYPE temp1 = B[i+2] - E[i+1] * D[i+1];
	DTYPE temp2 = B[i+3] - E[i+2] * D[i+2];
 	DTYPE temp3 = B[i+4] - E[i+3] * D[i+3];
	DTYPE temp4 = B[i+5] - E[i+4] * D[i+4];
	DTYPE temp5 = B[i+6] - E[i+5] * D[i+5];
  	DTYPE temp6 = B[i+7] - E[i+6] * D[i+6];
	DTYPE temp7 = B[i+8] - E[i+7] * D[i+7];

	write_channel_intel (c0, temp0);
  	write_channel_intel (c1, temp1);
  	write_channel_intel (c2, temp2);
  	write_channel_intel (c3, temp3);
	write_channel_intel (c4, temp0);
  	write_channel_intel (c5, temp1);
  	write_channel_intel (c6, temp2);
  	write_channel_intel (c7, temp3);

}

#endif

#endif

}

__kernel void S211K2 (__global DTYPE* restrict A,
#ifdef FPGA_SINGLE
					__global DTYPE* restrict B,
#else
					__global const DTYPE* restrict B,
#endif
                    __global DTYPE* restrict BPrime,
                    __global const DTYPE* restrict C,
                    __global const DTYPE* restrict D,
                    __global const DTYPE* restrict E
#if FPGA_SINGLE
										,const int lll)
#else
										)
#endif

{

#ifdef GPU
	const int gid = get_global_id(0);
	const int size = get_global_size(0);
  const int index = gid+1;

	A[index] = BPrime[index-1] - E[index] * D[index];
	
#endif

#ifdef FPGA_NDRANGE
	const int gid = get_global_id(0);
  cosnt int size = get_global_size(0);
	const int index = gid+1;

	A[index] = BPrime[index-1] - E[index] * D[index];

#endif

#ifdef FPGA_SINGLE

	DTYPE temp;
  temp = B[0];
  A[1] = temp + C[1] * D[1];

#if UNROLL_FACTOR == 1
	#pragma ivdep
	for (int i = 2; i < lll; i++) {
  		DTYPE temp;
    	temp = read_channel_altera(c0);
		A[i] = temp + C[i] * D[i];
	}
#elif UNROLL_FACTOR == 2
	#pragma ivdep
	for (int i = 2; i < lll; i+=2) {
		DTYPE temp0;
    	DTYPE temp1;
   		temp0 = read_channel_altera(c0);
		temp1 = read_channel_altera(c1);

		A[i] = temp0 + C[i] * D[i];
    	A[i+1] = temp1 + C[i+1] * D[i+1];
	}
#elif UNROLL_FACTOR == 4
	#pragma ivdep
	for (int i = 2; i < lll; i+=4) {
		DTYPE temp0;
 	    DTYPE temp1;
    	DTYPE temp2;
    	DTYPE temp3;

   		temp0 = read_channel_altera(c0);
		temp1 = read_channel_altera(c1);
		temp2 = read_channel_altera(c2);
		temp3 = read_channel_altera(c3);

		A[i] = temp0 + C[i] * D[i];
    	A[i+1] = temp1 + C[i+1] * D[i+1];
    	A[i+2] = temp2 + C[i+2] * D[i+2];
    	A[i+3] = temp3 + C[i+3] * D[i+3];
	}
#elif UNROLL_FACTOR == 8
	#pragma ivdep
	for (int i = 2; i < lll; i+=8) {
		DTYPE temp0;
    	DTYPE temp1;
    	DTYPE temp2;
    	DTYPE temp3;
    	DTYPE temp4;
    	DTYPE temp5;
    	DTYPE temp6;
    	DTYPE temp7;

   		temp0 = read_channel_intel(c0);
		temp1 = read_channel_intel(c1);
		temp2 = read_channel_intel(c2);
		temp3 = read_channel_intel(c3);
    	temp4 = read_channel_intel(c4);
    	temp5 = read_channel_intel(c5);
    	temp6 = read_channel_intel(c6);
    	temp7 = read_channel_intel(c7);

		A[i] = temp0 + C[i] * D[i];
    	A[i+1] = temp1 + C[i+1] * D[i+1];
    	A[i+2] = temp2 + C[i+2] * D[i+2];
    	A[i+3] = temp3 + C[i+3] * D[i+3];
    	A[i+4] = temp4 + C[i+4] * D[i+4];
    	A[i+5] = temp5 + C[i+5] * D[i+5];
    	A[i+6] = temp6 + C[i+6] * D[i+6];
    	A[i+7] = temp7 + C[i+7] * D[i+7];
	}
#endif

#endif

}
