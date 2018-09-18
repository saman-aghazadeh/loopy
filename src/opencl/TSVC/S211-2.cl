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
typedef struct Msg {
	DTYPE temp0;
} msg;
#elif UNROLL_FACTOR == 2
typedef struct Msg {
	DTYPE temp0;
  DTYPE temp1;
} msg;
#elif UNROLL_FACTOR == 4
typedef struct Msg {
	DTYPE temp0;
  DTYPE temp1;
  DTYPE temp2;
  DTYPE temp3;
} msg;
#elif UNROLL_FACTOR == 8
typedef struct Msg {
	DTYPE temp0;
  DTYPE temp1;
  DTYPE temp2;
  DTYPE temp3;
  DTYPE temp4;
  DTYPE temp5;
  DTYPE temp6;
  DTYPE temp7;
} msg;
#endif
channel Msg c0 __attribute__((depth(4)));
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
	Msg msg;
  msg.temp0 = temp;
	write_channel_altera (c0, msg);
}
#elif UNROLL_FACTOR == 2
#pragma ivdep
for (int i = 1; i < (lll-1); i+=2) {
	DTYPE temp0 = B[i+1] - E[i] * D[i];
  	DTYPE temp1 = B[i+2] - E[i+1] * D[i+1];

	Msg msg;
  msg.temp0 = temp0;
  msg.temp1 = temp1;

	write_channel_altera (c0, msg);
}
#elif UNROLL_FACTOR == 4
#pragma ivdep
for (int i = 1; i < (lll-1); i+=4) {
	DTYPE temp0 = B[i+1] - E[i] * D[i];
  	DTYPE temp1 = B[i+2] - E[i+1] * D[i+1];
	DTYPE temp2 = B[i+3] - E[i+2] * D[i+2];
 	DTYPE temp3 = B[i+4] - E[i+3] * D[i+3];

	Msg msg;
  msg.temp0 = temp0;
  msg.temp1 = temp1;
  msg.temp2 = temp2;
  msg.temp3 = temp3;

	write_channel_altera (c0, msg);
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

	Msg msg;
  msg.temp0 = temp0;
  msg.temp1 = temp1;
  msg.temp2 = temp2;
  msg.temp3 = temp3;
  msg.temp4 = temp4;
  msg.temp5 = temp5;
  msg.temp6 = temp6;
  msg.temp7 = temp7;

	write_channel_altera (c0, msg);

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
    	MSG msg = read_channel_altera(c0);

		A[i] = msg.temp0 + C[i] * D[i];
	}
#elif UNROLL_FACTOR == 2
	#pragma ivdep
	for (int i = 2; i < lll; i+=2) {
   		Msg msg = read_channel_altera(c0);

		A[i] = msg.temp0 + C[i] * D[i];
    	A[i+1] = msg.temp1 + C[i+1] * D[i+1];
	}
#elif UNROLL_FACTOR == 4
	#pragma ivdep
	for (int i = 2; i < lll; i+=4) {
   		Msg msg = read_channel_altera(c0);

		A[i] = msg.temp0 + C[i] * D[i];
    	A[i+1] = msg.temp1 + C[i+1] * D[i+1];
    	A[i+2] = msg.temp2 + C[i+2] * D[i+2];
    	A[i+3] = msg.temp3 + C[i+3] * D[i+3];
	}
#elif UNROLL_FACTOR == 8
	#pragma ivdep
	for (int i = 2; i < lll; i+=8) {
   		Msg msg = read_channel_altera(c0);

		A[i] = msg.temp0 + C[i] * D[i];
    	A[i+1] = msg.temp1 + C[i+1] * D[i+1];
    	A[i+2] = msg.temp2 + C[i+2] * D[i+2];
    	A[i+3] = msg.temp3 + C[i+3] * D[i+3];
    	A[i+4] = msg.temp4 + C[i+4] * D[i+4];
    	A[i+5] = msg.temp5 + C[i+5] * D[i+5];
    	A[i+6] = msg.temp6 + C[i+6] * D[i+6];
	}
#endif

#endif

}
