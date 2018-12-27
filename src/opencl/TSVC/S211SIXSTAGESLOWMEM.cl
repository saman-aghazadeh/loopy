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

#if UNROLL_FACTOR == 1
#define VTYPE float
#elif UNROLL_FACTOR == 2
#define VTYPE float2
#elif UNROLL_FACTOR == 4
#define VTYPE float4
#elif UNROLL_FACTOR == 8
#define VTYPE float8
#elif UNROLL_FACTOR == 16
#define VTYPE float16
#endif

#include "funcs.h"

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
#elif UNROLL_FACTOR == 16
typedef struct Msg {
	DTYPE temp0;
  DTYPE temp1;
  DTYPE temp2;
  DTYPE temp3;
  DTYPE temp4;
  DTYPE temp5;
  DTYPE temp6;
  DTYPE temp7;
	DTYPE temp8;
  DTYPE temp9;
  DTYPE temp10;
  DTYPE temp11;
  DTYPE temp12;
  DTYPE temp13;
  DTYPE temp14;
  DTYPE temp15;
} msg;
#endif
channel VTYPE c0 __attribute__((depth(4)));
channel VTYPE c1 __attribute__((depth(4)));
channel VTYPE c2 __attribute__((depth(4)));
channel VTYPE c3 __attribute__((depth(4)));
channel VTYPE c4 __attribute__((depth(4)));
#endif


__kernel void S211K1 (__global DTYPE* restrict A,
					__global DTYPE* restrict B,
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

	DTYPE multiplier = 1.5;
	const int gid = get_global_id(0);
  const int index = gid+1;

#if INTENSITY1
	Bfunction (BPrime[index+8], B[index+9], multiplier);	
#elif INTENSITY2
	Bfunction2 (BPrime[index+8], B[index+9], multiplier);	
#elif INTENSITY3
	Bfunction3 (BPrime[index+8], B[index+9], multiplier);	
#elif INTENSITY4
	Bfunction4 (BPrime[index+8], B[index+9], multiplier);	
#elif INTENSITY5
	Bfunction5 (BPrime[index+8], B[index+9], multiplier);	
#endif

#endif


#ifdef FPGA_SINGLE

DTYPE multiplier = 1.5;

#if UNROLL_FACTOR == 1
#pragma ivdep
for (int i = 1; i < lll; i++) {

	VTYPE temp;

#if INTENSITY1
	BBfunction (temp, (VTYPE)(B[i+9]), multiplier);
#elif INTENSITY2
	BBfunction2 (temp, (VTYPE)(B[i+9]), multiplier);
#elif INTENSITY3
	megaBBfunction3 (temp, (VTYPE)(B[i+9]), multiplier);
#elif INTENSITY4
	megaBBfunction4 (temp, (VTYPE)(B[i+9]), multiplier);
#elif INTENSITY5
	megaBBfunction5 (temp, (VTYPE)(B[i+9]), multiplier);
#endif

	write_channel_altera (c0, temp);
}
#elif UNROLL_FACTOR == 2
#pragma ivdep
for (int i = 1; i < lll; i+=2) {
	VTYPE temp;

#if INTENSITY1
	BBfunction (temp, (VTYPE)(B[i+9],B[i+10]), multiplier);
#elif INTENSITY2
	BBfunction2 (temp, (VTYPE)(B[i+9],B[i+10]), multiplier);
#elif INTENSITY3
	BBfunction3 (temp, (VTYPE)(B[i+9],B[i+10]), multiplier);
#elif INTENSITY4
	BBfunction4 (temp, (VTYPE)(B[i+9],B[i+10]), multiplier);
#elif INTENSITY5
	BBfunction5 (temp, (VTYPE)(B[i+9],B[i+10]), multiplier);
#endif

	write_channel_altera (c0, temp);
}
#elif UNROLL_FACTOR == 4
#pragma ivdep
for (int i = 1; i < lll; i+=4) {
	VTYPE temp;

#if INTENSITY1
	BBfunction (temp, (VTYPE)(B[i+9],B[i+10],B[i+11],B[i+12]), multiplier);
#elif INTENSITY2
	BBfunction2 (temp, (VTYPE)(B[i+9],B[i+10],B[i+11],B[i+12]), multiplier);
#elif INTENSITY3
	BBfunction3 (temp, (VTYPE)(B[i+9],B[i+10],B[i+11],B[i+12]), multiplier);
#elif INTENSITY4
	BBfunction4 (temp, (VTYPE)(B[i+9],B[i+10],B[i+11],B[i+12]), multiplier);
#elif INTENSITY5
	BBfunction5 (temp, (VTYPE)(B[i+9],B[i+10],B[i+11],B[i+12]), multiplier);
#endif

	write_channel_altera (c0, temp);
}

#elif UNROLL_FACTOR == 8
#pragma ivdep
for (int i = 1; i < lll; i+=8) {
	VTYPE temp;

#if INTENSITY1
BBfunction (temp, (VTYPE)(B[i+9],B[i+10],B[i+11],B[i+12],B[i+13],B[i+14],B[i+15],B[i+16]), multiplier);
#elif INTENSITY2
BBfunction2 (temp, (VTYPE)(B[i+9],B[i+10],B[i+11],B[i+12],B[i+13],B[i+14],B[i+15],B[i+16]), multiplier);
#elif INTENSITY3
BBfunction3 (temp, (VTYPE)(B[i+9],B[i+10],B[i+11],B[i+12],B[i+13],B[i+14],B[i+15],B[i+16]), multiplier);
#elif INTENSITY4
BBfunction4 (temp, (VTYPE)(B[i+9],B[i+10],B[i+11],B[i+12],B[i+13],B[i+14],B[i+15],B[i+16]), multiplier);
#elif INTENSITY5
BBfunction5 (temp, (VTYPE)(B[i+9],B[i+10],B[i+11],B[i+12],B[i+13],B[i+14],B[i+15],B[i+16]), multiplier);
#endif

	write_channel_altera (c0, temp);

}

#elif UNROLL_FACTOR == 16
#pragma ivdep
for (int i = 1; i < lll; i+=16) {
	VTYPE temp;

#if INTENSITY1
	BBfunction (temp, (VTYPE)(B[i+9],B[i+10],B[i+11],B[i+12],B[i+13],B[i+14],B[i+15],B[i+16],B[i+17],B[i+18],B[i+19],B[i+20],B[i+21],B[i+22],B[i+23],B[i+24]), multiplier);
#elif INTENSITY2
	BBfunction2 (temp, (VTYPE)(B[i+9],B[i+10],B[i+11],B[i+12],B[i+13],B[i+14],B[i+15],B[i+16],B[i+17],B[i+18],B[i+19],B[i+20],B[i+21],B[i+22],B[i+23],B[i+24]), multiplier);
#elif INTENSITY3
	BBfunction3 (temp, (VTYPE)(B[i+9],B[i+10],B[i+11],B[i+12],B[i+13],B[i+14],B[i+15],B[i+16],B[i+17],B[i+18],B[i+19],B[i+20],B[i+21],B[i+22],B[i+23],B[i+24]), multiplier);
#elif INTENSITY4
	BBfunction4 (temp, (VTYPE)(B[i+9],B[i+10],B[i+11],B[i+12],B[i+13],B[i+14],B[i+15],B[i+16],B[i+17],B[i+18],B[i+19],B[i+20],B[i+21],B[i+22],B[i+23],B[i+24]), multiplier);
#elif INTENSITY5
	BBfunction5 (temp, (VTYPE)(B[i+9],B[i+10],B[i+11],B[i+12],B[i+13],B[i+14],B[i+15],B[i+16],B[i+17],B[i+18],B[i+19],B[i+20],B[i+21],B[i+22],B[i+23],B[i+24]), multiplier);
#endif

	write_channel_altera (c0, temp);

}

#endif

#endif

}


__kernel void S211K2 (__global DTYPE* restrict A,
					__global DTYPE* restrict B,
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

	DTYPE multiplier = 2.5;
	const int gid = get_global_id(0);
  const int index = gid+1;
#if INTENSITY1
	Bfunction (B[index+6], BPrime[index+7], multiplier);	
#elif INTENSITY2
	Bfunction2 (B[index+6], BPrime[index+7], multiplier);	
#elif INTENSITY3
	Bfunction3 (B[index+6], BPrime[index+7], multiplier);	
#elif INTENSITY4
	Bfunction4 (B[index+6], BPrime[index+7], multiplier);	
#elif INTENSITY5
	Bfunction5 (B[index+6], BPrime[index+7], multiplier);	
#endif

#endif

#ifdef FPGA_NDRANGE

#endif

#ifdef FPGA_SINGLE

  DTYPE multiplier = 2.5;		
	VTYPE output;


#if UNROLL_FACTOR == 1
	#pragma ivdep
	for (int i = 1; i < lll; i++) {
    	VTYPE msg = read_channel_altera(c0);

#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif


    write_channel_altera(c1, output);
	}
#elif UNROLL_FACTOR == 2
	#pragma ivdep
	for (int i = 1; i < lll; i+=2) {
   		VTYPE msg = read_channel_altera(c0);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

    write_channel_altera(c1, output);
	}
#elif UNROLL_FACTOR == 4
	#pragma ivdep
	for (int i = 1; i < lll; i+=4) {
   		VTYPE msg = read_channel_altera(c0);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif


		write_channel_altera(c1, output);
	}
#elif UNROLL_FACTOR == 8
	#pragma ivdep
	for (int i = 1; i < lll; i+=8) {
   		 VTYPE msg = read_channel_altera(c0);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

		write_channel_altera(c1, output);

	}
#elif UNROLL_FACTOR == 16
	#pragma ivdep
	for (int i = 1; i < lll; i+=16) {
   		VTYPE msg = read_channel_altera(c0);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

		write_channel_altera(c1, output);

	}


#endif

#endif

}

__kernel void S211K3 (__global DTYPE* restrict A,
					__global DTYPE* restrict B,
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
	DTYPE multiplier = 3.5;
  const int gid = get_global_id(0);
  const int index = gid+1;


#if INTENSITY1
	Bfunction (BPrime[index+4], B[index+5], multiplier);	
#elif INTENSITY2
	Bfunction2 (BPrime[index+4], B[index+5], multiplier);	
#elif INTENSITY3
	Bfunction3 (BPrime[index+4], B[index+5], multiplier);	
#elif INTENSITY4
	Bfunction4 (BPrime[index+4], B[index+5], multiplier);	
#elif INTENSITY5
	Bfunction5 (BPrime[index+4], B[index+5], multiplier);	
#endif

#endif

#ifdef FPGA_SINGLE

  DTYPE multiplier = 2.5;		
	VTYPE output;


#if UNROLL_FACTOR == 1
	#pragma ivdep
	for (int i = 1; i < lll; i++) {
    	VTYPE msg = read_channel_altera(c1);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

    write_channel_altera(c2, output);
	}
#elif UNROLL_FACTOR == 2
	#pragma ivdep
	for (int i = 1; i < lll; i+=2) {
   		VTYPE msg = read_channel_altera(c1);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

    write_channel_altera(c2, output);
	}
#elif UNROLL_FACTOR == 4
	#pragma ivdep
	for (int i = 1; i < lll; i+=4) {
   		VTYPE msg = read_channel_altera(c1);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif



		write_channel_altera(c2, output);
	}
#elif UNROLL_FACTOR == 8
	#pragma ivdep
	for (int i = 1; i < lll; i+=8) {
   		VTYPE msg = read_channel_altera(c1);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

		write_channel_altera(c2, output);

	}
#elif UNROLL_FACTOR == 16
	#pragma ivdep
	for (int i = 1; i < lll; i+=16) {
   		VTYPE msg = read_channel_altera(c1);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

		write_channel_altera(c2, output);
	
	}


#endif

#endif

}

__kernel void S211K4 (__global DTYPE* restrict A,
					__global DTYPE* restrict B,
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
	DTYPE multiplier = 3.5;
  const int gid = get_global_id(0);
  const int index = gid+1;


#if INTENSITY1
	Bfunction (B[index+2], BPrime[index+3], multiplier);	
#elif INTENSITY2
	Bfunction2 (B[index+2], BPrime[index+3], multiplier);	
#elif INTENSITY3
	Bfunction3 (B[index+2], BPrime[index+3], multiplier);	
#elif INTENSITY4
	Bfunction4 (B[index+2], BPrime[index+3], multiplier);	
#elif INTENSITY5
	Bfunction5 (B[index+2], BPrime[index+3], multiplier);	
#endif

#endif

#ifdef FPGA_SINGLE

  DTYPE multiplier = 2.5;		
	VTYPE output;


#if UNROLL_FACTOR == 1
	#pragma ivdep
	for (int i = 1; i < lll; i++) {
    	VTYPE msg = read_channel_altera(c2);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

    write_channel_altera(c3, output);
	}
#elif UNROLL_FACTOR == 2
	#pragma ivdep
	for (int i = 1; i < lll; i+=2) {
   		VTYPE msg = read_channel_altera(c2);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

    write_channel_altera(c3, output);
	}
#elif UNROLL_FACTOR == 4
	#pragma ivdep
	for (int i = 1; i < lll; i+=4) {
   		VTYPE msg = read_channel_altera(c2);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif



		write_channel_altera(c3, output);
	}
#elif UNROLL_FACTOR == 8
	#pragma ivdep
	for (int i = 1; i < lll; i+=8) {
   		VTYPE msg = read_channel_altera(c2);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

		write_channel_altera(c3, output);

	}
#elif UNROLL_FACTOR == 16
	#pragma ivdep
	for (int i = 1; i < lll; i+=16) {
   		VTYPE msg = read_channel_altera(c2);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

		write_channel_altera(c3, output);
	
	}


#endif

#endif

}

__kernel void S211K5 (__global DTYPE* restrict A,
					__global DTYPE* restrict B,
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
	DTYPE multiplier = 3.5;
  const int gid = get_global_id(0);
  const int index = gid+1;


#if INTENSITY1
	Bfunction (BPrime[index], B[index+1], multiplier);	
#elif INTENSITY2
	Bfunction2 (BPrime[index], B[index+1], multiplier);	
#elif INTENSITY3
	Bfunction3 (BPrime[index], B[index+1], multiplier);	
#elif INTENSITY4
	Bfunction4 (BPrime[index], B[index+1], multiplier);	
#elif INTENSITY5
	Bfunction5 (BPrime[index], B[index+1], multiplier);	
#endif

#endif

#ifdef FPGA_SINGLE

  DTYPE multiplier = 2.5;		
	VTYPE output;


#if UNROLL_FACTOR == 1
	#pragma ivdep
	for (int i = 1; i < lll; i++) {
    	VTYPE msg = read_channel_altera(c3);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

    write_channel_altera(c4, output);
	}
#elif UNROLL_FACTOR == 2
	#pragma ivdep
	for (int i = 1; i < lll; i+=2) {
   		VTYPE msg = read_channel_altera(c3);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

    write_channel_altera(c4, output);
	}
#elif UNROLL_FACTOR == 4
	#pragma ivdep
	for (int i = 1; i < lll; i+=4) {
   		VTYPE msg = read_channel_altera(c3);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif



		write_channel_altera(c4, output);
	}
#elif UNROLL_FACTOR == 8
	#pragma ivdep
	for (int i = 1; i < lll; i+=8) {
   		VTYPE msg = read_channel_altera(c3);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

		write_channel_altera(c4, output);

	}
#elif UNROLL_FACTOR == 16
	#pragma ivdep
	for (int i = 1; i < lll; i+=16) {
   		VTYPE msg = read_channel_altera(c3);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

		write_channel_altera(c4, output);
	
	}


#endif

#endif

}

__kernel void S211K6 (__global DTYPE* restrict A,
					__global DTYPE* restrict B,
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

	DTYPE multiplier = 4.5;
  const int gid = get_global_id(0);
  const int index = gid+1;
#if INTENSITY1
	Bfunction (A[index], BPrime[index-1], multiplier);	
#elif INTENSITY2
	Bfunction2 (A[index], BPrime[index-1], multiplier);	
#elif INTENSITY3
	Bfunction3 (A[index], BPrime[index-1], multiplier);	
#elif INTENSITY4
	Bfunction4 (A[index], BPrime[index-1], multiplier);	
#elif INTENSITY5
	Bfunction5 (A[index], BPrime[index-1], multiplier);	
#endif

#endif

#ifdef FPGA_NDRANGE

#endif

#ifdef FPGA_SINGLE

  DTYPE multiplier = 2.5;		
	VTYPE output;
#if UNROLL_FACTOR == 1
	#pragma ivdep
	for (int i = 1; i < lll; i++) {
    	VTYPE msg = read_channel_altera(c4);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

		A[i] = output.s0;
	}
#elif UNROLL_FACTOR == 2
	#pragma ivdep
	for (int i = 1; i < lll; i+=2) {
   		VTYPE msg = read_channel_altera(c4);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

		A[i] = output.s0;
    	A[i+1] = output.s1;
	}
#elif UNROLL_FACTOR == 4
	#pragma ivdep
	for (int i = 1; i < lll; i+=4) {
   		VTYPE msg = read_channel_altera(c4);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

		A[i] = output.s0;
    	A[i+1] = output.s1;
    	A[i+2] = output.s2;
    	A[i+3] = output.s3;
	}
#elif UNROLL_FACTOR == 8
	#pragma ivdep
	for (int i = 1; i < lll; i+=8) {
   		VTYPE msg = read_channel_altera(c4);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif


		A[i] = output.s0;
    	A[i+1] = output.s1;
    	A[i+2] = output.s2;
    	A[i+3] = output.s3;
    	A[i+4] = output.s4;
    	A[i+5] = output.s5;
    	A[i+6] = output.s6;
      A[i+7] = output.s7;
	}
#elif UNROLL_FACTOR == 16
	#pragma ivdep
	for (int i = 1; i < lll; i+=16) {
   		VTYPE msg = read_channel_altera(c4);
#if INTENSITY1
	BBfunction (output, msg, multiplier);
#elif INTENSITY2
	BBfunction2 (output, msg, multiplier);
#elif INTENSITY3
	BBfunction3 (output, msg, multiplier);
#elif INTENSITY4
	BBfunction4 (output, msg, multiplier);
#elif INTENSITY5
	BBfunction5 (output, msg, multiplier);
#endif

		A[i] = output.s0;
    	A[i+1] = output.s1;
    	A[i+2] = output.s2;
    	A[i+3] = output.s3;
    	A[i+4] = output.s4;
    	A[i+5] = output.s5;
    	A[i+6] = output.s6;
		A[i+7] = output.s7;
    	A[i+8] = output.s8;
    	A[i+9] = output.s9;
    	A[i+10] = output.sa;
    	A[i+11] = output.sb;
    	A[i+12] = output.sc;
    	A[i+13] = output.sd;
      A[i+14] = output.se;
			A[i+15] = output.sf;
	}


#endif

#endif

}
