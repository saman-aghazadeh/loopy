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
channel struct Msg c0 __attribute__((depth(4)));
channel struct Msg c1 __attribute__((depth(4)));
channel struct Msg c2 __attribute__((depth(4)));
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

DTYPE multiplier = 1.5;

#if UNROLL_FACTOR == 1
#pragma ivdep
for (int i = 1; i < lll; i++) {

	DTYPE temp;

#if INTENSITY1
	Bfunction (temp, B[i+1], multiplier);
#elif INTENSITY2
	Bfunction2 (temp, B[i+1], multiplier);
#elif INTENSITY3
	Bfunction3 (temp, B[i+1], multiplier);
#elif INTENSITY4
	Bfunction4 (temp, B[i+1], multiplier);
#elif INTENSITY5
	Bfunction5 (temp, B[i+1], multiplier);
#endif

	Msg msg;
  msg.temp0 = temp;
	write_channel_altera (c0, msg);
}
#elif UNROLL_FACTOR == 2
#pragma ivdep
for (int i = 1; i < lll; i+=2) {
	DTYPE temp0;
  DTYPE temp1;

#if INTENSITY1
	Bfunction (temp0, B[i+1], multiplier);
	Bfunction (temp1, B[i+2], multiplier);
#elif INTENSITY2
	Bfunction2 (temp0, B[i+1], multiplier);
	Bfunction2 (temp1, B[i+2], multiplier);
#elif INTENSITY3
	Bfunction3 (temp0, B[i+1], multiplier);
	Bfunction3 (temp1, B[i+2], multiplier);
#elif INTENSITY4
	Bfunction4 (temp0, B[i+1], multiplier);
	Bfunction4 (temp1, B[i+2], multiplier);
#elif INTENSITY5
	Bfunction5 (temp0, B[i+1], multiplier);
	Bfunction5 (temp1, B[i+2], multiplier);
#endif

	Msg msg;
  msg.temp0 = temp0;
  msg.temp1 = temp1;

	write_channel_altera (c0, msg);
}
#elif UNROLL_FACTOR == 4
#pragma ivdep
for (int i = 1; i < lll; i+=4) {
	DTYPE temp0;
  	DTYPE temp1;
	DTYPE temp2;
 	DTYPE temp3;

#if INTENSITY1
	Bfunction (temp0, B[i+1], multiplier);
	Bfunction (temp1, B[i+2], multiplier);
	Bfunction (temp2, B[i+3], multiplier);
	Bfunction (temp3, B[i+4], multiplier);
#elif INTENSITY2
	Bfunction2 (temp0, B[i+1], multiplier);
	Bfunction2 (temp1, B[i+2], multiplier);
	Bfunction2 (temp2, B[i+3], multiplier);
	Bfunction2 (temp3, B[i+4], multiplier);
#elif INTENSITY3
	Bfunction3 (temp0, B[i+1], multiplier);
	Bfunction3 (temp1, B[i+2], multiplier);
	Bfunction3 (temp2, B[i+3], multiplier);
	Bfunction3 (temp3, B[i+4], multiplier);
#elif INTENSITY4
	Bfunction4 (temp0, B[i+1], multiplier);
	Bfunction4 (temp1, B[i+2], multiplier);
	Bfunction4 (temp2, B[i+3], multiplier);
	Bfunction4 (temp3, B[i+4], multiplier);
#elif INTENSITY5
	Bfunction5 (temp0, B[i+1], multiplier);
	Bfunction5 (temp1, B[i+2], multiplier);
	Bfunction5 (temp2, B[i+3], multiplier);
	Bfunction5 (temp3, B[i+4], multiplier);
#endif

	Msg msg;
  msg.temp0 = temp0;
  msg.temp1 = temp1;
  msg.temp2 = temp2;
  msg.temp3 = temp3;

	write_channel_altera (c0, msg);
}

#elif UNROLL_FACTOR == 8
#pragma ivdep
for (int i = 1; i < lll; i+=8) {
	DTYPE temp0;
  DTYPE temp1;
	DTYPE temp2;
 	DTYPE temp3;
	DTYPE temp4;
	DTYPE temp5;
  DTYPE temp6;
	DTYPE temp7;

#if INTENSITY1
	Bfunction (temp0, B[i+1], multiplier);
	Bfunction (temp1, B[i+2], multiplier);
	Bfunction (temp2, B[i+3], multiplier);
	Bfunction (temp3, B[i+4], multiplier);
	Bfunction (temp4, B[i+5], multiplier);
	Bfunction (temp5, B[i+6], multiplier);
	Bfunction (temp6, B[i+7], multiplier);
	Bfunction (temp7, B[i+8], multiplier);
#elif INTENSITY2
	Bfunction2 (temp0, B[i+1], multiplier);
	Bfunction2 (temp1, B[i+2], multiplier);
	Bfunction2 (temp2, B[i+3], multiplier);
	Bfunction2 (temp3, B[i+4], multiplier);
	Bfunction2 (temp4, B[i+5], multiplier);
	Bfunction2 (temp5, B[i+6], multiplier);
	Bfunction2 (temp6, B[i+7], multiplier);
	Bfunction2 (temp7, B[i+8], multiplier);
#elif INTENSITY3
	Bfunction3 (temp0, B[i+1], multiplier);
	Bfunction3 (temp1, B[i+2], multiplier);
	Bfunction3 (temp2, B[i+3], multiplier);
	Bfunction3 (temp3, B[i+4], multiplier);
	Bfunction3 (temp4, B[i+5], multiplier);
	Bfunction3 (temp5, B[i+6], multiplier);
	Bfunction3 (temp6, B[i+7], multiplier);
	Bfunction3 (temp7, B[i+8], multiplier);
#elif INTENSITY4
	Bfunction4 (temp0, B[i+1], multiplier);
	Bfunction4 (temp1, B[i+2], multiplier);
	Bfunction4 (temp2, B[i+3], multiplier);
	Bfunction4 (temp3, B[i+4], multiplier);
	Bfunction4 (temp4, B[i+5], multiplier);
	Bfunction4 (temp5, B[i+6], multiplier);
	Bfunction4 (temp6, B[i+7], multiplier);
	Bfunction4 (temp7, B[i+8], multiplier);
#elif INTENSITY5
	Bfunction5 (temp0, B[i+1], multiplier);
	Bfunction5 (temp1, B[i+2], multiplier);
	Bfunction5 (temp2, B[i+3], multiplier);
	Bfunction5 (temp3, B[i+4], multiplier);
	Bfunction5 (temp4, B[i+5], multiplier);
	Bfunction5 (temp5, B[i+6], multiplier);
	Bfunction5 (temp6, B[i+7], multiplier);
	Bfunction5 (temp7, B[i+8], multiplier);
#endif

	struct Msg msg;
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

#elif UNROLL_FACTOR == 16
#pragma ivdep
for (int i = 1; i < lll; i+=16) {
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

#if INTENSITY1
	Bfunction (temp0, B[i+1], multiplier);
	Bfunction (temp1, B[i+2], multiplier);
	Bfunction (temp2, B[i+3], multiplier);
	Bfunction (temp3, B[i+4], multiplier);
	Bfunction (temp4, B[i+5], multiplier);
	Bfunction (temp5, B[i+6], multiplier);
	Bfunction (temp6, B[i+7], multiplier);
	Bfunction (temp7, B[i+8], multiplier);
	Bfunction (temp8, B[i+9], multiplier);
	Bfunction (temp9, B[i+10], multiplier);
	Bfunction (temp10, B[i+11], multiplier);
	Bfunction (temp11, B[i+12], multiplier);
	Bfunction (temp12, B[i+13], multiplier);
	Bfunction (temp13, B[i+14], multiplier);
	Bfunction (temp14, B[i+15], multiplier);
	Bfunction (temp15, B[i+16], multiplier);
#elif INTENSITY2
	Bfunction2 (temp0, B[i+1], multiplier);
	Bfunction2 (temp1, B[i+2], multiplier);
	Bfunction2 (temp2, B[i+3], multiplier);
	Bfunction2 (temp3, B[i+4], multiplier);
	Bfunction2 (temp4, B[i+5], multiplier);
	Bfunction2 (temp5, B[i+6], multiplier);
	Bfunction2 (temp6, B[i+7], multiplier);
	Bfunction2 (temp7, B[i+8], multiplier);
	Bfunction2 (temp8, B[i+9], multiplier);
	Bfunction2 (temp9, B[i+10], multiplier);
	Bfunction2 (temp10, B[i+11], multiplier);
	Bfunction2 (temp11, B[i+12], multiplier);
	Bfunction2 (temp12, B[i+13], multiplier);
	Bfunction2 (temp13, B[i+14], multiplier);
	Bfunction2 (temp14, B[i+15], multiplier);
	Bfunction2 (temp15, B[i+16], multiplier);
#elif INTENSITY3
	Bfunction3 (temp0, B[i+1], multiplier);
	Bfunction3 (temp1, B[i+2], multiplier);
	Bfunction3 (temp2, B[i+3], multiplier);
	Bfunction3 (temp3, B[i+4], multiplier);
	Bfunction3 (temp4, B[i+5], multiplier);
	Bfunction3 (temp5, B[i+6], multiplier);
	Bfunction3 (temp6, B[i+7], multiplier);
	Bfunction3 (temp7, B[i+8], multiplier);
	Bfunction3 (temp8, B[i+9], multiplier);
	Bfunction3 (temp9, B[i+10], multiplier);
	Bfunction3 (temp10, B[i+11], multiplier);
	Bfunction3 (temp11, B[i+12], multiplier);
	Bfunction3 (temp12, B[i+13], multiplier);
	Bfunction3 (temp13, B[i+14], multiplier);
	Bfunction3 (temp14, B[i+15], multiplier);
	Bfunction3 (temp15, B[i+16], multiplier);
#elif INTENSITY4
	Bfunction4 (temp0, B[i+1], multiplier);
	Bfunction4 (temp1, B[i+2], multiplier);
	Bfunction4 (temp2, B[i+3], multiplier);
	Bfunction4 (temp3, B[i+4], multiplier);
	Bfunction4 (temp4, B[i+5], multiplier);
	Bfunction4 (temp5, B[i+6], multiplier);
	Bfunction4 (temp6, B[i+7], multiplier);
	Bfunction4 (temp7, B[i+8], multiplier);
	Bfunction4 (temp8, B[i+9], multiplier);
	Bfunction4 (temp9, B[i+10], multiplier);
	Bfunction4 (temp10, B[i+11], multiplier);
	Bfunction4 (temp11, B[i+12], multiplier);
	Bfunction4 (temp12, B[i+13], multiplier);
	Bfunction4 (temp13, B[i+14], multiplier);
	Bfunction4 (temp14, B[i+15], multiplier);
	Bfunction4 (temp15, B[i+16], multiplier);
#elif INTENSITY5
	Bfunction5 (temp0, B[i+1], multiplier);
	Bfunction5 (temp1, B[i+2], multiplier);
	Bfunction5 (temp2, B[i+3], multiplier);
	Bfunction5 (temp3, B[i+4], multiplier);
	Bfunction5 (temp4, B[i+5], multiplier);
	Bfunction5 (temp5, B[i+6], multiplier);
	Bfunction5 (temp6, B[i+7], multiplier);
	Bfunction5 (temp7, B[i+8], multiplier);
	Bfunction5 (temp8, B[i+9], multiplier);
	Bfunction5 (temp9, B[i+10], multiplier);
	Bfunction5 (temp10, B[i+11], multiplier);
	Bfunction5 (temp11, B[i+12], multiplier);
	Bfunction5 (temp12, B[i+13], multiplier);
	Bfunction5 (temp13, B[i+14], multiplier);
	Bfunction5 (temp14, B[i+15], multiplier);
	Bfunction5 (temp15, B[i+16], multiplier);
#endif

	struct Msg msg;
  msg.temp0 = temp0;
  msg.temp1 = temp1;
  msg.temp2 = temp2;
  msg.temp3 = temp3;
  msg.temp4 = temp4;
  msg.temp5 = temp5;
  msg.temp6 = temp6;
  msg.temp7 = temp7;
  msg.temp8 = temp8;
  msg.temp9 = temp9;
  msg.temp10 = temp10;
  msg.temp11 = temp11;
  msg.temp12 = temp12;
  msg.temp13 = temp13;
  msg.temp14 = temp14;
  msg.temp15 = temp15;

	write_channel_altera (c0, msg);

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

#ifdef FPGA_NDRANGE

#endif

#ifdef FPGA_SINGLE

  DTYPE multiplier = 2.5;		
	struct Msg output;


#if UNROLL_FACTOR == 1
	#pragma ivdep
	for (int i = 1; i < lll; i++) {
    	MSG msg = read_channel_altera(c0);

#if INTENSITY1
	Bfunction (output.temp0, msg.temp0, multiplier);
#elif INTENSITY2
	Bfunction2 (output.temp0, msg.temp0, multiplier);
#elif INTENSITY3
	Bfunction3 (output.temp0, msg.temp0, multiplier);
#elif INTENSITY4
	Bfunction4 (output.temp0, msg.temp0, multiplier);
#elif INTENSITY5
	Bfunction5 (output.temp0, msg.temp0, multiplier);
#endif


    write_channel_altera(c1);
	}
#elif UNROLL_FACTOR == 2
	#pragma ivdep
	for (int i = 1; i < lll; i+=2) {
   		Msg msg = read_channel_altera(c0);
#if INTENSITY1
	Bfunction (output.temp0, msg.temp0, multiplier);
	Bfunction (output.temp1, msg.temp1, multiplier);
#elif INTENSITY2
	Bfunction2 (output.temp0, msg.temp0, multiplier);
	Bfunction2 (output.temp1, msg.temp1, multiplier);
#elif INTENSITY3
	Bfunction3 (output.temp0, msg.temp0, multiplier);
	Bfunction3 (output.temp1, msg.temp1, multiplier);
#elif INTENSITY4
	Bfunction4 (output.temp0, msg.temp0, multiplier);
	Bfunction4 (output.temp1, msg.temp1, multiplier);
#elif INTENSITY5
	Bfunction5 (output.temp0, msg.temp0, multiplier);
	Bfunction5 (output.temp1, msg.temp1, multiplier);
#endif

    write_channel_altera(c1, output);
	}
#elif UNROLL_FACTOR == 4
	#pragma ivdep
	for (int i = 1; i < lll; i+=4) {
   		Msg msg = read_channel_altera(c0);
#if INTENSITY1
	Bfunction (output.temp0, msg.temp0, multiplier);
	Bfunction (output.temp1, msg.temp1, multiplier);
	Bfunction (output.temp2, msg.temp2, multiplier);
	Bfunction (output.temp3, msg.temp3, multiplier);
#elif INTENSITY2
	Bfunction2 (output.temp0, msg.temp0, multiplier);
	Bfunction2 (output.temp1, msg.temp1, multiplier);
	Bfunction2 (output.temp2, msg.temp2, multiplier);
	Bfunction2 (output.temp3, msg.temp3, multiplier);
#elif INTENSITY3
	Bfunction3 (output.temp0, msg.temp0, multiplier);
	Bfunction3 (output.temp1, msg.temp1, multiplier);
	Bfunction3 (output.temp2, msg.temp2, multiplier);
	Bfunction3 (output.temp3, msg.temp3, multiplier);
#elif INTENSITY4
	Bfunction4 (output.temp0, msg.temp0, multiplier);
	Bfunction4 (output.temp1, msg.temp1, multiplier);
	Bfunction4 (output.temp2, msg.temp2, multiplier);
	Bfunction4 (output.temp3, msg.temp3, multiplier);
#elif INTENSITY5
	Bfunction5 (output.temp0, msg.temp0, multiplier);
	Bfunction5 (output.temp1, msg.temp1, multiplier);
	Bfunction5 (output.temp2, msg.temp2, multiplier);
	Bfunction5 (output.temp3, msg.temp3, multiplier);
#endif


		write_channel_altera(c1, output);
	}
#elif UNROLL_FACTOR == 8
	#pragma ivdep
	for (int i = 1; i < lll; i+=8) {
   		struct Msg msg = read_channel_altera(c0);
#if INTENSITY1
	Bfunction (output.temp0, msg.temp0, multiplier);
	Bfunction (output.temp1, msg.temp1, multiplier);
	Bfunction (output.temp2, msg.temp2, multiplier);
	Bfunction (output.temp3, msg.temp3, multiplier);
	Bfunction (output.temp4, msg.temp4, multiplier);
	Bfunction (output.temp5, msg.temp5, multiplier);
	Bfunction (output.temp6, msg.temp6, multiplier);
	Bfunction (output.temp7, msg.temp7, multiplier);
#elif INTENSITY2
	Bfunction2 (output.temp0, msg.temp0, multiplier);
	Bfunction2 (output.temp1, msg.temp1, multiplier);
	Bfunction2 (output.temp2, msg.temp2, multiplier);
	Bfunction2 (output.temp3, msg.temp3, multiplier);
	Bfunction2 (output.temp4, msg.temp4, multiplier);
	Bfunction2 (output.temp5, msg.temp5, multiplier);
	Bfunction2 (output.temp6, msg.temp6, multiplier);
	Bfunction2 (output.temp7, msg.temp7, multiplier);
#elif INTENSITY3
	Bfunction3 (output.temp0, msg.temp0, multiplier);
	Bfunction3 (output.temp1, msg.temp1, multiplier);
	Bfunction3 (output.temp2, msg.temp2, multiplier);
	Bfunction3 (output.temp3, msg.temp3, multiplier);
	Bfunction3 (output.temp4, msg.temp5, multiplier);
	Bfunction3 (output.temp5, msg.temp6, multiplier);
	Bfunction3 (output.temp6, msg.temp7, multiplier);
	Bfunction3 (output.temp7, msg.temp8, multiplier);
#elif INTENSITY4
	Bfunction4 (output.temp0, msg.temp0, multiplier);
	Bfunction4 (output.temp1, msg.temp1, multiplier);
	Bfunction4 (output.temp2, msg.temp2, multiplier);
	Bfunction4 (output.temp3, msg.temp3, multiplier);
	Bfunction4 (output.temp4, msg.temp4, multiplier);
	Bfunction4 (output.temp5, msg.temp5, multiplier);
	Bfunction4 (output.temp6, msg.temp6, multiplier);
	Bfunction4 (output.temp7, msg.temp7, multiplier);
#elif INTENSITY5
	Bfunction5 (output.temp0, msg.temp0, multiplier);
	Bfunction5 (output.temp1, msg.temp1, multiplier);
	Bfunction5 (output.temp2, msg.temp2, multiplier);
	Bfunction5 (output.temp3, msg.temp3, multiplier);
	Bfunction5 (output.temp4, msg.temp4, multiplier);
	Bfunction5 (output.temp5, msg.temp5, multiplier);
	Bfunction5 (output.temp6, msg.temp6, multiplier);
	Bfunction5 (output.temp7, msg.temp7, multiplier);
#endif

		write_channel_altera(c1, output);

	}
#elif UNROLL_FACTOR == 16
	#pragma ivdep
	for (int i = 1; i < lll; i+=16) {
   		struct Msg msg = read_channel_altera(c0);
#if INTENSITY1
	Bfunction (output.temp0, msg.temp0, multiplier);
	Bfunction (output.temp1, msg.temp1, multiplier);
	Bfunction (output.temp2, msg.temp2, multiplier);
	Bfunction (output.temp3, msg.temp3, multiplier);
	Bfunction (output.temp4, msg.temp4, multiplier);
	Bfunction (output.temp5, msg.temp5, multiplier);
	Bfunction (output.temp6, msg.temp6, multiplier);
	Bfunction (output.temp7, msg.temp7, multiplier);
	Bfunction (output.temp8, msg.temp8, multiplier);
	Bfunction (output.temp9, msg.temp9, multiplier);
	Bfunction (output.temp10, msg.temp10, multiplier);
	Bfunction (output.temp11, msg.temp11, multiplier);
	Bfunction (output.temp12, msg.temp12, multiplier);
	Bfunction (output.temp13, msg.temp13, multiplier);
	Bfunction (output.temp14, msg.temp14, multiplier);
	Bfunction (output.temp15, msg.temp15, multiplier);
#elif INTENSITY2
	Bfunction2 (output.temp0, msg.temp0, multiplier);
	Bfunction2 (output.temp1, msg.temp1, multiplier);
	Bfunction2 (output.temp2, msg.temp2, multiplier);
	Bfunction2 (output.temp3, msg.temp3, multiplier);
	Bfunction2 (output.temp4, msg.temp4, multiplier);
	Bfunction2 (output.temp5, msg.temp5, multiplier);
	Bfunction2 (output.temp6, msg.temp6, multiplier);
	Bfunction2 (output.temp7, msg.temp7, multiplier);
	Bfunction2 (output.temp8, msg.temp8, multiplier);
	Bfunction2 (output.temp9, msg.temp9, multiplier);
	Bfunction2 (output.temp10, msg.temp10, multiplier);
	Bfunction2 (output.temp11, msg.temp11, multiplier);
	Bfunction2 (output.temp12, msg.temp12, multiplier);
	Bfunction2 (output.temp13, msg.temp13, multiplier);
	Bfunction2 (output.temp14, msg.temp14, multiplier);
	Bfunction2 (output.temp15, msg.temp15, multiplier);
#elif INTENSITY3
	Bfunction3 (output.temp0, msg.temp0, multiplier);
	Bfunction3 (output.temp1, msg.temp1, multiplier);
	Bfunction3 (output.temp2, msg.temp2, multiplier);
	Bfunction3 (output.temp3, msg.temp3, multiplier);
	Bfunction3 (output.temp4, msg.temp4, multiplier);
	Bfunction3 (output.temp5, msg.temp5, multiplier);
	Bfunction3 (output.temp6, msg.temp6, multiplier);
	Bfunction3 (output.temp7, msg.temp7, multiplier);
	Bfunction3 (output.temp8, msg.temp8, multiplier);
	Bfunction3 (output.temp9, msg.temp9, multiplier);
	Bfunction3 (output.temp10, msg.temp10, multiplier);
	Bfunction3 (output.temp11, msg.temp11, multiplier);
	Bfunction3 (output.temp12, msg.temp12, multiplier);
	Bfunction3 (output.temp13, msg.temp13, multiplier);
	Bfunction3 (output.temp14, msg.temp14, multiplier);
	Bfunction3 (output.temp15, msg.temp15, multiplier);
#elif INTENSITY4
	Bfunction4 (output.temp0, msg.temp0, multiplier);
	Bfunction4 (output.temp1, msg.temp1, multiplier);
	Bfunction4 (output.temp2, msg.temp2, multiplier);
	Bfunction4 (output.temp3, msg.temp3, multiplier);
	Bfunction4 (output.temp4, msg.temp4, multiplier);
	Bfunction4 (output.temp5, msg.temp5, multiplier);
	Bfunction4 (output.temp6, msg.temp6, multiplier);
	Bfunction4 (output.temp7, msg.temp7, multiplier);
	Bfunction4 (output.temp8, msg.temp8, multiplier);
	Bfunction4 (output.temp9, msg.temp9, multiplier);
	Bfunction4 (output.temp10, msg.temp10, multiplier);
	Bfunction4 (output.temp11, msg.temp11, multiplier);
	Bfunction4 (output.temp12, msg.temp12, multiplier);
	Bfunction4 (output.temp13, msg.temp13, multiplier);
	Bfunction4 (output.temp14, msg.temp14, multiplier);
	Bfunction4 (output.temp15, msg.temp15, multiplier);
#elif INTENSITY5
	Bfunction5 (output.temp0, msg.temp0, multiplier);
	Bfunction5 (output.temp1, msg.temp1, multiplier);
	Bfunction5 (output.temp2, msg.temp2, multiplier);
	Bfunction5 (output.temp3, msg.temp3, multiplier);
	Bfunction5 (output.temp4, msg.temp4, multiplier);
	Bfunction5 (output.temp5, msg.temp5, multiplier);
	Bfunction5 (output.temp6, msg.temp6, multiplier);
	Bfunction5 (output.temp7, msg.temp7, multiplier);
	Bfunction5 (output.temp8, msg.temp8, multiplier);
	Bfunction5 (output.temp9, msg.temp9, multiplier);
	Bfunction5 (output.temp10, msg.temp10, multiplier);
	Bfunction5 (output.temp11, msg.temp11, multiplier);
	Bfunction5 (output.temp12, msg.temp12, multiplier);
	Bfunction5 (output.temp13, msg.temp13, multiplier);
	Bfunction5 (output.temp14, msg.temp14, multiplier);
	Bfunction5 (output.temp15, msg.temp15, multiplier);
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
	struct Msg output;


#if UNROLL_FACTOR == 1
	#pragma ivdep
	for (int i = 1; i < lll; i++) {
    	MSG msg = read_channel_altera(c1);
#if INTENSITY1
	Bfunction (output.temp0, msg.temp0, multiplier);
#elif INTENSITY2
	Bfunction2 (output.temp0, msg.temp0, multiplier);
#elif INTENSITY3
	Bfunction3 (output.temp0, msg.temp0, multiplier);
#elif INTENSITY4
	Bfunction4 (output.temp0, msg.temp0, multiplier);
#elif INTENSITY5
	Bfunction5 (output.temp0, msg.temp0, multiplier);
#endif

    write_channel_altera(c2, output);
	}
#elif UNROLL_FACTOR == 2
	#pragma ivdep
	for (int i = 1; i < lll; i+=2) {
   		Msg msg = read_channel_altera(c1);
#if INTENSITY1
	Bfunction (output.temp0, msg.temp0, multiplier);
	Bfunction (output.temp1, msg.temp1, multiplier);
#elif INTENSITY2
	Bfunction2 (output.temp0, msg.temp0, multiplier);
	Bfunction2 (output.temp1, msg.temp1, multiplier);
#elif INTENSITY3
	Bfunction3 (output.temp0, msg.temp0, multiplier);
	Bfunction3 (output.temp1, msg.temp1, multiplier);
#elif INTENSITY4
	Bfunction4 (output.temp0, msg.temp0, multiplier);
	Bfunction4 (output.temp1, msg.temp1, multiplier);
#elif INTENSITY5
	Bfunction5 (output.temp0, msg.temp0, multiplier);
	Bfunction5 (output.temp1, msg.temp1, multiplier);
#endif

    write_channel_altera(c2, output);
	}
#elif UNROLL_FACTOR == 4
	#pragma ivdep
	for (int i = 1; i < lll; i+=4) {
   		Msg msg = read_channel_altera(c1);
#if INTENSITY1
	Bfunction (output.temp0, msg.temp0, multiplier);
	Bfunction (output.temp1, msg.temp1, multiplier);
	Bfunction (output.temp2, msg.temp2, multiplier);
	Bfunction (output.temp3, msg.temp3, multiplier);
#elif INTENSITY2
	Bfunction2 (output.temp0, msg.temp0, multiplier);
	Bfunction2 (output.temp1, msg.temp1, multiplier);
	Bfunction2 (output.temp2, msg.temp2, multiplier);
	Bfunction2 (output.temp3, msg.temp3, multiplier);
#elif INTENSITY3
	Bfunction3 (output.temp0, msg.temp0, multiplier);
	Bfunction3 (output.temp1, msg.temp1, multiplier);
	Bfunction3 (output.temp2, msg.temp2, multiplier);
	Bfunction3 (output.temp3, msg.temp3, multiplier);
#elif INTENSITY4
	Bfunction4 (output.temp0, msg.temp0, multiplier);
	Bfunction4 (output.temp1, msg.temp1, multiplier);
	Bfunction4 (output.temp2, msg.temp2, multiplier);
	Bfunction4 (output.temp3, msg.temp3, multiplier);
#elif INTENSITY5
	Bfunction5 (output.temp0, msg.temp0, multiplier);
	Bfunction5 (output.temp1, msg.temp1, multiplier);
	Bfunction5 (output.temp2, msg.temp2, multiplier);
	Bfunction5 (output.temp3, msg.temp3, multiplier);
#endif



		write_channel_altera(c2, output);
	}
#elif UNROLL_FACTOR == 8
	#pragma ivdep
	for (int i = 1; i < lll; i+=8) {
   		struct Msg msg = read_channel_altera(c1);
#if INTENSITY1
	Bfunction (output.temp0, msg.temp0, multiplier);
	Bfunction (output.temp1, msg.temp1, multiplier);
	Bfunction (output.temp2, msg.temp2, multiplier);
	Bfunction (output.temp3, msg.temp3, multiplier);
	Bfunction (output.temp4, msg.temp4, multiplier);
	Bfunction (output.temp5, msg.temp5, multiplier);
	Bfunction (output.temp6, msg.temp6, multiplier);
	Bfunction (output.temp7, msg.temp7, multiplier);
#elif INTENSITY2
	Bfunction2 (output.temp0, msg.temp0, multiplier);
	Bfunction2 (output.temp1, msg.temp1, multiplier);
	Bfunction2 (output.temp2, msg.temp2, multiplier);
	Bfunction2 (output.temp3, msg.temp3, multiplier);
	Bfunction2 (output.temp4, msg.temp4, multiplier);
	Bfunction2 (output.temp5, msg.temp5, multiplier);
	Bfunction2 (output.temp6, msg.temp6, multiplier);
	Bfunction2 (output.temp7, msg.temp7, multiplier);
#elif INTENSITY3
	Bfunction3 (output.temp0, msg.temp0, multiplier);
	Bfunction3 (output.temp1, msg.temp1, multiplier);
	Bfunction3 (output.temp2, msg.temp2, multiplier);
	Bfunction3 (output.temp3, msg.temp3, multiplier);
	Bfunction3 (output.temp4, msg.temp5, multiplier);
	Bfunction3 (output.temp5, msg.temp6, multiplier);
	Bfunction3 (output.temp6, msg.temp7, multiplier);
	Bfunction3 (output.temp7, msg.temp8, multiplier);
#elif INTENSITY4
	Bfunction4 (output.temp0, msg.temp0, multiplier);
	Bfunction4 (output.temp1, msg.temp1, multiplier);
	Bfunction4 (output.temp2, msg.temp2, multiplier);
	Bfunction4 (output.temp3, msg.temp3, multiplier);
	Bfunction4 (output.temp4, msg.temp4, multiplier);
	Bfunction4 (output.temp5, msg.temp5, multiplier);
	Bfunction4 (output.temp6, msg.temp6, multiplier);
	Bfunction4 (output.temp7, msg.temp7, multiplier);
#elif INTENSITY5
	Bfunction5 (output.temp0, msg.temp0, multiplier);
	Bfunction5 (output.temp1, msg.temp1, multiplier);
	Bfunction5 (output.temp2, msg.temp2, multiplier);
	Bfunction5 (output.temp3, msg.temp3, multiplier);
	Bfunction5 (output.temp4, msg.temp4, multiplier);
	Bfunction5 (output.temp5, msg.temp5, multiplier);
	Bfunction5 (output.temp6, msg.temp6, multiplier);
	Bfunction5 (output.temp7, msg.temp7, multiplier);
#endif

		write_channel_altera(c2, output);

	}
#elif UNROLL_FACTOR == 16
	#pragma ivdep
	for (int i = 1; i < lll; i+=16) {
   		struct Msg msg = read_channel_altera(c1);
#if INTENSITY1
	Bfunction (output.temp0, msg.temp0, multiplier);
	Bfunction (output.temp1, msg.temp1, multiplier);
	Bfunction (output.temp2, msg.temp2, multiplier);
	Bfunction (output.temp3, msg.temp3, multiplier);
	Bfunction (output.temp4, msg.temp4, multiplier);
	Bfunction (output.temp5, msg.temp5, multiplier);
	Bfunction (output.temp6, msg.temp6, multiplier);
	Bfunction (output.temp7, msg.temp7, multiplier);
	Bfunction (output.temp8, msg.temp8, multiplier);
	Bfunction (output.temp9, msg.temp9, multiplier);
	Bfunction (output.temp10, msg.temp10, multiplier);
	Bfunction (output.temp11, msg.temp11, multiplier);
	Bfunction (output.temp12, msg.temp12, multiplier);
	Bfunction (output.temp13, msg.temp13, multiplier);
	Bfunction (output.temp14, msg.temp14, multiplier);
	Bfunction (output.temp15, msg.temp15, multiplier);
#elif INTENSITY2
	Bfunction2 (output.temp0, msg.temp0, multiplier);
	Bfunction2 (output.temp1, msg.temp1, multiplier);
	Bfunction2 (output.temp2, msg.temp2, multiplier);
	Bfunction2 (output.temp3, msg.temp3, multiplier);
	Bfunction2 (output.temp4, msg.temp4, multiplier);
	Bfunction2 (output.temp5, msg.temp5, multiplier);
	Bfunction2 (output.temp6, msg.temp6, multiplier);
	Bfunction2 (output.temp7, msg.temp7, multiplier);
	Bfunction2 (output.temp8, msg.temp8, multiplier);
	Bfunction2 (output.temp9, msg.temp9, multiplier);
	Bfunction2 (output.temp10, msg.temp10, multiplier);
	Bfunction2 (output.temp11, msg.temp11, multiplier);
	Bfunction2 (output.temp12, msg.temp12, multiplier);
	Bfunction2 (output.temp13, msg.temp13, multiplier);
	Bfunction2 (output.temp14, msg.temp14, multiplier);
	Bfunction2 (output.temp15, msg.temp15, multiplier);
#elif INTENSITY3
	Bfunction3 (output.temp0, msg.temp0, multiplier);
	Bfunction3 (output.temp1, msg.temp1, multiplier);
	Bfunction3 (output.temp2, msg.temp2, multiplier);
	Bfunction3 (output.temp3, msg.temp3, multiplier);
	Bfunction3 (output.temp4, msg.temp4, multiplier);
	Bfunction3 (output.temp5, msg.temp5, multiplier);
	Bfunction3 (output.temp6, msg.temp6, multiplier);
	Bfunction3 (output.temp7, msg.temp7, multiplier);
	Bfunction3 (output.temp8, msg.temp8, multiplier);
	Bfunction3 (output.temp9, msg.temp9, multiplier);
	Bfunction3 (output.temp10, msg.temp10, multiplier);
	Bfunction3 (output.temp11, msg.temp11, multiplier);
	Bfunction3 (output.temp12, msg.temp12, multiplier);
	Bfunction3 (output.temp13, msg.temp13, multiplier);
	Bfunction3 (output.temp14, msg.temp14, multiplier);
	Bfunction3 (output.temp15, msg.temp15, multiplier);
#elif INTENSITY4
	Bfunction4 (output.temp0, msg.temp0, multiplier);
	Bfunction4 (output.temp1, msg.temp1, multiplier);
	Bfunction4 (output.temp2, msg.temp2, multiplier);
	Bfunction4 (output.temp3, msg.temp3, multiplier);
	Bfunction4 (output.temp4, msg.temp4, multiplier);
	Bfunction4 (output.temp5, msg.temp5, multiplier);
	Bfunction4 (output.temp6, msg.temp6, multiplier);
	Bfunction4 (output.temp7, msg.temp7, multiplier);
	Bfunction4 (output.temp8, msg.temp8, multiplier);
	Bfunction4 (output.temp9, msg.temp9, multiplier);
	Bfunction4 (output.temp10, msg.temp10, multiplier);
	Bfunction4 (output.temp11, msg.temp11, multiplier);
	Bfunction4 (output.temp12, msg.temp12, multiplier);
	Bfunction4 (output.temp13, msg.temp13, multiplier);
	Bfunction4 (output.temp14, msg.temp14, multiplier);
	Bfunction4 (output.temp15, msg.temp15, multiplier);
#elif INTENSITY5
	Bfunction5 (output.temp0, msg.temp0, multiplier);
	Bfunction5 (output.temp1, msg.temp1, multiplier);
	Bfunction5 (output.temp2, msg.temp2, multiplier);
	Bfunction5 (output.temp3, msg.temp3, multiplier);
	Bfunction5 (output.temp4, msg.temp4, multiplier);
	Bfunction5 (output.temp5, msg.temp5, multiplier);
	Bfunction5 (output.temp6, msg.temp6, multiplier);
	Bfunction5 (output.temp7, msg.temp7, multiplier);
	Bfunction5 (output.temp8, msg.temp8, multiplier);
	Bfunction5 (output.temp9, msg.temp9, multiplier);
	Bfunction5 (output.temp10, msg.temp10, multiplier);
	Bfunction5 (output.temp11, msg.temp11, multiplier);
	Bfunction5 (output.temp12, msg.temp12, multiplier);
	Bfunction5 (output.temp13, msg.temp13, multiplier);
	Bfunction5 (output.temp14, msg.temp14, multiplier);
	Bfunction5 (output.temp15, msg.temp15, multiplier);
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
	struct Msg output;
#if UNROLL_FACTOR == 1
	#pragma ivdep
	for (int i = 1; i < lll; i++) {
    	MSG msg = read_channel_altera(c2);
#if INTENSITY1
	Bfunction (output.temp0, msg.temp0, multiplier);
#elif INTENSITY2
	Bfunction2 (output.temp0, msg.temp0, multiplier);
#elif INTENSITY3
	Bfunction3 (output.temp0, msg.temp0, multiplier);
#elif INTENSITY4
	Bfunction4 (output.temp0, msg.temp0, multiplier);
#elif INTENSITY5
	Bfunction5 (output.temp0, msg.temp0, multiplier);
#endif

		A[i] = output.temp0;
	}
#elif UNROLL_FACTOR == 2
	#pragma ivdep
	for (int i = 1; i < lll; i+=2) {
   		Msg msg = read_channel_altera(c2);
#if INTENSITY1
	Bfunction (output.temp0, msg.temp0, multiplier);
	Bfunction (output.temp1, msg.temp1, multiplier);
#elif INTENSITY2
	Bfunction2 (output.temp0, msg.temp0, multiplier);
	Bfunction2 (output.temp1, msg.temp1, multiplier);
#elif INTENSITY3
	Bfunction3 (output.temp0, msg.temp0, multiplier);
	Bfunction3 (output.temp1, msg.temp1, multiplier);
#elif INTENSITY4
	Bfunction4 (output.temp0, msg.temp0, multiplier);
	Bfunction4 (output.temp1, msg.temp1, multiplier);
#elif INTENSITY5
	Bfunction5 (output.temp0, msg.temp0, multiplier);
	Bfunction5 (output.temp1, msg.temp1, multiplier);
#endif

		A[i] = output.temp0;
    	A[i+1] = output.temp1;
	}
#elif UNROLL_FACTOR == 4
	#pragma ivdep
	for (int i = 1; i < lll; i+=4) {
   		Msg msg = read_channel_altera(c2);
#if INTENSITY1
	Bfunction (output.temp0, msg.temp0, multiplier);
	Bfunction (output.temp1, msg.temp1, multiplier);
	Bfunction (output.temp2, msg.temp2, multiplier);
	Bfunction (output.temp3, msg.temp3, multiplier);
#elif INTENSITY2
	Bfunction2 (output.temp0, msg.temp0, multiplier);
	Bfunction2 (output.temp1, msg.temp1, multiplier);
	Bfunction2 (output.temp2, msg.temp2, multiplier);
	Bfunction2 (output.temp3, msg.temp3, multiplier);
#elif INTENSITY3
	Bfunction3 (output.temp0, msg.temp0, multiplier);
	Bfunction3 (output.temp1, msg.temp1, multiplier);
	Bfunction3 (output.temp2, msg.temp2, multiplier);
	Bfunction3 (output.temp3, msg.temp3, multiplier);
#elif INTENSITY4
	Bfunction4 (output.temp0, msg.temp0, multiplier);
	Bfunction4 (output.temp1, msg.temp1, multiplier);
	Bfunction4 (output.temp2, msg.temp2, multiplier);
	Bfunction4 (output.temp3, msg.temp3, multiplier);
#elif INTENSITY5
	Bfunction5 (output.temp0, msg.temp0, multiplier);
	Bfunction5 (output.temp1, msg.temp1, multiplier);
	Bfunction5 (output.temp2, msg.temp2, multiplier);
	Bfunction5 (output.temp3, msg.temp3, multiplier);
#endif

		A[i] = output.temp0;
    	A[i+1] = output.temp1;
    	A[i+2] = output.temp2;
    	A[i+3] = output.temp3;
	}
#elif UNROLL_FACTOR == 8
	#pragma ivdep
	for (int i = 1; i < lll; i+=8) {
   		struct Msg msg = read_channel_altera(c2);
#if INTENSITY1
	Bfunction (output.temp0, msg.temp0, multiplier);
	Bfunction (output.temp1, msg.temp1, multiplier);
	Bfunction (output.temp2, msg.temp2, multiplier);
	Bfunction (output.temp3, msg.temp3, multiplier);
	Bfunction (output.temp4, msg.temp4, multiplier);
	Bfunction (output.temp5, msg.temp5, multiplier);
	Bfunction (output.temp6, msg.temp6, multiplier);
	Bfunction (output.temp7, msg.temp7, multiplier);
#elif INTENSITY2
	Bfunction2 (output.temp0, msg.temp0, multiplier);
	Bfunction2 (output.temp1, msg.temp1, multiplier);
	Bfunction2 (output.temp2, msg.temp2, multiplier);
	Bfunction2 (output.temp3, msg.temp3, multiplier);
	Bfunction2 (output.temp4, msg.temp4, multiplier);
	Bfunction2 (output.temp5, msg.temp5, multiplier);
	Bfunction2 (output.temp6, msg.temp6, multiplier);
	Bfunction2 (output.temp7, msg.temp7, multiplier);
#elif INTENSITY3
	Bfunction3 (output.temp0, msg.temp0, multiplier);
	Bfunction3 (output.temp1, msg.temp1, multiplier);
	Bfunction3 (output.temp2, msg.temp2, multiplier);
	Bfunction3 (output.temp3, msg.temp3, multiplier);
	Bfunction3 (output.temp4, msg.temp5, multiplier);
	Bfunction3 (output.temp5, msg.temp6, multiplier);
	Bfunction3 (output.temp6, msg.temp7, multiplier);
	Bfunction3 (output.temp7, msg.temp8, multiplier);
#elif INTENSITY4
	Bfunction4 (output.temp0, msg.temp0, multiplier);
	Bfunction4 (output.temp1, msg.temp1, multiplier);
	Bfunction4 (output.temp2, msg.temp2, multiplier);
	Bfunction4 (output.temp3, msg.temp3, multiplier);
	Bfunction4 (output.temp4, msg.temp4, multiplier);
	Bfunction4 (output.temp5, msg.temp5, multiplier);
	Bfunction4 (output.temp6, msg.temp6, multiplier);
	Bfunction4 (output.temp7, msg.temp7, multiplier);
#elif INTENSITY5
	Bfunction5 (output.temp0, msg.temp0, multiplier);
	Bfunction5 (output.temp1, msg.temp1, multiplier);
	Bfunction5 (output.temp2, msg.temp2, multiplier);
	Bfunction5 (output.temp3, msg.temp3, multiplier);
	Bfunction5 (output.temp4, msg.temp4, multiplier);
	Bfunction5 (output.temp5, msg.temp5, multiplier);
	Bfunction5 (output.temp6, msg.temp6, multiplier);
	Bfunction5 (output.temp7, msg.temp7, multiplier);
#endif


		A[i] = output.temp0;
    	A[i+1] = output.temp1;
    	A[i+2] = output.temp2;
    	A[i+3] = output.temp3;
    	A[i+4] = output.temp4;
    	A[i+5] = output.temp5;
    	A[i+6] = output.temp6;
      A[i+7] = output.temp7;
	}
#elif UNROLL_FACTOR == 16
	#pragma ivdep
	for (int i = 1; i < lll; i+=16) {
   		struct Msg msg = read_channel_altera(c2);
#if INTENSITY1
	Bfunction (output.temp0, msg.temp0, multiplier);
	Bfunction (output.temp1, msg.temp1, multiplier);
	Bfunction (output.temp2, msg.temp2, multiplier);
	Bfunction (output.temp3, msg.temp3, multiplier);
	Bfunction (output.temp4, msg.temp4, multiplier);
	Bfunction (output.temp5, msg.temp5, multiplier);
	Bfunction (output.temp6, msg.temp6, multiplier);
	Bfunction (output.temp7, msg.temp7, multiplier);
	Bfunction (output.temp8, msg.temp8, multiplier);
	Bfunction (output.temp9, msg.temp9, multiplier);
	Bfunction (output.temp10, msg.temp10, multiplier);
	Bfunction (output.temp11, msg.temp11, multiplier);
	Bfunction (output.temp12, msg.temp12, multiplier);
	Bfunction (output.temp13, msg.temp13, multiplier);
	Bfunction (output.temp14, msg.temp14, multiplier);
	Bfunction (output.temp15, msg.temp15, multiplier);
#elif INTENSITY2
	Bfunction2 (output.temp0, msg.temp0, multiplier);
	Bfunction2 (output.temp1, msg.temp1, multiplier);
	Bfunction2 (output.temp2, msg.temp2, multiplier);
	Bfunction2 (output.temp3, msg.temp3, multiplier);
	Bfunction2 (output.temp4, msg.temp4, multiplier);
	Bfunction2 (output.temp5, msg.temp5, multiplier);
	Bfunction2 (output.temp6, msg.temp6, multiplier);
	Bfunction2 (output.temp7, msg.temp7, multiplier);
	Bfunction2 (output.temp8, msg.temp8, multiplier);
	Bfunction2 (output.temp9, msg.temp9, multiplier);
	Bfunction2 (output.temp10, msg.temp10, multiplier);
	Bfunction2 (output.temp11, msg.temp11, multiplier);
	Bfunction2 (output.temp12, msg.temp12, multiplier);
	Bfunction2 (output.temp13, msg.temp13, multiplier);
	Bfunction2 (output.temp14, msg.temp14, multiplier);
	Bfunction2 (output.temp15, msg.temp15, multiplier);
#elif INTENSITY3
	Bfunction3 (output.temp0, msg.temp0, multiplier);
	Bfunction3 (output.temp1, msg.temp1, multiplier);
	Bfunction3 (output.temp2, msg.temp2, multiplier);
	Bfunction3 (output.temp3, msg.temp3, multiplier);
	Bfunction3 (output.temp4, msg.temp4, multiplier);
	Bfunction3 (output.temp5, msg.temp5, multiplier);
	Bfunction3 (output.temp6, msg.temp6, multiplier);
	Bfunction3 (output.temp7, msg.temp7, multiplier);
	Bfunction3 (output.temp8, msg.temp8, multiplier);
	Bfunction3 (output.temp9, msg.temp9, multiplier);
	Bfunction3 (output.temp10, msg.temp10, multiplier);
	Bfunction3 (output.temp11, msg.temp11, multiplier);
	Bfunction3 (output.temp12, msg.temp12, multiplier);
	Bfunction3 (output.temp13, msg.temp13, multiplier);
	Bfunction3 (output.temp14, msg.temp14, multiplier);
	Bfunction3 (output.temp15, msg.temp15, multiplier);
#elif INTENSITY4
	Bfunction4 (output.temp0, msg.temp0, multiplier);
	Bfunction4 (output.temp1, msg.temp1, multiplier);
	Bfunction4 (output.temp2, msg.temp2, multiplier);
	Bfunction4 (output.temp3, msg.temp3, multiplier);
	Bfunction4 (output.temp4, msg.temp4, multiplier);
	Bfunction4 (output.temp5, msg.temp5, multiplier);
	Bfunction4 (output.temp6, msg.temp6, multiplier);
	Bfunction4 (output.temp7, msg.temp7, multiplier);
	Bfunction4 (output.temp8, msg.temp8, multiplier);
	Bfunction4 (output.temp9, msg.temp9, multiplier);
	Bfunction4 (output.temp10, msg.temp10, multiplier);
	Bfunction4 (output.temp11, msg.temp11, multiplier);
	Bfunction4 (output.temp12, msg.temp12, multiplier);
	Bfunction4 (output.temp13, msg.temp13, multiplier);
	Bfunction4 (output.temp14, msg.temp14, multiplier);
	Bfunction4 (output.temp15, msg.temp15, multiplier);
#elif INTENSITY5
	Bfunction5 (output.temp0, msg.temp0, multiplier);
	Bfunction5 (output.temp1, msg.temp1, multiplier);
	Bfunction5 (output.temp2, msg.temp2, multiplier);
	Bfunction5 (output.temp3, msg.temp3, multiplier);
	Bfunction5 (output.temp4, msg.temp4, multiplier);
	Bfunction5 (output.temp5, msg.temp5, multiplier);
	Bfunction5 (output.temp6, msg.temp6, multiplier);
	Bfunction5 (output.temp7, msg.temp7, multiplier);
	Bfunction5 (output.temp8, msg.temp8, multiplier);
	Bfunction5 (output.temp9, msg.temp9, multiplier);
	Bfunction5 (output.temp10, msg.temp10, multiplier);
	Bfunction5 (output.temp11, msg.temp11, multiplier);
	Bfunction5 (output.temp12, msg.temp12, multiplier);
	Bfunction5 (output.temp13, msg.temp13, multiplier);
	Bfunction5 (output.temp14, msg.temp14, multiplier);
	Bfunction5 (output.temp15, msg.temp15, multiplier);
#endif

		A[i] = output.temp0;
    	A[i+1] = output.temp1;
    	A[i+2] = output.temp2;
    	A[i+3] = output.temp3;
    	A[i+4] = output.temp4;
    	A[i+5] = output.temp5;
    	A[i+6] = output.temp6;
		A[i+7] = output.temp7;
    	A[i+8] = output.temp8;
    	A[i+9] = output.temp9;
    	A[i+10] = output.temp10;
    	A[i+11] = output.temp11;
    	A[i+12] = output.temp12;
    	A[i+13] = output.temp13;
      A[i+14] = output.temp14;
			A[i+15] = output.temp15;
	}


#endif

#endif

}
