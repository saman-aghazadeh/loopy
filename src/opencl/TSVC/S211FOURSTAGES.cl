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

	DTYPE multiplier = 1.5;
	const int gid = get_global_id(0);
  const int index = gid+1;

	BPrime[index+4] = B[index+5] * multiplier;

#endif

#ifdef FPGA_NDRANGE

#endif

#ifdef FPGA_SINGLE

DTYPE multiplier = 1.5;

#if UNROLL_FACTOR == 1
#pragma ivdep
for (int i = 1; i < lll; i++) {
	DTYPE temp = B[i+1] * multiplier;
	Msg msg;
  msg.temp0 = temp;
	write_channel_altera (c0, msg);
}
#elif UNROLL_FACTOR == 2
#pragma ivdep
for (int i = 1; i < lll; i+=2) {
	DTYPE temp0 = B[i+1] * multiplier;
  	DTYPE temp1 = B[i+2] * multiplier;

	Msg msg;
  msg.temp0 = temp0;
  msg.temp1 = temp1;

	write_channel_altera (c0, msg);
}
#elif UNROLL_FACTOR == 4
#pragma ivdep
for (int i = 1; i < lll; i+=4) {
	DTYPE temp0 = B[i+1] * multiplier;
  	DTYPE temp1 = B[i+2] * multiplier;
	DTYPE temp2 = B[i+3] * multiplier;
 	DTYPE temp3 = B[i+4] * multiplier;

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
	DTYPE temp0 = B[i+1] * multiplier;
  DTYPE temp1 = B[i+2] * multiplier;
	DTYPE temp2 = B[i+3] * multiplier;
 	DTYPE temp3 = B[i+4] * multiplier;
	DTYPE temp4 = B[i+5] * multiplier;
	DTYPE temp5 = B[i+6] * multiplier;
  DTYPE temp6 = B[i+7] * multiplier;
	DTYPE temp7 = B[i+8] * multiplier;

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
	DTYPE temp0 = B[i+1] * multiplier;
  DTYPE temp1 = B[i+2] * multiplier;
	DTYPE temp2 = B[i+3] * multiplier;
 	DTYPE temp3 = B[i+4] * multiplier;
	DTYPE temp4 = B[i+5] * multiplier;
	DTYPE temp5 = B[i+6] * multiplier;
  DTYPE temp6 = B[i+7] * multiplier;
	DTYPE temp7 = B[i+8] * multiplier;
	DTYPE temp8 = B[i+9] * multiplier;
  DTYPE temp9 = B[i+10] * multiplier;
	DTYPE temp10 = B[i+11] * multiplier;
 	DTYPE temp11 = B[i+12] * multiplier;
	DTYPE temp12 = B[i+13] * multiplier;
	DTYPE temp13 = B[i+14] * multiplier;
  DTYPE temp14 = B[i+15] * multiplier;
	DTYPE temp15 = B[i+16] * multiplier;

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

	DTYPE multiplier = 2.5;
	const int gid = get_global_id(0);
  const int index = gid+1;

	B[i+2] = BPrime[i+3] * multiplier;

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

		output.temp0 = msg.temp0 * multiplier;
    write_channel_altera(c1);
	}
#elif UNROLL_FACTOR == 2
	#pragma ivdep
	for (int i = 1; i < lll; i+=2) {
   		Msg msg = read_channel_altera(c0);

		output.temp0 = msg.temp0 * multiplier;
    output.temp1 = msg.temp1 * multiplier;
    write_channel_altera(c1, output);
	}
#elif UNROLL_FACTOR == 4
	#pragma ivdep
	for (int i = 1; i < lll; i+=4) {
   		Msg msg = read_channel_altera(c0);

		output.temp0 = msg.temp0 * multiplier;
   	output.temp1 = msg.temp1 * multiplier;
    output.temp2 = msg.temp2 * multiplier;
    output.temp3 = msg.temp3 * multiplier;

		write_channel_altera(c1, output);
	}
#elif UNROLL_FACTOR == 8
	#pragma ivdep
	for (int i = 1; i < lll; i+=8) {
   		struct Msg msg = read_channel_altera(c0);

		output.temp0 = msg.temp0 * multiplier;
   	output.temp1 = msg.temp1 * multiplier;
    output.temp2 = msg.temp2 * multiplier;
    output.temp3 = msg.temp3 * multiplier;
    output.temp4 = msg.temp4 * multiplier;
    output.temp5 = msg.temp5 * multiplier;
    output.temp6 = msg.temp6 * multiplier;
    output.temp7 = msg.temp7 * multiplier;

		write_channel_altera(c1, output);

	}
#elif UNROLL_FACTOR == 16
	#pragma ivdep
	for (int i = 1; i < lll; i+=16) {
   		struct Msg msg = read_channel_altera(c0);

		output.temp0 = msg.temp0 * multiplier;
    output.temp1 = msg.temp1 * multiplier;
    output.temp2 = msg.temp2 * multiplier;
    output.temp3 = msg.temp3 * multiplier;
    output.temp4 = msg.temp4 * multiplier;
   	output.temp5 = msg.temp5 * multiplier;
    output.temp6 = msg.temp6 * multiplier;
		output.temp7 = msg.temp7 * multiplier;
    output.temp8 = msg.temp8 * multiplier;
   	output.temp9 = msg.temp9 * multiplier;
    output.temp10 = msg.temp10 * multiplier;
    output.temp11 = msg.temp11 * multiplier;
    output.temp12 = msg.temp12 * multiplier;
    output.temp13 = msg.temp13 * multiplier;
    output.temp14 = msg.temp14 * multiplier;
		output.temp15 = msg.temp15 * multiplier;

		write_channel_altera(c1, output);

	}


#endif

#endif

}

__kernel void S211K3 (__global DTYPE* restrict A,
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
	DTYPE multiplier = 3.5;
  const int gid = get_global_id(0);
  const int index = gid+1;

	BPrime[i] = B[i+1] * multiplier;

#endif

#ifdef FPGA_NDRANGE

#endif

#ifdef FPGA_SINGLE

  DTYPE multiplier = 2.5;		
	struct Msg output;


#if UNROLL_FACTOR == 1
	#pragma ivdep
	for (int i = 1; i < lll; i++) {
    	MSG msg = read_channel_altera(c1);

		output.temp0 = msg.temp0 * multiplier;
    write_channel_altera(c2, output);
	}
#elif UNROLL_FACTOR == 2
	#pragma ivdep
	for (int i = 1; i < lll; i+=2) {
   		Msg msg = read_channel_altera(c1);

		output.temp0 = msg.temp0 * multiplier;
    output.temp1 = msg.temp1 * multiplier;
    write_channel_altera(c2, output);
	}
#elif UNROLL_FACTOR == 4
	#pragma ivdep
	for (int i = 1; i < lll; i+=4) {
   		Msg msg = read_channel_altera(c1);

		output.temp0 = msg.temp0 * multiplier;
   	output.temp1 = msg.temp1 * multiplier;
    output.temp2 = msg.temp2 * multiplier;
    output.temp3 = msg.temp3 * multiplier;

		write_channel_altera(c2, output);
	}
#elif UNROLL_FACTOR == 8
	#pragma ivdep
	for (int i = 1; i < lll; i+=8) {
   		struct Msg msg = read_channel_altera(c1);

		output.temp0 = msg.temp0 * multiplier;
   	output.temp1 = msg.temp1 * multiplier;
    output.temp2 = msg.temp2 * multiplier;
    output.temp3 = msg.temp3 * multiplier;
    output.temp4 = msg.temp4 * multiplier;
    output.temp5 = msg.temp5 * multiplier;
    output.temp6 = msg.temp6 * multiplier;
    output.temp7 = msg.temp7 * multiplier;

		write_channel_altera(c2, output);

	}
#elif UNROLL_FACTOR == 16
	#pragma ivdep
	for (int i = 1; i < lll; i+=16) {
   		struct Msg msg = read_channel_altera(c1);

		output.temp0 = msg.temp0 * multiplier;
    output.temp1 = msg.temp1 * multiplier;
    output.temp2 = msg.temp2 * multiplier;
    output.temp3 = msg.temp3 * multiplier;
    output.temp4 = msg.temp4 * multiplier;
   	output.temp5 = msg.temp5 * multiplier;
    output.temp6 = msg.temp6 * multiplier;
		output.temp7 = msg.temp7 * multiplier;
    output.temp8 = msg.temp8 * multiplier;
   	output.temp9 = msg.temp9 * multiplier;
    output.temp10 = msg.temp10 * multiplier;
    output.temp11 = msg.temp11 * multiplier;
    output.temp12 = msg.temp12 * multiplier;
    output.temp13 = msg.temp13 * multiplier;
    output.temp14 = msg.temp14 * multiplier;
		output.temp15 = msg.temp15 * multiplier;

		write_channel_altera(c2, output);
	
	}


#endif

#endif

}

__kernel void S211K4 (__global DTYPE* restrict A,
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

	DTYPE multiplier = 4.5;
  const int gid = get_global_id(0);
  cont int index = gid+1;

	A[i] = BPrime[gid-1] * multiplier;

#endif

#ifdef FPGA_NDRANGE

#endif

#ifdef FPGA_SINGLE

  DTYPE multiplier = 2.5;		

#if UNROLL_FACTOR == 1
	#pragma ivdep
	for (int i = 1; i < lll; i++) {
    	MSG msg = read_channel_altera(c2);

		A[i] = msg.temp0 * multiplier;
	}
#elif UNROLL_FACTOR == 2
	#pragma ivdep
	for (int i = 1; i < lll; i+=2) {
   		Msg msg = read_channel_altera(c2);

		A[i] = msg.temp0 * multiplier;
    	A[i+1] = msg.temp1 * multiplier;
	}
#elif UNROLL_FACTOR == 4
	#pragma ivdep
	for (int i = 1; i < lll; i+=4) {
   		Msg msg = read_channel_altera(c2);

		A[i] = msg.temp0 * multiplier;
    	A[i+1] = msg.temp1 * multiplier;
    	A[i+2] = msg.temp2 * multiplier;
    	A[i+3] = msg.temp3 * multiplier;
	}
#elif UNROLL_FACTOR == 8
	#pragma ivdep
	for (int i = 1; i < lll; i+=8) {
   		struct Msg msg = read_channel_altera(c2);

		A[i] = msg.temp0 * multiplier;
    	A[i+1] = msg.temp1 * multiplier;
    	A[i+2] = msg.temp2 * multiplier;
    	A[i+3] = msg.temp3 * multiplier;
    	A[i+4] = msg.temp4 * multiplier;
    	A[i+5] = msg.temp5 * multiplier;
    	A[i+6] = msg.temp6 * multiplier;
      A[i+7] = msg.temp7 * multiplier;
	}
#elif UNROLL_FACTOR == 16
	#pragma ivdep
	for (int i = 1; i < lll; i+=16) {
   		struct Msg msg = read_channel_altera(c2);

		A[i] = msg.temp0 * multiplier;
    	A[i+1] = msg.temp1 * multiplier;
    	A[i+2] = msg.temp2 * multiplier;
    	A[i+3] = msg.temp3 * multiplier;
    	A[i+4] = msg.temp4 * multiplier;
    	A[i+5] = msg.temp5 * multiplier;
    	A[i+6] = msg.temp6 * multiplier;
		A[i+7] = msg.temp7 * multiplier;
    	A[i+8] = msg.temp8 * multiplier;
    	A[i+9] = msg.temp9 * multiplier;
    	A[i+10] = msg.temp10 * multiplier;
    	A[i+11] = msg.temp11 * multiplier;
    	A[i+12] = msg.temp12 * multiplier;
    	A[i+13] = msg.temp13 * multiplier;
      A[i+14] = msg.temp14 * multiplier;
			A[i+15] = msg.temp15 * multiplier;
	}


#endif

#endif

}
