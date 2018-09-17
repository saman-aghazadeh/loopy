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

#if UNROLL_FACTOR==1
channel float c0 __attribute__((depth(4)));
#elif UNROLL_FACTOR==2
channel float c0 __attribute__((depth(4)));
channel float c1 __attribute__((depth(4)));
#elif UNROLL_FACTOR==4
channel float c0 __attribute__((depth(4)));
channel float c1 __attribute__((depth(4)));
channel float c2 __attribute__((depth(4)));
channel float c3 __attribute__((depth(4)));
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

#ifdef UNROLL_FACTOR==1
#pragma ivdep
for (int i = 1; i < (lll-1); i++) {
	DTYPE temp = B[i+1] - E[i] * D[i];
	write_channel_altera (c0, temp);
}
elif UNROLL_FACTOR==2
#pragma ivdep
for (int i = 1; i < (lll-1); i+=2) {
	DTYPE temp0 = B[i+1] - E[i] * D[i];
  DTYPE temp1 = B[i+2] - E[i] * D[i];

	write_channel_altera (c0, temp0);
  write_channel_altera (c1, temp1);
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

#if UNROLL_FACTOR==1
	#pragma ivdep
	for (int i = 2; i < lll; i++) {
  	DTYPE temp;
    temp = read_channel_altera(c0);
		A[i] = temp + C[i] * D[i];
	}
#elif UNROLL_FACTOR==2
	#pragma ivdep
	for (int i = 2; i < lll; i+=2) {
		DTYPE temp0;
    DTYPE temp1;
   	temp0 = read_channel_altera(c0);
		temp1 = read_channel_altera(c1);

		A[i] = temp0 + C[i] * D[i];
    A[i+1] = temp1 + C[i+1] * D[i*1];
	}
#endif

#endif

#endif

}
