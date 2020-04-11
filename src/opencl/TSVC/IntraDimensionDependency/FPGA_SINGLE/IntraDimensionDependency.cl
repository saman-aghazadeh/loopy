#include "constants.h"
#include "../funcs.h"

#ifdef INT_PRECISION
#define DTYPE int
#elif SINGLE_PRECISION
#define DTYPE float
#elif DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define DTYPE double
#endif

#pragma OPENCL EXTENSION cl_intel_channels : enable

typedef struct {
	DTYPE data[VEC_SIZE];
} input_data;

typedef struct {
	DTYPE data[VEC_SIZE];
} weight_data_inner;

typedef struct {
	weight_data_inner data[STAGE_SIZE];
} weight_data;

typedef struct {
	DTYPE data[VEC_SIZE];
} output_data;

typedef struct {
	DTYPE data[VEC_SIZE];
} pe_acc_data;

typedef struct {
	int num_stages;
	int num_vecs;
} configuration;

channel configuration 	memrd_data_configuration_channel;
channel configuration 	memrd_weight_configuration_channel;
channel configuration	pe_configuration_channel;
channel configuration	pe_acc_configuration_channel;
channel configuration 	memwr_configuration_channel;

channel input_data	input_data_channel;
channel weight_data	weight_data_channel;
channel	pe_acc_data	pe_acc_data_channel;
channel output_data	output_data_channel;

#include "controller.cl"
#include "memReadData.cl"
#include "memReadWeight.cl"
#include "PE.cl"
#include "PE_acc.cl"
#include "memWrite.cl"
