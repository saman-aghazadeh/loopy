__kernel
__attribute__((max_global_work_dim(0)))
void memReadWeight(__global weight_data *restrict A)
{

	configuration config = read_channel_intel(memrd_weight_configuration_channel);

	// printf ("FPGA][MEM_READ_WEIGHT] num_vecs=%d, num_stages=%d\n",
	// 	config.num_vecs, config.num_stages);
	
	int total = config.num_vecs * config.num_stages;

	for (int i = 0; i < total; i++) {
		weight_data weight = A[i];
		write_channel_intel(weight_data_channel, weight);
	}

}
