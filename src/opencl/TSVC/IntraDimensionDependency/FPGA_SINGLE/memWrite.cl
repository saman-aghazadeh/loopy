__kernel
__attribute__((max_global_work_dim(0)))
void memWrite(__global output_data *restrict A)
{

	configuration config = read_channel_intel(memwr_configuration_channel);

	for (int i = 0; i < config.num_vecs; i++) {
		output_data output = read_channel_intel(output_data_channel);
		A[i] = output;
	}

}
