__kernel
__attribute__((max_global_work_dim(0)))
void memReadData(__global input_data *restrict A)
{

	configuration config = read_channel_intel(memrd_data_configuration_channel);
	
	for (int i = 0; i < config.num_vecs; i++) {
		input_data data = A[i];

		write_channel_intel(input_data_channel, data);
	}

}
