__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE_acc()
{

	configuration config = read_channel_intel(pe_acc_configuration_channel);

	for (int i = 0; i < config.num_vecs; i++) {
		output_data data;

		#pragma unroll
		for (int v = 0; v < VEC_SIZE; v++) {
			data.data[v] = 0;
		}

		for (int j = 0; j < config.num_stages; j++) {
			
			pe_acc_data pe_data = read_channel_intel(pe_acc_data_channel);
			
			#pragma unroll
			for (int v = 0; v < VEC_SIZE; v++) {
				data.data[v] += pe_data.data[v];
			}	
			
		}

		write_channel_intel(output_data_channel, data);	
	}

}
