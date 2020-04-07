__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE() 
{

	configuration config = read_channel_intel(pe_configuration_channel);

	for (int i = 0; i < config.num_vecs; i++) {
		output_data buffer;
		input_data data = read_channel_intel(input_data_channel);

		#pragma unroll
		for (int v = 0; v < VEC_SIZE; v++) buffer.data[v] = data.data[v];

		for (int j = 0; j < config.num_stages; j++) {
			weight_data weight = read_channel_intel(weight_data_channel);
			
			// Now the computation
			#pragma unroll
			for (int itrX = 0; itrX < VEC_SIZE; itrX++) {
				#pragma unroll
				for (int itrY = 0; itrY < STAGE_SIZE; itrY++) {
					buffer.data[itrX] += buffer.data[itrX] * weight.data[itrY].data[itrX];
				}
			}	
		}
		
		write_channel_intel(output_data_channel, buffer);
	}	

}
