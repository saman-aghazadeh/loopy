__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void PE() 
{

	configuration config = read_channel_intel(pe_configuration_channel);

	// printf ("[FPGA][PE] num_vecs=%d, num_stages=%d\n",
	// 	config.num_vecs, config.num_stages);
	
	
	for (int i = 0; i < config.num_vecs; i++) {
		input_data data = read_channel_intel(input_data_channel);

		#pragma ivdep // just a hack to solve the situation fast
		for (int j = 0; j < config.num_stages; j++) {
		
			output_data buffer_inner;
			#pragma unroll
			for (int v = 0; v < VEC_SIZE; v++) buffer_inner.data[v] = data.data[v];
	 
			weight_data weight = read_channel_intel(weight_data_channel);
			
			// Now the computation
			#pragma unroll
			for (int itrX = 0; itrX < VEC_SIZE; itrX++) {
				#pragma unroll
				for (int itrY = 0; itrY < STAGE_SIZE; itrY++) {
#ifdef	INTENSITY1
					Afunction1(buffer_inner.data[itrX], weight.data[itrY].data[itrX]);
#elif 	INTENSITY2	
					Afunction2(buffer_inner.data[itrX], weight.data[itrY].data[itrX]);
#elif 	INTENSITY3
					Afunction3(buffer_inner.data[itrX], weight.data[itrY].data[itrX]);
#elif 	INTENSITY4
					Afunction4(buffer_inner.data[itrX], weight.data[itrY].data[itrX]);
#elif 	INTENSITY5
					Afunction15(buffer_inner.data[itrX], weight.data[itrY].data[itrX]);
#endif
				}
			}

			pe_acc_data buffer_inner_conv;
			
			#pragma unroll
			for (int v = 0; v < VEC_SIZE; v++) {
				buffer_inner_conv.data[v] = buffer_inner.data[v];
			}

			write_channel_intel(pe_acc_data_channel, buffer_inner_conv);

		}
		
	}	

}
