__kernel
__attribute__((max_global_work_dim(0)))

void controller(
	int num_vecs,
	int num_stages)

{

	// printf ("[FPGA][CONTROLLER] num_vecs=%d, num_stages=%d\n",
	//	num_vecs, num_stages);
	
	configuration config;
	config.num_vecs = num_vecs;
	config.num_stages = num_stages;

	write_channel_intel(memrd_data_configuration_channel, config);
	write_channel_intel(memrd_weight_configuration_channel, config);
	write_channel_intel(pe_configuration_channel, config);
	write_channel_intel(pe_acc_configuration_channel, config);
	write_channel_intel(memwr_configuration_channel, config);

}
