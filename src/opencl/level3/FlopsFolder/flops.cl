__kernel void flops(__global float *data) {

	int gid = get_global_id(0);
  double s = data[gid];
 	s = s *0.35;
  data[gid] = s;
}