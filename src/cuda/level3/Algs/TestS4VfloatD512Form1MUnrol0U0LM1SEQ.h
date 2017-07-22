#ifndef TestS4VfloatD512Form1MUnrol0U0LM1SEQ_H_
#define TestS4VfloatD512Form1MUnrol0U0LM1SEQ_H_

__global__ void TestS4VfloatD512Form1MUnrol0U0LM1SEQ( float *data, float *rands, int index, int rand_max){
	float4 temp;
	__shared__ float localRands[512];
	int depth = 512;

	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int lid = threadIdx.x;
	int blockSize = blockDim.x;
	for(int i = lid; i < depth; i += blockSize) {
		localRands[i] = rands[i];
	}
	__syncthreads();



	temp.x = data[gid];
	temp.y = data[gid];
	temp.z = data[gid];
	temp.w = data[gid];;
	for (int i = 0; i < 512; i++){
		temp.x = (float) localRands[i] * temp.x + localRands[i];
		temp.y = (float) localRands[i] * temp.y + localRands[i];
		temp.z = (float) localRands[i] * temp.z + localRands[i];
		temp.w = (float) localRands[i] * temp.w + localRands[i];;
	}
	data[gid] = temp.x + temp.y + temp.z + temp.w;

}


void TestS4VfloatD512Form1MUnrol0U0LM1SEQ_wrapper (float *data, float *rands, int index, int rand_max, int numBlocks, int threadPerBlock) {
	TestS4VfloatD512Form1MUnrol0U0LM1SEQ<<<numBlocks, threadPerBlock>>> (data, rands, index, rand_max);
}

#endif 
