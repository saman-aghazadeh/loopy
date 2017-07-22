#ifndef TestS2VfloatD256Form1MUnrol0U0LM1SEQ_H_
#define TestS2VfloatD256Form1MUnrol0U0LM1SEQ_H_

__global__ void TestS2VfloatD256Form1MUnrol0U0LM1SEQ( float *data, float *rands, int index, int rand_max){
	float2 temp;
	__shared__ float localRands[256];
	int depth = 256;

	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int lid = threadIdx.x;
	int blockSize = blockDim.x;
	for(int i = lid; i < depth; i += blockSize) {
		localRands[i] = rands[i];
	}
	__syncthreads();



	temp.x = data[gid];
	temp.y = data[gid];;
	for (int i = 0; i < 256; i++){
		temp.x = (float) localRands[i] * temp.x + localRands[i];
		temp.y = (float) localRands[i] * temp.y + localRands[i];;
	}
	data[gid] = temp.x + temp.y;

}


void TestS2VfloatD256Form1MUnrol0U0LM1SEQ_wrapper (float *data, float *rands, int index, int rand_max, int numBlocks, int threadPerBlock) {
	TestS2VfloatD256Form1MUnrol0U0LM1SEQ<<<numBlocks, threadPerBlock>>> (data, rands, index, rand_max);
}

#endif 
