#ifndef TestS4VfloatD512Form1MUnrol0U16LM0SEQ_H_
#define TestS4VfloatD512Form1MUnrol0U16LM0SEQ_H_

__global__ void TestS4VfloatD512Form1MUnrol0U16LM0SEQ( float *data, float *rands, int index, int rand_max){
	float4 temp;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	temp.x = data[gid];
	temp.y = data[gid];
	temp.z = data[gid];
	temp.w = data[gid];;
	#pragma unroll 16
	for (int i = 0; i < 512; i++){
		temp.x = (float) rands[i] * temp.x + rands[i];
		temp.y = (float) rands[i] * temp.y + rands[i];
		temp.z = (float) rands[i] * temp.z + rands[i];
		temp.w = (float) rands[i] * temp.w + rands[i];;
	}
	data[gid] = temp.x + temp.y + temp.z + temp.w;

}


void TestS4VfloatD512Form1MUnrol0U16LM0SEQ_wrapper (float *data, float *rands, int index, int rand_max, int numBlocks, int threadPerBlock) {
	TestS4VfloatD512Form1MUnrol0U16LM0SEQ<<<numBlocks, threadPerBlock>>> (data, rands, index, rand_max);
}

#endif 
