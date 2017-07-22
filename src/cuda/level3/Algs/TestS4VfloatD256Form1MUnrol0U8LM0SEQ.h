#ifndef TestS4VfloatD256Form1MUnrol0U8LM0SEQ_H_
#define TestS4VfloatD256Form1MUnrol0U8LM0SEQ_H_

__global__ void TestS4VfloatD256Form1MUnrol0U8LM0SEQ( float *data, float *rands, int index, int rand_max){
	float4 temp;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	temp.x = data[gid];
	temp.y = data[gid];
	temp.z = data[gid];
	temp.w = data[gid];;
	#pragma unroll 8
	for (int i = 0; i < 256; i++){
		temp.x = (float) rands[i] * temp.x + rands[i];
		temp.y = (float) rands[i] * temp.y + rands[i];
		temp.z = (float) rands[i] * temp.z + rands[i];
		temp.w = (float) rands[i] * temp.w + rands[i];;
	}
	data[gid] = temp.x + temp.y + temp.z + temp.w;

}


void TestS4VfloatD256Form1MUnrol0U8LM0SEQ_wrapper (float *data, float *rands, int index, int rand_max, int numBlocks, int threadPerBlock) {
	TestS4VfloatD256Form1MUnrol0U8LM0SEQ<<<numBlocks, threadPerBlock>>> (data, rands, index, rand_max);
}

#endif 
