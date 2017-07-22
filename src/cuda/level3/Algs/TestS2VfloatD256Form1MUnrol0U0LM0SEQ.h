#ifndef TestS2VfloatD256Form1MUnrol0U0LM0SEQ_H_
#define TestS2VfloatD256Form1MUnrol0U0LM0SEQ_H_

__global__ void TestS2VfloatD256Form1MUnrol0U0LM0SEQ( float *data, float *rands, int index, int rand_max){
	float2 temp;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	temp.x = data[gid];
	temp.y = data[gid];;
	for (int i = 0; i < 256; i++){
		temp.x = (float) rands[i] * temp.x + rands[i];
		temp.y = (float) rands[i] * temp.y + rands[i];;
	}
	data[gid] = temp.x + temp.y;

}


void TestS2VfloatD256Form1MUnrol0U0LM0SEQ_wrapper (float *data, float *rands, int index, int rand_max, int numBlocks, int threadPerBlock) {
	TestS2VfloatD256Form1MUnrol0U0LM0SEQ<<<numBlocks, threadPerBlock>>> (data, rands, index, rand_max);
}

#endif 
