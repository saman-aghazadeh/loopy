#ifndef CUDAKERNELEXECUTION_H_
#define CUDAKERNELEXECUTION_H_

#include <map>
#include <string>
using namespace std;

#include "TestS4VfloatD256Form1MUnrol0U0LM0SEQ.h"
#include "TestS4VfloatD256Form1MUnrol0U0LM1SEQ.h"
#include "TestS4VfloatD256Form1MUnrol0U8LM0SEQ.h"
#include "TestS4VfloatD256Form1MUnrol0U8LM1SEQ.h"
#include "TestS4VfloatD256Form1MUnrol0U16LM0SEQ.h"
#include "TestS4VfloatD256Form1MUnrol0U16LM1SEQ.h"
#include "TestS4VfloatD512Form1MUnrol0U0LM0SEQ.h"
#include "TestS4VfloatD512Form1MUnrol0U0LM1SEQ.h"
#include "TestS4VfloatD512Form1MUnrol0U8LM0SEQ.h"
#include "TestS4VfloatD512Form1MUnrol0U8LM1SEQ.h"
#include "TestS4VfloatD512Form1MUnrol0U16LM0SEQ.h"
#include "TestS4VfloatD512Form1MUnrol0U16LM1SEQ.h"
#include "TestS4VfloatD256Form1MUnrol0U0LM0SEQ.h"
#include "TestS4VfloatD256Form1MUnrol0U0LM1SEQ.h"
#include "TestS4VfloatD256Form1MUnrol0U8LM0SEQ.h"
#include "TestS4VfloatD256Form1MUnrol0U8LM1SEQ.h"
#include "TestS4VfloatD256Form1MUnrol0U16LM0SEQ.h"
#include "TestS4VfloatD256Form1MUnrol0U16LM1SEQ.h"
#include "TestS4VfloatD512Form1MUnrol0U0LM0SEQ.h"
#include "TestS4VfloatD512Form1MUnrol0U0LM1SEQ.h"
#include "TestS4VfloatD512Form1MUnrol0U8LM0SEQ.h"
#include "TestS4VfloatD512Form1MUnrol0U8LM1SEQ.h"
#include "TestS4VfloatD512Form1MUnrol0U16LM0SEQ.h"
#include "TestS4VfloatD512Form1MUnrol0U16LM1SEQ.h"
#include "TestS2VfloatD256Form1MUnrol0U0LM0SEQ.h"
#include "TestS2VfloatD256Form1MUnrol0U0LM1SEQ.h"
#include "TestS2VfloatD256Form1MUnrol0U8LM0SEQ.h"
#include "TestS2VfloatD256Form1MUnrol0U8LM1SEQ.h"
#include "TestS2VfloatD256Form1MUnrol0U16LM0SEQ.h"
#include "TestS2VfloatD256Form1MUnrol0U16LM1SEQ.h"
#include "TestS2VfloatD512Form1MUnrol0U0LM0SEQ.h"
#include "TestS2VfloatD512Form1MUnrol0U0LM1SEQ.h"
#include "TestS2VfloatD512Form1MUnrol0U8LM0SEQ.h"
#include "TestS2VfloatD512Form1MUnrol0U8LM1SEQ.h"
#include "TestS2VfloatD512Form1MUnrol0U16LM0SEQ.h"
#include "TestS2VfloatD512Form1MUnrol0U16LM1SEQ.h"
#include "TestS2VfloatD256Form1MUnrol0U0LM0SEQ.h"
#include "TestS2VfloatD256Form1MUnrol0U0LM1SEQ.h"
#include "TestS2VfloatD256Form1MUnrol0U8LM0SEQ.h"
#include "TestS2VfloatD256Form1MUnrol0U8LM1SEQ.h"
#include "TestS2VfloatD256Form1MUnrol0U16LM0SEQ.h"
#include "TestS2VfloatD256Form1MUnrol0U16LM1SEQ.h"
#include "TestS2VfloatD512Form1MUnrol0U0LM0SEQ.h"
#include "TestS2VfloatD512Form1MUnrol0U0LM1SEQ.h"
#include "TestS2VfloatD512Form1MUnrol0U8LM0SEQ.h"
#include "TestS2VfloatD512Form1MUnrol0U8LM1SEQ.h"
#include "TestS2VfloatD512Form1MUnrol0U16LM0SEQ.h"
#include "TestS2VfloatD512Form1MUnrol0U16LM1SEQ.h"
#include "TestS4VfloatD256Form1MUnrol1U0LM0SEQ.h"
#include "TestS4VfloatD256Form1MUnrol1U0LM0RAND.h"
#include "TestS4VfloatD256Form1MUnrol1U0LM1SEQ.h"
#include "TestS4VfloatD256Form1MUnrol1U0LM1RAND.h"
#include "TestS4VfloatD512Form1MUnrol1U0LM0SEQ.h"
#include "TestS4VfloatD512Form1MUnrol1U0LM0RAND.h"
#include "TestS4VfloatD512Form1MUnrol1U0LM1SEQ.h"
#include "TestS4VfloatD512Form1MUnrol1U0LM1RAND.h"
#include "TestS4VfloatD256Form1MUnrol1U0LM0SEQ.h"
#include "TestS4VfloatD256Form1MUnrol1U0LM0RAND.h"
#include "TestS4VfloatD256Form1MUnrol1U0LM1SEQ.h"
#include "TestS4VfloatD256Form1MUnrol1U0LM1RAND.h"
#include "TestS4VfloatD512Form1MUnrol1U0LM0SEQ.h"
#include "TestS4VfloatD512Form1MUnrol1U0LM0RAND.h"
#include "TestS4VfloatD512Form1MUnrol1U0LM1SEQ.h"
#include "TestS4VfloatD512Form1MUnrol1U0LM1RAND.h"
#include "TestS2VfloatD256Form1MUnrol1U0LM0SEQ.h"
#include "TestS2VfloatD256Form1MUnrol1U0LM0RAND.h"
#include "TestS2VfloatD256Form1MUnrol1U0LM1SEQ.h"
#include "TestS2VfloatD256Form1MUnrol1U0LM1RAND.h"
#include "TestS2VfloatD512Form1MUnrol1U0LM0SEQ.h"
#include "TestS2VfloatD512Form1MUnrol1U0LM0RAND.h"
#include "TestS2VfloatD512Form1MUnrol1U0LM1SEQ.h"
#include "TestS2VfloatD512Form1MUnrol1U0LM1RAND.h"
#include "TestS2VfloatD256Form1MUnrol1U0LM0SEQ.h"
#include "TestS2VfloatD256Form1MUnrol1U0LM0RAND.h"
#include "TestS2VfloatD256Form1MUnrol1U0LM1SEQ.h"
#include "TestS2VfloatD256Form1MUnrol1U0LM1RAND.h"
#include "TestS2VfloatD512Form1MUnrol1U0LM0SEQ.h"
#include "TestS2VfloatD512Form1MUnrol1U0LM0RAND.h"
#include "TestS2VfloatD512Form1MUnrol1U0LM1SEQ.h"
#include "TestS2VfloatD512Form1MUnrol1U0LM1RAND.h"

map<string, void(*)(float*, float*, int, int, int, int)> kernelMaps;
void init_kernel_map () {
	kernelMaps[string("TestS4VfloatD256Form1MUnrol0U0LM0SEQ")] = TestS4VfloatD256Form1MUnrol0U0LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol0U0LM1SEQ")] = TestS4VfloatD256Form1MUnrol0U0LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol0U8LM0SEQ")] = TestS4VfloatD256Form1MUnrol0U8LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol0U8LM1SEQ")] = TestS4VfloatD256Form1MUnrol0U8LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol0U16LM0SEQ")] = TestS4VfloatD256Form1MUnrol0U16LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol0U16LM1SEQ")] = TestS4VfloatD256Form1MUnrol0U16LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol0U0LM0SEQ")] = TestS4VfloatD512Form1MUnrol0U0LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol0U0LM1SEQ")] = TestS4VfloatD512Form1MUnrol0U0LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol0U8LM0SEQ")] = TestS4VfloatD512Form1MUnrol0U8LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol0U8LM1SEQ")] = TestS4VfloatD512Form1MUnrol0U8LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol0U16LM0SEQ")] = TestS4VfloatD512Form1MUnrol0U16LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol0U16LM1SEQ")] = TestS4VfloatD512Form1MUnrol0U16LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol0U0LM0SEQ")] = TestS4VfloatD256Form1MUnrol0U0LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol0U0LM1SEQ")] = TestS4VfloatD256Form1MUnrol0U0LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol0U8LM0SEQ")] = TestS4VfloatD256Form1MUnrol0U8LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol0U8LM1SEQ")] = TestS4VfloatD256Form1MUnrol0U8LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol0U16LM0SEQ")] = TestS4VfloatD256Form1MUnrol0U16LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol0U16LM1SEQ")] = TestS4VfloatD256Form1MUnrol0U16LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol0U0LM0SEQ")] = TestS4VfloatD512Form1MUnrol0U0LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol0U0LM1SEQ")] = TestS4VfloatD512Form1MUnrol0U0LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol0U8LM0SEQ")] = TestS4VfloatD512Form1MUnrol0U8LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol0U8LM1SEQ")] = TestS4VfloatD512Form1MUnrol0U8LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol0U16LM0SEQ")] = TestS4VfloatD512Form1MUnrol0U16LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol0U16LM1SEQ")] = TestS4VfloatD512Form1MUnrol0U16LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol0U0LM0SEQ")] = TestS2VfloatD256Form1MUnrol0U0LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol0U0LM1SEQ")] = TestS2VfloatD256Form1MUnrol0U0LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol0U8LM0SEQ")] = TestS2VfloatD256Form1MUnrol0U8LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol0U8LM1SEQ")] = TestS2VfloatD256Form1MUnrol0U8LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol0U16LM0SEQ")] = TestS2VfloatD256Form1MUnrol0U16LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol0U16LM1SEQ")] = TestS2VfloatD256Form1MUnrol0U16LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol0U0LM0SEQ")] = TestS2VfloatD512Form1MUnrol0U0LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol0U0LM1SEQ")] = TestS2VfloatD512Form1MUnrol0U0LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol0U8LM0SEQ")] = TestS2VfloatD512Form1MUnrol0U8LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol0U8LM1SEQ")] = TestS2VfloatD512Form1MUnrol0U8LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol0U16LM0SEQ")] = TestS2VfloatD512Form1MUnrol0U16LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol0U16LM1SEQ")] = TestS2VfloatD512Form1MUnrol0U16LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol0U0LM0SEQ")] = TestS2VfloatD256Form1MUnrol0U0LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol0U0LM1SEQ")] = TestS2VfloatD256Form1MUnrol0U0LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol0U8LM0SEQ")] = TestS2VfloatD256Form1MUnrol0U8LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol0U8LM1SEQ")] = TestS2VfloatD256Form1MUnrol0U8LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol0U16LM0SEQ")] = TestS2VfloatD256Form1MUnrol0U16LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol0U16LM1SEQ")] = TestS2VfloatD256Form1MUnrol0U16LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol0U0LM0SEQ")] = TestS2VfloatD512Form1MUnrol0U0LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol0U0LM1SEQ")] = TestS2VfloatD512Form1MUnrol0U0LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol0U8LM0SEQ")] = TestS2VfloatD512Form1MUnrol0U8LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol0U8LM1SEQ")] = TestS2VfloatD512Form1MUnrol0U8LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol0U16LM0SEQ")] = TestS2VfloatD512Form1MUnrol0U16LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol0U16LM1SEQ")] = TestS2VfloatD512Form1MUnrol0U16LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol1U0LM0SEQ")] = TestS4VfloatD256Form1MUnrol1U0LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol1U0LM0RAND")] = TestS4VfloatD256Form1MUnrol1U0LM0RAND_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol1U0LM1SEQ")] = TestS4VfloatD256Form1MUnrol1U0LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol1U0LM1RAND")] = TestS4VfloatD256Form1MUnrol1U0LM1RAND_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol1U0LM0SEQ")] = TestS4VfloatD512Form1MUnrol1U0LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol1U0LM0RAND")] = TestS4VfloatD512Form1MUnrol1U0LM0RAND_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol1U0LM1SEQ")] = TestS4VfloatD512Form1MUnrol1U0LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol1U0LM1RAND")] = TestS4VfloatD512Form1MUnrol1U0LM1RAND_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol1U0LM0SEQ")] = TestS4VfloatD256Form1MUnrol1U0LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol1U0LM0RAND")] = TestS4VfloatD256Form1MUnrol1U0LM0RAND_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol1U0LM1SEQ")] = TestS4VfloatD256Form1MUnrol1U0LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD256Form1MUnrol1U0LM1RAND")] = TestS4VfloatD256Form1MUnrol1U0LM1RAND_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol1U0LM0SEQ")] = TestS4VfloatD512Form1MUnrol1U0LM0SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol1U0LM0RAND")] = TestS4VfloatD512Form1MUnrol1U0LM0RAND_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol1U0LM1SEQ")] = TestS4VfloatD512Form1MUnrol1U0LM1SEQ_wrapper;
	kernelMaps[string("TestS4VfloatD512Form1MUnrol1U0LM1RAND")] = TestS4VfloatD512Form1MUnrol1U0LM1RAND_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol1U0LM0SEQ")] = TestS2VfloatD256Form1MUnrol1U0LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol1U0LM0RAND")] = TestS2VfloatD256Form1MUnrol1U0LM0RAND_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol1U0LM1SEQ")] = TestS2VfloatD256Form1MUnrol1U0LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol1U0LM1RAND")] = TestS2VfloatD256Form1MUnrol1U0LM1RAND_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol1U0LM0SEQ")] = TestS2VfloatD512Form1MUnrol1U0LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol1U0LM0RAND")] = TestS2VfloatD512Form1MUnrol1U0LM0RAND_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol1U0LM1SEQ")] = TestS2VfloatD512Form1MUnrol1U0LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol1U0LM1RAND")] = TestS2VfloatD512Form1MUnrol1U0LM1RAND_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol1U0LM0SEQ")] = TestS2VfloatD256Form1MUnrol1U0LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol1U0LM0RAND")] = TestS2VfloatD256Form1MUnrol1U0LM0RAND_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol1U0LM1SEQ")] = TestS2VfloatD256Form1MUnrol1U0LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD256Form1MUnrol1U0LM1RAND")] = TestS2VfloatD256Form1MUnrol1U0LM1RAND_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol1U0LM0SEQ")] = TestS2VfloatD512Form1MUnrol1U0LM0SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol1U0LM0RAND")] = TestS2VfloatD512Form1MUnrol1U0LM0RAND_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol1U0LM1SEQ")] = TestS2VfloatD512Form1MUnrol1U0LM1SEQ_wrapper;
	kernelMaps[string("TestS2VfloatD512Form1MUnrol1U0LM1RAND")] = TestS2VfloatD512Form1MUnrol1U0LM1RAND_wrapper;
}

#endif // CUDAKERNELEXECUTION_h_
