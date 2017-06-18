#ifndef _TESTS_H_
#define _TESTS_H_

#include <vector>
#include <map>
#include <string>
using namespace std;


struct _algorithm_type {
	const char* name;
	int vectorSize;
	int numLoops;
	vector<int> loopsLengths;
	vector<int> loopsDepth;
	bool loopCarriedDataDependency;
	vector<int> loopCarriedDDLengths;
	const char* variable;
	const char* varDeclFormula;
	const char* varInitFormula;
	const char* returnFormula;
	const char* formula;
	int halfBufSizeMin;
	int halfBufSizeMax;
	int halfBufSizeStride;
	int localWorkSizeMin;
	int localWorkSizeMax;
	int localWorkSizeStride;
	int flopCount;
	const char* varType;
	bool doManualUnroll;
	bool doLocalMemory;
	int unrollFactor;
};

struct _algorithm_type tests[] = {
	{"TestS16VfloatI1048576D32Form1MUnrol0U0", 16, 1, vector<int>({1048576}), vector<int>({32}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 0},
	{"TestS16VfloatI1048576D32Form1MUnrol0U8", 16, 1, vector<int>({1048576}), vector<int>({32}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 8},
	{"TestS16VfloatI1048576D32Form1MUnrol0U16", 16, 1, vector<int>({1048576}), vector<int>({32}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 16},
	{"TestS16VfloatI1048576D32Form1MUnrol0U32", 16, 1, vector<int>({1048576}), vector<int>({32}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 32},
	{"TestS16VfloatI1048576D64Form1MUnrol0U0", 16, 1, vector<int>({1048576}), vector<int>({64}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 0},
	{"TestS16VfloatI1048576D64Form1MUnrol0U8", 16, 1, vector<int>({1048576}), vector<int>({64}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 8},
	{"TestS16VfloatI1048576D64Form1MUnrol0U16", 16, 1, vector<int>({1048576}), vector<int>({64}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 16},
	{"TestS16VfloatI1048576D64Form1MUnrol0U32", 16, 1, vector<int>({1048576}), vector<int>({64}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 32},
	{"TestS16VfloatI1048576D128Form1MUnrol0U0", 16, 1, vector<int>({1048576}), vector<int>({128}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 0},
	{"TestS16VfloatI1048576D128Form1MUnrol0U8", 16, 1, vector<int>({1048576}), vector<int>({128}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 8},
	{"TestS16VfloatI1048576D128Form1MUnrol0U16", 16, 1, vector<int>({1048576}), vector<int>({128}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 16},
	{"TestS16VfloatI1048576D128Form1MUnrol0U32", 16, 1, vector<int>({1048576}), vector<int>({128}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 32},
	{"TestS16VfloatI1048576D256Form1MUnrol0U0", 16, 1, vector<int>({1048576}), vector<int>({256}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 0},
	{"TestS16VfloatI1048576D256Form1MUnrol0U8", 16, 1, vector<int>({1048576}), vector<int>({256}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 8},
	{"TestS16VfloatI1048576D256Form1MUnrol0U16", 16, 1, vector<int>({1048576}), vector<int>({256}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 16},
	{"TestS16VfloatI1048576D256Form1MUnrol0U32", 16, 1, vector<int>({1048576}), vector<int>({256}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 0,false, 32},
	{"TestS16VfloatI1048576D32Form1MUnrol1U0", 16, 1, vector<int>({1048576}), vector<int>({32}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 1,false, 0},
	{"TestS16VfloatI1048576D64Form1MUnrol1U0", 16, 1, vector<int>({1048576}), vector<int>({64}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 1,false, 0},
	{"TestS16VfloatI1048576D128Form1MUnrol1U0", 16, 1, vector<int>({1048576}), vector<int>({128}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 1,false, 0},
	{"TestS16VfloatI1048576D256Form1MUnrol1U0", 16, 1, vector<int>({1048576}), vector<int>({256}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 256, 256, 2, 1, "float", 1,false, 0},
	{0, 0, 0, vector<int>(), vector<int>(), 0, vector<int>(), 0, 0, 0, 0, 0, 0, 0, 0, 0, false, false, 0}
};

#endif
