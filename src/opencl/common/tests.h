#ifndef _TESTS_H_
#define _TESTS_H_

#include <vector>
#include <map>
#include <string>


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
};

struct _algorithm_type tests[] = {
	{"Test_S16_Vfloat_I1048576_D64_Form1", 16, 1, vector<int>({1048576}), vector<int>({64}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 2, 32, 128, 2, 1, "float"},
	{"Test_S16_Vfloat_I1048576_D64_Form2", 16, 1, vector<int>({1048576}), vector<int>({64}), false, vector<int>({0}), "temp", "float16 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 128, 2, 1, "float"},
	{0, 0, 0, vector<int>(), vector<int>(), 0, vector<int>(), 0, 0, 0, 0, 0, 0, 0, 0, 0}
};

#endif
