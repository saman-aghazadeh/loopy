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
	const returnFormula;
	const char* formula;
	int halfBufSizeMin;
	int halfBufSizeMax;
	int halfBufSizeStride;
	int localWorkSizeMin;
	int localWorkSizeMax;
	int localWorkSizeStride;
	int flopCount;
	const char* varType;
};struct _algorithm_type tests[] = {
	{"Test_S2_Vfloat_I1048576_D512_Form1", 2, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "float2 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S2_Vfloat_I1048576_D1024_Form1", 2, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "float2 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S2_Vfloat_I1048576_D2048_Form1", 2, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "float2 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S2_Vfloat_I524288_D512_Form1", 2, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "float2 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S2_Vfloat_I524288_D1024_Form1", 2, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "float2 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S2_Vfloat_I524288_D2048_Form1", 2, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "float2 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S4_Vfloat_I1048576_D512_Form1", 4, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "float4 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S4_Vfloat_I1048576_D1024_Form1", 4, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "float4 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S4_Vfloat_I1048576_D2048_Form1", 4, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "float4 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S4_Vfloat_I524288_D512_Form1", 4, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "float4 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S4_Vfloat_I524288_D1024_Form1", 4, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "float4 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S4_Vfloat_I524288_D2048_Form1", 4, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "float4 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S8_Vfloat_I1048576_D512_Form1", 8, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "float8 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S8_Vfloat_I1048576_D1024_Form1", 8, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "float8 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S8_Vfloat_I1048576_D2048_Form1", 8, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "float8 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S8_Vfloat_I524288_D512_Form1", 8, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "float8 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S8_Vfloat_I524288_D1024_Form1", 8, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "float8 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S8_Vfloat_I524288_D2048_Form1", 8, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "float8 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S16_Vfloat_I1048576_D512_Form1", 16, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S16_Vfloat_I1048576_D1024_Form1", 16, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S16_Vfloat_I1048576_D2048_Form1", 16, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S16_Vfloat_I524288_D512_Form1", 16, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S16_Vfloat_I524288_D1024_Form1", 16, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S16_Vfloat_I524288_D2048_Form1", 16, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S2_Vdouble_I1048576_D512_Form1", 2, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "double2 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S2_Vdouble_I1048576_D1024_Form1", 2, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "double2 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S2_Vdouble_I1048576_D2048_Form1", 2, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "double2 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S2_Vdouble_I524288_D512_Form1", 2, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "double2 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S2_Vdouble_I524288_D1024_Form1", 2, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "double2 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S2_Vdouble_I524288_D2048_Form1", 2, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "double2 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S4_Vdouble_I1048576_D512_Form1", 4, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "double4 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S4_Vdouble_I1048576_D1024_Form1", 4, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "double4 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S4_Vdouble_I1048576_D2048_Form1", 4, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "double4 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S4_Vdouble_I524288_D512_Form1", 4, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "double4 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S4_Vdouble_I524288_D1024_Form1", 4, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "double4 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S4_Vdouble_I524288_D2048_Form1", 4, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "double4 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S8_Vdouble_I1048576_D512_Form1", 8, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "double8 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S8_Vdouble_I1048576_D1024_Form1", 8, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "double8 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S8_Vdouble_I1048576_D2048_Form1", 8, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "double8 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S8_Vdouble_I524288_D512_Form1", 8, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "double8 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S8_Vdouble_I524288_D1024_Form1", 8, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "double8 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S8_Vdouble_I524288_D2048_Form1", 8, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "double8 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S16_Vdouble_I1048576_D512_Form1", 16, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "double16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S16_Vdouble_I1048576_D1024_Form1", 16, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "double16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S16_Vdouble_I1048576_D2048_Form1", 16, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "double16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S16_Vdouble_I524288_D512_Form1", 16, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "double16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S16_Vdouble_I524288_D1024_Form1", 16, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "double16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S16_Vdouble_I524288_D2048_Form1", 16, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "double16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (double) rands[!] * @", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S2_Vfloat_I1048576_D512_Form2", 2, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "float2 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S2_Vfloat_I1048576_D1024_Form2", 2, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "float2 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S2_Vfloat_I1048576_D2048_Form2", 2, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "float2 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S2_Vfloat_I524288_D512_Form2", 2, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "float2 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S2_Vfloat_I524288_D1024_Form2", 2, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "float2 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S2_Vfloat_I524288_D2048_Form2", 2, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "float2 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S4_Vfloat_I1048576_D512_Form2", 4, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "float4 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S4_Vfloat_I1048576_D1024_Form2", 4, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "float4 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S4_Vfloat_I1048576_D2048_Form2", 4, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "float4 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S4_Vfloat_I524288_D512_Form2", 4, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "float4 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S4_Vfloat_I524288_D1024_Form2", 4, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "float4 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S4_Vfloat_I524288_D2048_Form2", 4, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "float4 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S8_Vfloat_I1048576_D512_Form2", 8, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "float8 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S8_Vfloat_I1048576_D1024_Form2", 8, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "float8 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S8_Vfloat_I1048576_D2048_Form2", 8, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "float8 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S8_Vfloat_I524288_D512_Form2", 8, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "float8 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S8_Vfloat_I524288_D1024_Form2", 8, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "float8 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S8_Vfloat_I524288_D2048_Form2", 8, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "float8 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S16_Vfloat_I1048576_D512_Form2", 16, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "float16 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S16_Vfloat_I1048576_D1024_Form2", 16, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "float16 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S16_Vfloat_I1048576_D2048_Form2", 16, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "float16 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S16_Vfloat_I524288_D512_Form2", 16, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "float16 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S16_Vfloat_I524288_D1024_Form2", 16, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "float16 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S16_Vfloat_I524288_D2048_Form2", 16, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "float16 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "float"},
	{"Test_S2_Vdouble_I1048576_D512_Form2", 2, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "double2 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S2_Vdouble_I1048576_D1024_Form2", 2, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "double2 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S2_Vdouble_I1048576_D2048_Form2", 2, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "double2 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S2_Vdouble_I524288_D512_Form2", 2, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "double2 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S2_Vdouble_I524288_D1024_Form2", 2, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "double2 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S2_Vdouble_I524288_D2048_Form2", 2, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "double2 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S4_Vdouble_I1048576_D512_Form2", 4, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "double4 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S4_Vdouble_I1048576_D1024_Form2", 4, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "double4 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S4_Vdouble_I1048576_D2048_Form2", 4, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "double4 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S4_Vdouble_I524288_D512_Form2", 4, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "double4 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S4_Vdouble_I524288_D1024_Form2", 4, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "double4 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S4_Vdouble_I524288_D2048_Form2", 4, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "double4 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S8_Vdouble_I1048576_D512_Form2", 8, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "double8 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S8_Vdouble_I1048576_D1024_Form2", 8, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "double8 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S8_Vdouble_I1048576_D2048_Form2", 8, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "double8 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S8_Vdouble_I524288_D512_Form2", 8, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "double8 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S8_Vdouble_I524288_D1024_Form2", 8, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "double8 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S8_Vdouble_I524288_D2048_Form2", 8, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "double8 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S16_Vdouble_I1048576_D512_Form2", 16, 1, vector<int>({1048576}), vector<int>({512}), false, vector<int>({0}), "temp", "double16 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S16_Vdouble_I1048576_D1024_Form2", 16, 1, vector<int>({1048576}), vector<int>({1024}), false, vector<int>({0}), "temp", "double16 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S16_Vdouble_I1048576_D2048_Form2", 16, 1, vector<int>({1048576}), vector<int>({2048}), false, vector<int>({0}), "temp", "double16 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S16_Vdouble_I524288_D512_Form2", 16, 1, vector<int>({524288}), vector<int>({512}), false, vector<int>({0}), "temp", "double16 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S16_Vdouble_I524288_D1024_Form2", 16, 1, vector<int>({524288}), vector<int>({1024}), false, vector<int>({0}), "temp", "double16 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
	{"Test_S16_Vdouble_I524288_D2048_Form2", 16, 1, vector<int>({524288}), vector<int>({2048}), false, vector<int>({0}), "temp", "double16 temp$", "temp0 = data[gid]", "data[gid] = temp0.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 2, 32, 512, 2, 1, "double"},
};

#endif
