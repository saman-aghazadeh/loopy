#ifndef CUDAKERNELEXECUTION_H_
#define CUDAKERNELEXECUTION_H_

#include <map>
#include <string>
using namespace std;

#include "GAP0-FloatParam-DepDegree1/kernelWGSXMAPIXLLXOPS8.h";
#include "GAP0-FloatParam-DepDegree1/kernelWGSXMAPIXLLXOPS16.h";
#include "GAP0-FloatParam-DepDegree1/kernelWGSXMAPIXLLXOPS32.h";
#include "GAP0-FloatParam-DepDegree1/kernelWGSXMAPIXLLXOPS64.h";
#include "GAP0-FloatParam-DepDegree1/kernelWGSXMAPIXLLXOPS128.h";
#include "GAP0-FloatParam-DepDegree1/kernelWGSXMAPIXLLXOPS256.h";
#include "GAP0-FloatParam-DepDegree1/kernelWGSXMAPIXLLXOPS512.h";
#include "GAP0-FloatParam-DepDegree1/kernelWGSXMAPIXLLXOPS1024.h";

map<string, void(*)(const float*, float*, const float, const float, const float, int, int)> kernelMaps;
void init_kernel_map () {
	kernelMaps[string("kernelWGSXMAPIXLLXOPS8")] = WGSXMAPIXLLXOPS8_wrapper;
	kernelMaps[string("kernelWGSXMAPIXLLXOPS16")] = WGSXMAPIXLLXOPS16_wrapper;
	kernelMaps[string("kernelWGSXMAPIXLLXOPS32")] = WGSXMAPIXLLXOPS32_wrapper;
	kernelMaps[string("kernelWGSXMAPIXLLXOPS64")] = WGSXMAPIXLLXOPS64_wrapper;
	kernelMaps[string("kernelWGSXMAPIXLLXOPS128")] = WGSXMAPIXLLXOPS128_wrapper;
	kernelMaps[string("kernelWGSXMAPIXLLXOPS256")] = WGSXMAPIXLLXOPS256_wrapper;
	kernelMaps[string("kernelWGSXMAPIXLLXOPS512")] = WGSXMAPIXLLXOPS512_wrapper;
	kernelMaps[string("kernelWGSXMAPIXLLXOPS1024")] = WGSXMAPIXLLXOPS1024_wrapper;
}

#endif // CUDAKERNELEXECUTION_h_
