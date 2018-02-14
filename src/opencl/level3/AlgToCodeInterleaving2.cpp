#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "support.h"
#include "Event.h"
#include "ProgressBar.h"
#include "tests.h"
#include "aocl_utils.h"
#include "ExecutionConfig.h"
#include <time.h>
#include <stdlib.h>
#include <ctime>
#include <OpenCLEngine.h>
#include "Algorithm.h"
#include "AlgorithmFactory.h"
#include "Verify.h"

#define PRIVATE_VECTOR_SIZE 5
#define VERBOSE false
#define VERIFICATION false
#define TEMP_INIT_VALUE 1.0
// Single work item mode
#define SWI_MODE false

using namespace std;

int executionMode = ExecutionMode::CALCULATION;
int targetDevice = TargetDevice::FPGA;

/*
struct _algorithm_type {

	const char* name;					// Name of the algorithm
  int vectorSize;						// Size of the vector which operations gonna
  													// be based on
	int numLoops;							// Total number of nestedLoops
  vector<int> loopsLengths;	// Length of each loop in an array fashion
	vector<int> loopsDepth;		// Depth of operations located inside each
  													// each loop
	bool loopCarriedDataDependency;	// It'll define the existence
  													// of any loop carried data dependency
  vector<int> loopCarriedDDLengths;	// length of operations which will
  													// loop carried data dependency inside the
  													// loop
  const char* variable;			// variable being used in all formulas
  const char* varDeclFormula;	// Declaration formula for the variable
  const char* varInitFormula;	// A formula for initializing the variable
  const char* returnFormula;	// A formula specifying how to return the
  														// calculated value
  const char* formula;			// The formula being used for operations
	int halfBufSizeMin;				// Specify the buffer sizes for which to
  												 	// perform the test
  int halfBufSizeMax;				// We specify the minimum, the maximum and
  													// the geometric stride
  int halfBufSizeStride;		// Geometric stride (Values are in thousands of elements)
	int localWorkSizeMin;			// Minimum size of the work items
  int localWorkSizeMax;			// Maximum size of the work items
  int localWorkSizeStride;	// Geometric stride
  int flopCount;						// Number of floating point operations
  												 	// per linei
  const char* varType;				// Type of variable which is going to be used
  bool doManualUnroll;			// unrolling the loop ourself in the code or
  													// let the compiler do it
	bool doLocalMemory;			  // Copy Data from global memory to local memory
  int unrollFactor;
};
*/

// NOTICE: For current implementation we will always assume we have
// only one foor loop. This is the first implementation assumption
// and gonna be changed in the next phase implementation
/*
struct _algorithm_type tests[] = {
  {"Test11", 16, 1, vector<int>({2097152}), vector<int>({256}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 1024, 128, 512, 2, 1, "float", true, true},
  {"Test12", 16, 1, vector<int>({2097152}), vector<int>({256}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[i] * @", 1024, 1024, 1024, 128, 512, 2, 1, "float", false, true},
  {"Test13", 16, 1, vector<int>({2097152}), vector<int>({256}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 1024, 128, 512, 2, 1, "float", true, false},
  {"Test14", 16, 1, vector<int>({2097152}), vector<int>({256}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[i] * @", 1024, 1024, 1024, 128, 512, 2, 1, "float", false, false},
  {"Test15", 16, 1, vector<int>({2097152}), vector<int>({256}), false, vector<int>({0}), "temp", "float16 temp$", "temp0 = data[gid]", "data[gid] = temp$.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 1024, 128, 512, 2, 1, "float", true, true},
  //{"Test16", 16, 1, vector<int>({2097152}), vector<int>({256}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[i] * @#", 1024, 1024, 1024, 128, 512, 2, 1, "float", false, true},
  {"Test17", 16, 1, vector<int>({2097152}), vector<int>({256}), false, vector<int>({0}), "temp", "float16 temp$", "temp0 = data[gid]", "data[gid] = temp$.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 1024, 128, 512, 2, 1, "float", true, false},
  {"Test18", 16, 1, vector<int>({2097152}), vector<int>({256}), true, vector<int>({128}), "temp", "float16 tempnow; float16 tempbefore", "tempbefore = data[gid]", "data[gid] = tempnow.s0", "tempnow = (float) rands[i] * tempnow + tempbefore", 1024, 1024, 1024, 128, 512, 2, 2, "float", false, false},
  //{"Test18", 16, 1, vector<int>({2097152}), vector<int>({256}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@$ = (float) rands[i] * @#", 1024, 1024, 1024, 128, 512, 2, 1, "float", false, false},
  {"Test21", 4, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), "temp", "float4 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @",1024, 1024, 1024, 32, 512, 2, 1, "float"},
  {"Test22", 4, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), "temp", "float4 temp$", "temp0 = data[gid]", "data[gid] = temp$.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 1024, 32, 512, 2, 1, "float"},
  {"Test31", 8, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), "temp", "float8 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 1024, 32, 512, 2, 1, "float"},
  {"Test32", 8, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), "temp", "float8 temp$" ,"temp0 = data[gid]", "data[gid] = temp$.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 1024, 32, 512, 2, 1, "float"},
  {"Test41", 16, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), "temp", "float16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (float) rands[!] * @", 1024, 1024, 1024, 32, 512, 2, 1, "float"},
  {"Test42", 16, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), "temp", "float16 temp$", "temp0 = data[gid]", "data[gid] = temp$.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 1024, 32, 512, 2, 1, "float"},
  {"Test51", 2, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), "temp", "double2 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (double) rands[!] * @", 1024, 1024, 1024, 32, 512, 2, 1, "double"},
  {"Test52", 2, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), "temp", "double2 temp$", "temp0 = data[gid]", "data[gid] = temp$.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 1024, 32, 512, 2, 1, "double"},
  {"Test61", 4, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), "temp", "double4 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (double) rands[!] * @", 1024, 1024, 1024, 32, 512, 2, 1, "double"},
  {"Test62", 4, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), "temp", "double4 temp$", "temp0 = data[gid]", "data[gid] = temp$.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 1024, 32, 512, 2, 1, "double"},
  {"Test71", 8, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), "temp", "double8 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (double) rands[!] * @", 1024, 1024, 1024, 32, 512, 2, 1, "double"},
  {"Test72", 8, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), "temp", "double8 temp$", "temp0 = data[gid]", "data[gid] = temp$.s0", "@$ = (double) rands[!] * @#", 1024, 1024, 1024, 32, 512, 2, 1, "double"},
  {"Test81", 16, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), "temp", "double16 temp", "temp = data[gid]", "data[gid] = temp.s0", "@ = (double) rands[!] * @", 1024, 1024, 1024, 32, 512, 2, 1, "double"},
  {"Test82", 16, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), "temp", "double16 temp$", "temp0 = data[gid]", "data[gid] = temp$.s0", "@$ = (float) rands[!] * @#", 1024, 1024, 1024, 32, 512, 2, 1, "double"},
  {0, 0, 0, vector<int>(), vector<int>(), 0, vector<int>(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};
*/

// Cleaning up the allocated resources
//void cleanup (cl_program program, cl_kernel kernel, cl_mem memObject) {

//	if (memObject != NULL)
//    clReleaseMemObject (memObject);

//  if (kernel != NULL)
//    clReleaseKernel (kernel);

//  if (program != NULL)
//    clReleaseProgram (program);

//}

void RunBenchmark (cl_device_id id,
                   cl_context ctx,
                   cl_command_queue queue,
                   ResultDatabase &resultDB,
                   OptionParser &op) {

	srand (time(NULL));
	OpenCLEngine<float> openCLEngine (ctx, id, executionMode, targetDevice, tests);

  bool onlyMeta = true;
  if (executionMode == ExecutionMode::GENERATION)
    onlyMeta = false;
  else if (executionMode == ExecutionMode::CALCULATION)
    onlyMeta = true;
  else if (executionMode == ExecutionMode::ALL)
	  onlyMeta = false;

	AlgorithmFactory algorithmFactory;

  int kernelCounter = 1;
  bool localMemory = false;


  /*
  for (int workGroupSize = 256; workGroupSize <= 256; workGroupSize *= 2) {
    for (int memAllocationPerWorkItem = 2;
         memAllocationPerWorkItem <= 2;
         memAllocationPerWorkItem *= 2) {

      //if (workGroupSize * memAllocationPerWorkItem > 1024 && localMemory)
      //  continue;

      //if (workGroupSize * memAllocationPerWorkItem > 4096)
      //  continue;

      for (int loopLength = 65536; loopLength <= 1048576; loopLength *= 2) {
        	int *WGS = new int[1];
        	int *vWGS = new int[1];
        	WGS[0] = workGroupSize;
        	vWGS[0] = workGroupSize;

        	algorithmFactory.createNewAlgorithm ()
          	.targetDeviceIs (AlgorithmTargetDevice::FPGA)
          	.targetLanguageIs (AlgorithmTargetLanguage::OpenCL)
          	.NameIs (string("WGS") + to_string(workGroupSize) +
                   string("MAPI") + to_string(memAllocationPerWorkItem) +
#if SWI_MODE==true
                   string("LL") + to_string(loopLength) +
                   string("OPS") + to_string(1024) +
                   string("SWI"))
#else
            			 string("LL") + to_string(loopLength) +
            			 string("OPS") + to_string(1024))
#endif
        		.KernelNameIs (string("WGS") + string("X") +
                   string("MAPI") + string("X") +
#if SWI_MODE==true
                   string("LL") + to_string(loopLength) +
                   string("OPS") + to_string(1024) +
                   string("SWI"))
#else
                   string("LL") + string("X") +
                   string("OPS") + to_string(1024))
#endif
      			.verificationFunctionIs (verifyWGSXMAPIXLLXOPS1024GAP0)
          	.memAllocationPerWorkItemIs (memAllocationPerWorkItem)
            .workGroupSizeIs (WGS)
            .virtualWorkGroupSizeIs (vWGS)
            .memReuseFactorIs (1024)
            .startKernelFunctionSimpleV1 ()
            .createFor (1024, false, loopLength, "temp1 += temp1 * MF", 1, false, 2, 0)
            .generateForsSimpleV1 (onlyMeta)
            .popMetasSimpleV1 ()
            .endKernelFunction ()
            .verbose ()
            .writeToFile (string("/home/user/sbiookag/Algs/1For/nodep/GAP0-FloatParam-Loop/kernel") +
                     			 string("WGS") + string("X") +
                           string("MAPI") + string("X") +
#if SWI_MODE==true
                           string("LL") + to_string(loopLength) +
#else
                           string("LL") + string("X") +
#endif
	                         string("OPS") + to_string(1024) +
#if SWI_MODE==true
						   						 string("SWI") +
#endif
                           string(".cl"));

             kernelCounter++;
      	}
    }
  }
	*/
	/*
  for (int workGroupSize = 256; workGroupSize <= 256; workGroupSize *= 2) {
    for (int memAllocationPerWorkItem = 2;
         memAllocationPerWorkItem <= 2;
         memAllocationPerWorkItem *= 2) {

      //if (workGroupSize * memAllocationPerWorkItem > 1024 && localMemory)
      //  continue;

      //if (workGroupSize * memAllocationPerWorkItem > 4096)
      //  continue;

      for (int loopLength = 65536; loopLength <= 1048576; loopLength *= 2) {
        //if (loopLength == 262144) break;
        	int *WGS = new int[1];
        	int *vWGS = new int[1];
        	WGS[0] = workGroupSize;
        	vWGS[0] = workGroupSize;

        	algorithmFactory.createNewAlgorithm ()
          	.targetDeviceIs (AlgorithmTargetDevice::FPGA)
          	.targetLanguageIs (AlgorithmTargetLanguage::OpenCL)
          	.NameIs (string("WGS") + to_string(workGroupSize) +
                   string("MAPI") + to_string(memAllocationPerWorkItem) +
#if SWI_MODE==true
                   string("LL") + to_string(loopLength) +
                   string("OPS") + to_string(512) +
                   string("SWI"))
#else
            			 string("LL") + to_string(loopLength) +
            			 string("OPS") + to_string(512))
#endif
        		.KernelNameIs (string("WGS") + string("X") +
                   string("MAPI") + string("X") +
#if SWI_MODE==true
                   string("LL") + to_string(loopLength) +
                   string("OPS") + to_string(512) +
                   string("SWI"))
#else
           				 string("LL") + string("X") +
           				 string("OPS") + to_string(512))
#endif
      			 .verificationFunctionIs (verifyWGSXMAPIXLLXOPS512GAP0)
          	 .memAllocationPerWorkItemIs (memAllocationPerWorkItem)
             .workGroupSizeIs (WGS)
             .virtualWorkGroupSizeIs (vWGS)
             .memReuseFactorIs (1024)
             .startKernelFunctionSimpleV1 ()
             .createFor	(512, false, loopLength, "temp1 += temp1 * MF", 1, false, 2, 0)
             .generateForsSimpleV1 (onlyMeta)
             .popMetasSimpleV1 ()
             .endKernelFunction ()
             .verbose ()
             .writeToFile (string("/home/user/sbiookag/Algs/1For/nodep/GAP0-FloatParam-Loop/kernel") +
                     			 string("WGS") + string("X") +
                           string("MAPI") + string("X") +
#if SWI_MODE==true
                           string("LL") + to_string(loopLength) +
#else
                           string("LL") + string("X") +
#endif
	                         string("OPS") + to_string(512) +
#if SWI_MODE==true
						               string("SWI") +
#endif
                           string(".cl"));

             kernelCounter++;
      	}
    }
  }
  */
  /*

  for (int workGroupSize = 256; workGroupSize <= 256; workGroupSize *= 2) {
    for (int memAllocationPerWorkItem = 2;
         memAllocationPerWorkItem <= 2;
         memAllocationPerWorkItem *= 2) {

      //if (workGroupSize * memAllocationPerWorkItem > 1024 && localMemory)
                           //continue;

                           //if (workGroupSize * memAllocationPerWorkItem > 4096)
                           //continue;

      for (int loopLength = 65536; loopLength <= 1048576; loopLength *= 2) {
        	int *WGS = new int[1];
        	int *vWGS = new int[1];
        	WGS[0] = workGroupSize;
        	vWGS[0] = workGroupSize;

        	algorithmFactory.createNewAlgorithm ()
          	.targetDeviceIs (AlgorithmTargetDevice::FPGA)
          	.targetLanguageIs (AlgorithmTargetLanguage::OpenCL)	
          	.NameIs (string("WGS") + to_string(workGroupSize) +
                   string("MAPI") + to_string(memAllocationPerWorkItem) +
#if SWI_MODE==true
                   string("LL") + to_string(loopLength) +
                   string("OPS") + to_string(256) +
                   string("SWI"))
#else
            			 string("LL") + to_string(loopLength) +
            			 string("OPS") + to_string(256))
#endif
        		.KernelNameIs (string("WGS") + string("X") +
                   string("MAPI") + string("X") +
#if SWI_MODE==true
                   string("LL") + to_string(loopLength) +
                   string("OPS") + to_string(256) +
                   string("SWI"))
#else
           				 string("LL") + string("X") +
           				 string("OPS") + to_string(256))
#endif
          	 .memAllocationPerWorkItemIs (memAllocationPerWorkItem)
             .workGroupSizeIs (WGS)
             .virtualWorkGroupSizeIs (vWGS)
             .memReuseFactorIs (1024)
             .startKernelFunctionSimpleV1 ()
             .createFor	(256, false, loopLength, "temp1 += temp1 * MF", 1, true, 2, 0)
             .generateForsSimpleV1 (onlyMeta)
             .popMetasSimpleV1 ()
             .endKernelFunction ()
             .verbose ()
             .writeToFile (string("/home/user/sbiookag/Algs/1For/nodep/GAP0-FloatParam-Loop/kernel") +
                           string("WGS") + string("X") +
                           string("MAPI") + string("X") +
#if SWI_MODE==true
                           string("LL") + to_string(loopLength) +
#else
                           string("LL") + string("X") +
#endif
		                       string("OPS") + to_string(256) +
#if SWI_MODE==true
		                       string("SWI") +
#endif
                           string(".cl"));

             kernelCounter++;
      	}
    }
  }
  */
  /*
  for (int workGroupSize = 256; workGroupSize <= 256; workGroupSize *= 2) {
    for (int memAllocationPerWorkItem = 2;
         memAllocationPerWorkItem <= 2;
         memAllocationPerWorkItem *= 2) {

      //if (workGroupSize * memAllocationPerWorkItem > 1024 && localMemory)
      //continue;

      //if (workGroupSize * memAllocationPerWorkItem > 4096)
      //continue;

      for (int loopLength = 65536; loopLength <= 1048576; loopLength *= 2) {
        //if (loopLength == 524288) continue;
        	int *WGS = new int[1];
        	int *vWGS = new int[1];
        	WGS[0] = workGroupSize;
        	vWGS[0] = workGroupSize;

        	algorithmFactory.createNewAlgorithm ()
          	.targetDeviceIs (AlgorithmTargetDevice::FPGA)
          	.targetLanguageIs (AlgorithmTargetLanguage::OpenCL)
          	.NameIs (string("WGS") + to_string(workGroupSize) +
                   string("MAPI") + to_string(memAllocationPerWorkItem) +
#if SWI_MODE==true
                   string("LL") + to_string(loopLength) +
                   string("OPS") + to_string(128) +
                   string("SWI"))
#else
            			 string("LL") + to_string(loopLength) +
            			 string("OPS") + to_string(128))
#endif
        		.KernelNameIs (string("WGS") + string("X") +
                   string("MAPI") + string("X") +
#if SWI_MODE==true
                   string("LL") + to_string(loopLength) +
                   string("OPS") + to_string(128) +
                   string("SWI"))
#else
           				 string("LL") + string("X") +
           				 string("OPS") + to_string(128))
#endif
             .verificationFunctionIs (verifyWGSXMAPIXLLXOPS128GAP0)
          	 .memAllocationPerWorkItemIs (memAllocationPerWorkItem)
             .workGroupSizeIs (WGS)
             .virtualWorkGroupSizeIs (vWGS)
             .memReuseFactorIs (1024)
             .startKernelFunctionSimpleV1 ()
             .createFor	(128, false, loopLength, "temp1 += temp1 * MF", 1, false, 2, 0)
             .generateForsSimpleV1 (onlyMeta)
             .popMetasSimpleV1 ()
             .endKernelFunction ()
             .verbose ()
             .writeToFile (string("/home/user/sbiookag/Algs/1For/nodep/GAP0-FloatParam-Loop/kernel") +
                     	     string("WGS") + string("X") +
                           string("MAPI") + string("X") +
#if SWI_MODE==true
                           string("LL") + to_string(loopLength) +
#else
                           string("LL") + string("X") +
#endif
				                   string("OPS") + to_string(128) +
#if SWI_MODE==true
						               string("SWI") +
#endif
                           string(".cl"));

             kernelCounter++;
      	}
    }
  }
	*/
  /*
  for (int workGroupSize = 256; workGroupSize <= 256; workGroupSize *= 2) {
    for (int memAllocationPerWorkItem = 2;
         memAllocationPerWorkItem <= 2;
         memAllocationPerWorkItem *= 2) {

                     //if (workGroupSize * memAllocationPerWorkItem > 1024 && localMemory)
                     //continue;

                     //if (workGroupSize * memAllocationPerWorkItem > 4096)
                     //continue;

      for (int loopLength = 65536; loopLength <= 1048576; loopLength *= 2) {
                     //if (loopLength == 262144) break;
        	int *WGS = new int[1];
        	int *vWGS = new int[1];
        	WGS[0] = workGroupSize;
        	vWGS[0] = workGroupSize;

        	algorithmFactory.createNewAlgorithm ()
          	.targetDeviceIs (AlgorithmTargetDevice::FPGA)
          	.targetLanguageIs (AlgorithmTargetLanguage::OpenCL)
          	.NameIs (string("WGS") + to_string(workGroupSize) +
                   string("MAPI") + to_string(memAllocationPerWorkItem) +
#if SWI_MODE==true
                   string("LL") + to_string(loopLength) +
                   string("OPS") + to_string(64) +
                   string("SWI"))
#else
            			 string("LL") + to_string(loopLength) +
            			 string("OPS") + to_string(64))
#endif
        		.KernelNameIs (string("WGS") + string("X") +
                   string("MAPI") + string("X") +
#if SWI_MODE==true
                   string("LL") + to_string(loopLength) +
                   string("OPS") + to_string(64) +
                   string("SWI"))
#else
               string("LL") + string("X") +
           				 string("OPS") + to_string(64))
#endif
             .verificationFunctionIs (verifyWGSXMAPIXLLXOPS64GAP0)
          	 .memAllocationPerWorkItemIs (memAllocationPerWorkItem)
                .workGroupSizeIs (WGS)
                .virtualWorkGroupSizeIs (vWGS)
             .memReuseFactorIs (1024)
             .startKernelFunctionSimpleV1 ()
             .createFor	(64, false, loopLength, "temp1 += temp1 * MF", 1, false, 2, 0)
             .generateForsSimpleV1 (onlyMeta)
             .popMetasSimpleV1 ()
             .endKernelFunction ()
             .verbose ()
             .writeToFile (string("/home/user/sbiookag/Algs/1For/nodep/GAP0-FloatParam-Loop/kernel") +
                          string("WGS") + string("X") +
                          string("MAPI") + string("X") +
#if SWI_MODE==true
                           string("LL") + to_string(loopLength) +
#else
                           string("LL") + string("X") +
#endif
						               string("OPS") + to_string(64) +
#if SWI_MODE==true
					                 string("SWI") +
#endif
                           string(".cl"));

             kernelCounter++;
      	}
    }
  }
	*/

	int workGroupSizes[13] = {1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256}; 

	for (int ops = 4; ops <= 4; ops = ops * 2) {
    for (int workGroupSizeIndex = 0; workGroupSizeIndex < 13; workGroupSizeIndex++) {
      int workGroupSize = workGroupSizes[workGroupSizeIndex];
    	for (int memAllocationPerWorkItem = 2;
         	memAllocationPerWorkItem <= 2;
         	memAllocationPerWorkItem *= 2) {
                     //if (workGroupSize * memAllocationPerWorkItem > 1024 && localMemory)
                     //continue;

                     //if (workGroupSize * memAllocationPerWorkItem > 4096)
                     //continue;

      	for (int loopLength = 65536; loopLength <= 1048576; loopLength *= 2) {
          int realLoopLength = loopLength;
          	while (true) {
							if ((realLoopLength/32) % workGroupSize == 0) break;
              else realLoopLength++;
          	}
                     //if (loopLength == 262144) break;
        		int *WGS = new int[1];
        		int *vWGS = new int[1];
        		WGS[0] = workGroupSize;
        		vWGS[0] = workGroupSize;

        		algorithmFactory.createNewAlgorithm ()
          		.targetDeviceIs (AlgorithmTargetDevice::FPGA)
          		.targetLanguageIs (AlgorithmTargetLanguage::OpenCL)
          		.NameIs (string("WGS") + to_string(workGroupSize)
                       + string("MAPI") + to_string(memAllocationPerWorkItem)
                       + string("LL") + to_string(loopLength)
                       + string("OPS") + to_string(ops))
        			.KernelNameIs (string("WGS") + to_string(workGroupSize))
          	 	.memAllocationPerWorkItemIs (memAllocationPerWorkItem)
              .workGroupSizeIs (WGS)
              .virtualWorkGroupSizeIs (vWGS)
             	.memReuseFactorIs (1024)
             	.startKernelFunctionSimpleV2 ()
              .createFor	(ops, false, realLoopLength, "temp1 += temp1 * MF", 1, false, 2, 32)
             	.generateForsSimpleV2 (onlyMeta)
             	.popMetasSimpleV2 ()
             	.endKernelFunction ()
             	.verbose ()
            	.writeToFile (string("/home/user/sbiookag/Algs/interleaving-test2/kernelWGS")
                            + to_string(workGroupSize) + string(".cl"));

             	kernelCounter++;
      		}
    	}
  	}
	}
	

/*
	for (int workGroupSize1 = 16; workGroupSize1 <= 16; workGroupSize1 *= 2) {
    for (int workGroupSize2 = 16; workGroupSize2 <= =16; workGroupSize2 *= 2) {
    	for (int memAllocationPerWorkItem = 2;
           memAllocationPerWorkItem <= 16;
           memAllocationPerWorkItem *= 2) {

				for (int loopLength1 = 256; loopLength1 <= 1024; loopLength1 *= 2)1{
          for (int loopLength2 = 256; loopLength2 <= 1024; loopLength2 *=2 ) {
            int *WGS = new int[2];
            int *vWGS = new int[2];
						WGS[0] = workGroupSize1; WGS[1] = workGroupSize2;
            vWGS[0] = workGroupSize1; vWGS[1] = workGroupSize2;

            algorithmFactory.createNewAlgorithm ()
              .targetDeviceIs (AlgorithmTargetDevice::GPU)
              .targetLanguageIs (AlgorithmTargetLanguage::OpenCL)
              .NameIs (string("WGSF") + to_string(workGroupSize1) +
                       string("WGSS") + to_string(workGroupSize2) +
                       string("MAPI") + to_string(memAllocationPerWorkItem) +
                       string("LLF") + to_string(loopLength1) +
                       string("LLS") + to_string(loopLength2) +
                       string("OPSF") + to_string(0) +
                       string("OPSS") + to_string(1024))
              .KernelNameIs (string("WGSF") + string("X") +
                             string("WGSS") + string("X") +
                             string("MAPI") + string("X") +
                             string("LLF") + string("X") +
                             string("LLS") + string("X") +
                             string("OPSF") + to_string(0) +
                             string("OPSS") + to_string(1024))

              .memAllocationPerWorkItemIs (memAllocationPerWorkItem)
							.workGroupSizeIs (WGS)
              .virtualWorkGroupSizeIs (vWGS)
              .memReuseFactorIs (1024)
              .startKernelFunctionSimpleV1 ()
              .createFor (1024, false, loopLength2, "temp1 += temp1 * MF", 1, false, 2)
              .generateForsSimpleV1 (onlyMeta)
              .popMetasSimpleV1 ()
              .endKernelFunction ()
              .verbose ()
              .writeToFile (string("/home/users/saman/Algs/2For/nodep/GAP0/kernel")+
                            string("WGSF") + string("X") +
                            string("WGSS") + string("X") +
                            string("MAPI") + string("X") +
                            string("LLF") + string("X") +
                            string("LLS") + string("X") +
                            string("OPSF") + to_string(0) +
                            string("OPSS") + to_string(1024) +
                            string(".cl"));
            kernelCounter++;
          }
        }
      }
    }
  }
	*/
	if (executionMode == ExecutionMode::CALCULATION || executionMode == ExecutionMode::ALL)
		openCLEngine.executionCL (id, ctx, queue, resultDB, op, (char *)"float", algorithmFactory);

}

void addBenchmarkSpecOptions (OptionParser &op) {

}

void cleanup () {

}
