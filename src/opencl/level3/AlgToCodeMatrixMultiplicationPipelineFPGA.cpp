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
#include <thread>
#include <chrono>
#include <future>


#include <aocl_mmd.h>


#define PRIVATE_VECTOR_SIZE 5
#define VERBOSE false
#define VERIFICATION false
#define TEMP_INIT_VALUE 1.0
// Single work item mode
#define SWI_MODE true

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

void powerMeasure (cl_device_id id, future<void> futureObj) {

	ofstream myFile;
  myFile.open ("power.out");

	typedef void* (*get_board_extension_function_address_fn_t)(const char* func_name, cl_device_id device);
  typedef void* (*aocl_mmd_card_info_fn_t)(const char*, aocl_mmd_info_t, size_t, void*, size_t*);

  void *tempPointer;
  float power;
  size_t returnedSize;

  get_board_extension_function_address_fn_t board_extension_function_address =
    (get_board_extension_function_address_fn_t) clGetExtensionFunctionAddress ("clGetBoardExtensionFunctionAddressAltera");

	if (board_extension_function_address == NULL) {
    printf ("Failed to get clGetBoardExtensionFunctionAddressAltera\n");
  }

  tempPointer = board_extension_function_address ("aocl_mmd_card_info", id);
	aocl_mmd_card_info_fn_t aocl_mmd_card_info_fn = (aocl_mmd_card_info_fn_t)tempPointer;

  if (aocl_mmd_card_info_fn == NULL) {
    printf ("Failed to get aocl_mmd_card_info_fn address");
  }

  while (futureObj.wait_for (chrono::milliseconds(1)) == future_status::timeout) {
  	aocl_mmd_card_info_fn("aclnalla_pcie0", AOCL_MMD_POWER, sizeof(float), (void*) &power, &returnedSize);
    myFile << "Power is " << power << endl;
    this_thread::sleep_for (chrono::milliseconds(500));
  	//printf ("Power = %f W\n", power);
  }

}


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

  promise<void> exitSignal;
  future<void> futureObj = exitSignal.get_future ();

	thread power (powerMeasure, id, move(futureObj));

	algorithmFactory.createNewAlgorithm ()
    .targetDeviceIs (AlgorithmTargetDevice::FPGA)
    .targetLanguageIs (AlgorithmTargetLanguage::OpenCL)
    .NameIs ("MatrixMultiplication")
    .KernelNameIs ("MatrixMultiplication")
    .verificationFunctionIs (verifyMatrixMultiplication)
    .generateForsMatrixMultiplication (onlyMeta)
		.writeToFile (string("/home/user/sbiookag/Algs/mm/bin2/MatrixMultiplication.cl"));


	if (executionMode == ExecutionMode::CALCULATION || executionMode == ExecutionMode::ALL) {
		for (int i = 1; i < 4096; i=i*2) {
			cout << "Set batch_size to " << i << endl;
      openCLEngine.executeMatrixPipeline (id, ctx, queue, resultDB, op, (char *)"float", algorithmFactory,
                                          8, 8, 8, 8, 8, 8, i);
      algorithmFactory.resetIndex ();
    }
  }

	exitSignal.set_value ();


}

void addBenchmarkSpecOptions (OptionParser &op) {

}

void cleanup () {

}
