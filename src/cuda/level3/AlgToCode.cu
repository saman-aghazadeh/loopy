#include <stdio.h>
#include "cudacommon.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "CudaEngine.h"
#include "tests.h"
#include "ExecutionConfig.h"

#define VERBOSE false
#define VERIFICATION false
#define TEMP_INIT_VALUE 1.0

using namespace std;

int executionMode = ExecutionMode::CALCULATION;
int targetService = TargetDevice::GPU;

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific command line argument parsing.
//
//   -nopinned
//   This option controls whether page-locked or "pinned" memory is used.
//   The use of pinned memory typically results in higher bandwidth for data
//   transfer between host and device.
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
  op.addOption("nopinned", OPT_BOOL, "",
               "disable usage of pinned (pagelocked) memory", 'p');
}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   Measures the bandwidth of the bus connecting the host processor to the
//   OpenCL device.  This benchmark repeatedly transfers data chunks of various
//   sizes across the bus to the host from the device and calculates the
//   bandwidth for each chunk size.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB,
                  OptionParser &op)
{

	srand (time(NULL));
	CudaEngine<float> cudaEngine (ExecutionMode::CALCULATION, TargetDevice::GPU, tests);

  if (executionMode == ExecutionMode::GENERATION) {
    cout << "---- Benchmark Representation ----" << endl;
		cudaEngine.print_benchmark ();
    cout << "---- ----" << endl;
    cudaEngine.generateCUDAs ();
  } else if (executionMode == ExecutionMode::CALCULATION) {
		cudaEngine.generateCUDAMetas ();
    cudaEngine.executionCUDA (resultDB, op, (char *)"float");
  } else if (executionMode == ExecutionMode::ALL) {
    cout << "---- Benchmark Representation ----" << endl;
    cudaEngine.print_benchmark ();
    cout << "---- ----" << endl;
    cudaEngine.generateCUDAs ();
    cudaEngine.generateCUDAMetas ();
    cout << "Start Execution" << endl;
    cudaEngine.executionCUDA (resultDB, op, (char *)"float");
  }
}
