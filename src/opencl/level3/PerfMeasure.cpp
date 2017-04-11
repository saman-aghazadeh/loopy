#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "support.h"
#include "Event.h"
#include "ResultDatabase.h"
#include "OptionParser.h"
#include "ProgressBar.h"

using namespace std;

// GENERATION will only generate the set of kernels in the
// kernels folder. CALCULATION will use the generated kernels
// and run them one by one.
// Use should run the code in GENERATION mode first, and then
// change the mode into CALCULATION whether on FPGA or GPU.
enum ExecutionMode { GENERATION, CALCULATION };

// Path to the folder where the generated kernels will reside. Change it effectively
// based on your own system.
std::string kernels_folder = "/home/users/saman/shoc/src/opencl/level3/Kernels/";



struct _benchmark_type {

	const char* name;					// Associated name with the kernel
  const char* indexVar;			// Name of the private scalar used as
  													// an accumulator
  const char* indexVarInit;	// Initialization formula for the index
  													// variable
  const char* opFormula;		// Arithmetic formula for the accumulator
  int numStreams;						// Number of paraller streams
  int numUnrolls;						// Number of times the loop was unrolled
  int numRepeats;						// Number of loop iterations (>=1)
  int flopCount;						// Number of floating point operations in one
  													// formula
  int halfBufSizeMin;				// Specify the buffer sizes for which to
  													// perform the test
  int halfBufSizeMax;				// We specify the minimum, the maximum and
  													// the geometic stride
  int halfBufSizeStride;		// geometric stride (Values are in thousands of elements)
};




