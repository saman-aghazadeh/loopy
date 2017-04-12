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
ExecutionMode executionMode = ExecutionMode.GENERATION;

// Defines whether we are going to run our code on FPGA or GPU
enum TargetDevice { FPGA, GPU };

// Path to the folder where the generated kernels will reside. Change it effectively
// based on your own system.
std::string kernels_folder = "/home/users/saman/shoc/src/opencl/level3/Kernels/";

// All possible flags while running the CL kernel
static const char *opts = "";

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
  TargetDevice target;			// defines which device it intends to run on
};

struct _benchmark_type tests[] ={
  {"Add1", "s", "data[gid]", "10.f-$", 1, 240, 20, 1, 1024, 1024, 4, TargetDevice.GPU},
  {"Add2", "s", "data[gid]", "10.f-$", 2, 120, 20, 1, 1024, 1024, 4, TargetDevice.GPU},
  {"Add4", "s", "data[gid]", "10.f-$", 4, 60, 20, 1, 1024, 1024, 4, TargetDevice.GPU},
  {"Add8", "s", "data[gid]", "10.f-$", 8, 30, 20, 1, 1024, 1024, 4, TargetDevice.GPU},
  {"Add16", "s", "data[gid]", "10.f-$", 16, 20, 20, 1, 1024, 1024, 4, TargetDevice.GPU},
  {"Mul1", "s", "data[gid]-data[gid]+0.999f", "$*$*1.01f", 1, 200, 20, 2, 1024, 1024, 4, TargetDevice.GPU},
  {"Mul2", "s", "data[gid]-data[gid]+0.999f", "$*$*1.01f", 2, 100, 20, 2, 1024, 1024, 4, TargetDevice.GPU},
  {"Mul4", "s", "data[gid]-data[gid]+0.999f", "$*$*1.01f", 4, 50, 20, 2, 1024, 1024, 4, TargetDevice.GPU},
	{"Mul8", "s", "data[gid]-data[gid]+0.999f", "$*$*1.01f", 8, 25, 20, 2, 1024, 1024, 4, TargetDevice.GPU},
  {"Mul16", "s", "data[gid]-data[gid]+0.999f", "$*$*1.01f", 16, 15, 20, 2, 1024, 1024, 4, TargetDevice.GPU},
  {"MAdd1", "s", "data[gid]", "10.0f-$*0.9899f", 1, 240, 20, 2, 1024, 1024, 4, TargetDevice.GPU},
  {"MAdd2", "s", "data[gid]", "10.0f-$*0.9899f", 2, 120, 20, 2, 1024, 1024, 4, TargetDevice.GPU},
  {"MAdd4", "s", "data[gid]", "10.0f-$*0.9899f", 4, 60, 20, 2, 1024, 1024, 4, TargetDevice.GPU},
  {"MAdd8", "s", "data[gid]", "10.0f-$*0.9899f", 8, 30, 20, 2, 1024, 1024, 4, TargetDevice.GPU},
  {"MAdd16", "s", "data[gid]", "10.0f-$*0.9899f", 16, 20, 20, 2, 1024, 1024, 4, TargetDevice.GPU},
  {"MulMAdd1", "s", "data[gid]", "(3.75f-0.355f*$)*$", 1, 160, 20, 3, 1024, 1024, 4, TargetDevice.GPU},
  {"MulMAdd2", "s", "data[gid]", "(3.75f-0.355f*$)*$", 2, 80, 20, 3, 1024, 1024, 4, TargetDevice. GPU},
  {"MulMAdd4", "s", "data[gid]", "(3.75f-0.355f*$)*$", 4, 40, 20, 3, 1024, 1024, 4, TargetDevice.GPU},
  {"MulMAdd8", "s", "data[gid]", "(3.75f-0.355f*$)*$", 8, 20, 20, 3, 1024, 1024, 4, TargetDevice.GPU},
  {0, 0, 0, 0, 0, 0, 0, 0, 0}
};

// Creates the program object based on the platform,
// whether it will be GPU or FPGA.
cl_program createProgram (cl_context context,
                          cl_device_id device,
                          const char* fileName);

// Creates the memory objects which needs to be resided
// on the target device
bool createMemObjects (cl_context context, cl_command_queue queuem
                       cl_mem *memObjects,
                       const int memFloatsSize, float *data, float* nIters);

// Cleaning up the allocated resources
void cleanup (cl_program program, cl_kernel kernel, cl_mem memObject);

// Generate OpenCL kernel code based on benchmark type struct
void generateKernel (ostringstream &oss, struct _benchmark_type &test, const char* tName, const char* header);

// Generating OpenCL codes
void generation (cl_device_id id,
  							 cl_context ctx,
  							 cl_command_queue queue,
  							 ResultDatabase &resultDB,
                 OptionParser &op);

// Executing OpenCL codes
void execution  (cl_device_id id,
                 cl_context ctx,
                 cl_command_queue queue,
                 ResultDatabase &resultDB,
                 OptionParser &op);

cl_program createProgram (cl_context context,
                          cl_device_id device,
                          const char* fileName) {

	cl_int err;
  cl_program program;

  // Open kernel file and check whether exists or not
  std::ifstream kernelFile (fileName, std::ios::in);
  if (!kernelFile.is_open()) {
    std::cerr << "Failed to open file for reading!" << fileName << std::endl;
    exit (0);
  }

  std::ostringstream oss;
  oss << kernelFile.rdbuf();

  // Create and build a program
  std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
  program = clCreateProgramWithSource (context, 1, (const char **)&srcStr,
                                       NULL, &errNum);
  CL_CHECK_ERROR (err);
  err = clBuildProgram (program, 0, NULL, NULL, NULL, NULL);
  CL_CHECK_ERROR (err);

  return program;

}

bool createMemObjects (cl_context context, cl_command_queue queue,
                       cl_mem *memObjects,
                       const int memFloatsSize, float *data, float *nIters) {

	cl_int err;

  memObjects[0] = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  memFloatsSize * sizeof(float), NULL, &err);
  CL_CHECK_ERROR (err);
  memObjects[1] = clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(int), NULL, &err);
  CL_CHECK_ERROR (err);

  if (memObjects[0] == NULL || memObjects[1] == NULL) {
    std::cerr << "Error creating memory objects. " << std::endl;
    return false;
  }

  // Enqueue data buffer
  Event evWriteData ("write-data");
  err = clEnqueueWriteBuffer (queue, memObjects[0], CL_FALSE, 0, memFloatsSize * sizeof(float),
                              data, 0, NULL, &evWriteData.CLEvent());
  CL_CHECK_ERROR (err);
  err = clWaitForEvents (1, &evWriteData.CLEvent());
  CL_CHECK_ERROR (err);

 	// Enqueue nIters buffer
  Event evWriteIters ("write-iters");
  err = clEnqueueWriteBuffer (queue, memObjects[1], CL_FALSE, 0, sizeof(int),
                              nIters, 0, NULL, &evWriteIters.CLEvent());
	CL_CHECK_ERROR (err);
  err = clWaitForEvents (1, &evWriteIters.CLEvent());
  CL_CHECK_ERROR (err);

  return true;
}

void cleanup (cl_program program, cl_kernel kernel, cl_mem memObject) {

	if (memObject != NULL)
    clReleaseMemObject (memObject);

  if (kernel != NULL)
    clReleaseKernel (kernel);

  if (program != NULL)
    clReleaseProgram (program);

}

// **********************************************************************************
// Function: generateKernel
//
// Purpose:
//	 Generate an OpenCL kernel based on the content of the _benchmark_type
//	 structure.
//
// Arguments:
//	 oss: output string stream for writing the generated kernel
//	 test: structure containing the benchmark parameters
//
// Returns:	 nothing
//
// Programmer: Saman Biookaghazadeh
// Creation: April 12, 2017
//***********************************************************************************

void generateKernel (ostringstream &oss, struct _benchmark_type &test, const char* tName, const char* header) {

  std::ofstream kernelDump;
  std::string dumpFileName = kernel_dumps_folder + "/" + test.name;
  kernelDump.open (dumpFileName.c_str());

  string kName = test.name;
  oss << header << endl;
	oss << string("__kernel void ") << kName << "(__global " << tName << " *data, int nIters) {\n"
  		<< "  int gid = get_global_id(0), globalSize = get_global_size(0);\n";
  // use vector types to store the index variables when the number of streams is > 1
  // OpenCL has vectors of length 1 (scalar), 2, 4, 8, 16. Use the largest vector possible
  // keep track of how many vectors of each size we are going to use.
  int numVecs[5] = {0, 0, 0, 0, 0};
  int startIdx[5] = {0, 0, 0, 0, 0};
	int i, nStreams = test.numStreams, nVectors = 0;
  oss << "  " << tName << " " << test.indexVar << " = " << test.indexVarInit << ";\n";
	float iniVal = 0.f;
  for (i=4; i>=0; --i) {
    numVecs[i] = nStreams / (1<<i);
    nStreams -= (numVecs[i]* (1<<i));
    if (i == 4) startIdx[i] = 0;
    else startIdx[i] == startIdx[i+1] + numVecs[i+1];
    nVectors += numVecs[i];

    for (int vv=startIdx[i]; vv<startIdx[i]+numVecs[i]; ++vv) {
      oss << "  " << tName;
      if (i>0) oss << (1<<i);
      oss << " " << test.indexVar << vv << " = "
        	<< test.indexVar << " + ";
      if (i>0) oss << "(" << tName << (i<<1) << ")(";
      oss << iniVal;
      iniVal += 0.1;
      for (int ss = 1; ss < (1<<i); ++ss) {
        oss << "," << iniVal;
        iniVal += 0.1;
      }
      if (i>0) oss << ")";
      oss << ";\n";
    }
  }
  if (test.numRepeats > 1)
    oss << "  for (int j = 0; j < nIters; ++j){\n";

  // Write the body of the loop
  char buf[32];
  for (int uu = 0; uu < test.numUnrolls; ++uu) {
    for (int ss = 0; ss < nVectors; ++s) {
      string opCode = string (test.opFormula);
      int pos = -1;
      sprintf (buf, "%s%d", test.indexVar, ss);
      string lVar = string (buf);
      while ((pos=opCode.find("$")) != (-1))
        opCode.replace (pos, 1, lVar);
      oss << " " << lVar << "=" << opCode << ";";
    }
    oss << "\n";
  }

	if (test.numRepeats > 1)
    oss << "  }\n";

  // Now sum up all the vectors;
  for (i = 4; i >= 0; --i) {
    if (numVecs[i] > 1) {
      oss << "   " << test.indexVar << startIdx[i] << " = " << test.indexVar << startIdx[i];
			for (int ss = startIdx[i] + 1; ss < startIdx[i] + numVecs[i]; ++ss)
        ss << "+" << test.indexVar << ss;
      oss << ";\n";
    }
  }

	oss << "   data[gid] = ";

  // Find the size of the largest vector use;
  bool first = true;
  for (i = 4; i >= 0; --i) {
    if (numVecs[i] > 0) {
      for (int ss = 0; ss < (1<<i); ++ss) {
        if (!first) {
          oss << "+";
        } else
          first = false;
        oss << test.indexVar << startIdx[i];
        if (i > 0)
          oss << ".s" << hex << ss << dec;
      }
    }
  }

  oss << ";\n}";
  kernelDump << oss.str();
  kernelDump.close()
}

void addBenchmarkSpecOptions (OptionParser &op) {

}

void generation (cl_device_id id,
                 cl_context ctx,
                 cl_command_queue queue,
                 ResultDatabase &resultDB,
                 OptionParser &op) {

	int npasses = 2;
	float repeatF = 5.0f;

  // Check for double precision support
  int hasDoubleFp = 0;
  string doublePragma = "";

  if (checkExtension(id, "cl_khr_fp64")) {
    hasDoubleFp = 1;
    doublePragma = "#pragma OPENCL EXTENSION cl_khr_fp64: enable";
  }else if (checkExtension(id, "cl_amd_fp64")) {
    hasDoubleFp = 1;
    doublePragma = "#pragma OPENCL EXTENSION cl_amd_fp64: enable";
  }

  // Set variables for the rest of the code

  char* typeName = NULL;
  char* precision = NULL;
  char* pragmaText = NULL;
  if (hasDoubleFp) {
    typeName = "double"; precision = "-DP"; pragmaText = doublePragma;
  } else {
    typeName = "float"; precision = "-SP"; pragmaText = "";
  }

  int aIdx = 0;
	while ((tests != 0) & (tests[aIdx].name != 0)) {
    ostringstream oss;
    struct _benchmark_type temp = tests[aIdx];

		generateKernel (oss, temp, typeName, pragmaText);

    aIdx++;
  }

}

void RunBenchmark (cl_device_id id,
                   cl_context ctx,
                   cl_command_queue queue,
                   ResultDatabase &resultDB,
                   OptionParser &op)
{
	
}

