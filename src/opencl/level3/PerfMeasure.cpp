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
class ExecutionMode {
public:
  enum executionMode {GENERATION, CALCULATION};
};
int executionMode = ExecutionMode::CALCULATION;

// Defines whether we are going to run our code on FPGA or GPU
class TargetDevice {
public:
  enum targetDevice {FPGA, GPU};
};
int targetDevice = TargetDevice::GPU;

// Path to the folder where the generated kernels will reside. Change it effectively
// based on your own system.
std::string kernels_folder = "/home/users/saman/shoc/src/opencl/level3/Kernels/";

// All possible flags while running the CL kernel
//static const char *opts = "-cl-mad-enable -cl-no-signed-zeros "
// 												"-cl-unsafe-math-optimizations -cl-finite-math-only";

static const char *opts = "-cl-opt-disable";

struct _benchmark_type {

	const char* name;					// Associated name with the kernel
  const char* nameExtension;// And Extension to name for extra
  													// tests with different config but same
  													// kernel
  const char* reportExtenstion;	// And extension being used while reporting
  													// to the console
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
  int localWorkSizeMin;			// Minimum Size of the work items
	int localWorkSizeMax;			// Maximum Size of the work items
  int localWorkSizeStride;	// Geometric stride
  int target;			// defines which device it intends to run on
};

struct _benchmark_type tests[] ={
  {"Add1", "-240Unroll", "-282Iters", "s", "data[gid]", "10.f-$", 1, 240, 282, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::GPU},
  {"Add2", "-120Unroll", "-282Iters", "s", "data[gid]", "10.f-$", 2, 120, 282, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::GPU},
  {"Add2", "-240Unroll", "-282Iters", "s", "data[gid]", "10.f-$", 2, 240, 282, 1, 1024, 1024, 1024, 1, 128, 2, TargetDevice::FPGA},
  {"Add2", "-240Unroll", "-141Iters", "s", "data[gid]", "10.f-$", 2, 240, 141, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::GPU},
  {"Add4", "-60Unroll", "-282Iters", "s", "data[gid]", "10.f-$", 4, 60, 282, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::GPU},
  {"Add4", "-120Unroll", "-282Iters", "s", "data[gid]", "10.f-$", 4, 120, 282, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"Add4", "-120Unroll", "-141Iters", "s", "data[gid]", "10.f-$", 4, 120, 141, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::GPU},
  {"Add4", "-240Unroll", "-282Iters", "s", "data[gid]", "10.f-$", 4, 240, 282, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"Add4", "-240Unroll", "-70Iters", "s", "data[gid]", "10.f-$", 4, 240, 70, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::GPU},
  {"Add8", "-30Unroll", "-282Iters", "s", "data[gid]", "10.f-$", 8, 30, 282, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::GPU},
	{"Add8", "-60Unroll", "-282Iters", "s", "data[gid]", "10.f-$", 8, 60, 282, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"Add8", "-60Unroll", "-141Iters", "s", "data[gid]", "10.f-$", 8, 60, 141, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::GPU},
  {"Add8", "-120Unroll", "-282Iters", "s", "data[gid]", "10.f-$", 8, 120, 282, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"Add8", "-120Unroll", "-70Iters", "s", "data[gid]", "10.f-$", 8, 120, 70, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::GPU},
  {"Add8", "-240Unroll", "-282Iters", "s", "data[gid]", "10.f-$", 8, 240, 282, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"Add8", "-240Unroll", "-35Iters", "s", "data[gid]", "10.f-$", 8, 240, 35, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::GPU},
  {"Add16", "-20Unroll", "-282Iters", "s", "data[gid]", "10.f-$", 16, 20, 282, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::GPU},
  {"Add16", "-30Unroll", "-282Iters", "s", "data[gid]", "10.f-$", 16, 30, 282, 1, 1024, 1024, 1024, 32, 128, 4, TargetDevice::FPGA},
  {"Add16", "-30Unroll", "-141Iters", "s", "data[gid]", "10.f-$", 16, 30, 141, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::GPU},
  {"Add16", "-60Unroll", "-282Iters", "s", "data[gid]", "10.f-$", 16, 60, 282, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"Add16", "-60Unroll", "-70Iters", "s", "data[gid]", "10.f-$", 16, 60, 70, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::GPU},
  {"Add16", "-120Unroll", "-282Iters", "s", "data[gid]", "10.f-$", 16, 120, 282, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"Add16", "-120Unroll", "-35Iters", "s", "data[gid]", "10.f-$", 16, 120, 35, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::GPU},
  {"Add16", "-240Unroll", "-282Iters", "s", "data[gid]", "10.f-$", 16, 240, 282, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"Add16", "-240Unroll", "-17Iters", "s", "data[gid]", "10.f-$", 16, 240, 17, 1, 1024, 1024, 1024, 32, 128, 2, TargetDevice::GPU},
  {"Mul1", "-200Unroll", "-282Iters", "s", "data[gid]-data[gid]+0.999f", "$*$*1.01f", 1, 200, 282, 2, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"Mul2", "-100Unroll", "-282Iters", "s", "data[gid]-data[gid]+0.999f", "$*$*1.01f", 2, 100, 282, 2, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"Mul4", "-50Unroll", "-282Iters", "s", "data[gid]-data[gid]+0.999f", "$*$*1.01f", 4, 50, 282, 2, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
	{"Mul8", "-25Unroll", "-282Iters", "s", "data[gid]-data[gid]+0.999f", "$*$*1.01f", 8, 25, 282, 2, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"Mul16", "-15Unroll", "-282Iters", "s", "data[gid]-data[gid]+0.999f", "$*$*1.01f", 16, 15, 282, 2, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"MAdd1", "-240Unroll", "-282Iters", "s", "data[gid]", "10.0f-$*0.9899f", 1, 240, 282, 2, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"MAdd2", "-120Unroll", "-282Iters", "s", "data[gid]", "10.0f-$*0.9899f", 2, 120, 282, 2, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"MAdd4", "-60Unroll", "-282Iters", "s", "data[gid]", "10.0f-$*0.9899f", 4, 60, 282, 2, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"MAdd8", "-30Unroll", "-282Iters", "s", "data[gid]", "10.0f-$*0.9899f", 8, 30, 282, 2, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"MAdd16", "-20Unroll", "-282Iters", "s", "data[gid]", "10.0f-$*0.9899f", 16, 20, 282, 2, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"MulMAdd1", "-160Unroll", "-282Iters", "s", "data[gid]", "(3.75f-0.355f*$)*$", 1, 160, 282, 3, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"MulMAdd2", "-80Unroll", "-282Iters", "s", "data[gid]", "(3.75f-0.355f*$)*$", 2, 80, 282, 3, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"MulMAdd4", "-40Unroll", "-282Iters", "s", "data[gid]", "(3.75f-0.355f*$)*$", 4, 40, 282, 3, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {"MulMAdd8", "-20Unroll", "-282Iters", "s", "data[gid]", "(3.75f-0.355f*$)*$", 8, 20, 282, 3, 1024, 1024, 1024, 32, 128, 2, TargetDevice::FPGA},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};

// Creates the program object based on the platform,
// whether it will be GPU or FPGA.
cl_program createProgram (cl_context context,
                          cl_device_id device,
                          const char* fileName);

// Creates the memory objects which needs to be resided
// on the target device
bool createMemObjects (cl_context context, cl_command_queue queue,
                       cl_mem *memObjects,
                       const int memDoublesSize, double *data);

// Refilling the allocated memory on the device side
bool refillMemObject (cl_context context, cl_command_queue queue,
                      cl_mem *memObject, const int memDoublesSize, double* data);

// Cleaning up the allocated resources
void cleanup (cl_program program, cl_kernel kernel, cl_mem memObject);

// Generate OpenCL kernel code based on benchmark type struct
void generateKernel (ostringstream &oss, struct _benchmark_type &test, const char* tName, const char* header, char* precision);

// Generating OpenCL codes
void generation (cl_device_id id,
  							 cl_context ctx,
  							 cl_command_queue queue,
  							 ResultDatabase &resultDB,
                 OptionParser &op);

// Executing OpenCL codes
template <class T>
void execution  (cl_device_id id,
                 cl_context ctx,
                 cl_command_queue queue,
                 ResultDatabase &resultDB,
                 OptionParser &op,
                 char* precision);

// Print some useful info onto the console
void printInfo (cl_device_id id);

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
                                       NULL, &err);
  CL_CHECK_ERROR (err);
  err = clBuildProgram (program, 0, NULL, opts, NULL, NULL);
  CL_CHECK_ERROR (err);

  return program;

}
template <class T>
bool createMemObjects (cl_context context, cl_command_queue queue,
                       cl_mem *memObjects, int singleElementSize,
                       const int memSize, T *data) {

	cl_int err;

  *memObjects = clCreateBuffer (context, CL_MEM_READ_WRITE,
                                  memSize * singleElementSize, NULL, &err);
  CL_CHECK_ERROR (err);

  if (*memObjects == NULL) {
    std::cerr << "Error creating memory objects. " << std::endl;
    return false;
  }

  // Enqueue data buffer
  Event evWriteData ("write-data");
  err = clEnqueueWriteBuffer (queue, *memObjects, CL_FALSE, 0, memSize * singleElementSize,
                              data, 0, NULL, &evWriteData.CLEvent());
  CL_CHECK_ERROR (err);
  err = clWaitForEvents (1, &evWriteData.CLEvent());
  CL_CHECK_ERROR (err);

  return true;
}

template <class T>
bool refillMemObject (cl_context context, cl_command_queue queue,
                      cl_mem* memObject, int singleElementSize,
                      const int memSize, T *data) {

  cl_int err;

  Event evWriteData ("write-data");
  err = clEnqueueWriteBuffer (queue, *memObject, true, 0,
                              memSize * singleElementSize, data,
                              0, NULL, &evWriteData.CLEvent());
  CL_CHECK_ERROR (err);
  err = clWaitForEvents (1, &evWriteData.CLEvent());
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

void generateKernel (ostringstream &oss,
                     struct _benchmark_type &test,
                     const char* tName,
                     const char* header,
                     char* precision) {

  std::ofstream kernelDump;
  std::string dumpFileName = kernels_folder + "/" + test.name + test.nameExtension + precision + ".cl";
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
      if (i>0) oss << "(" << tName << (1<<i) << ")(";
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
    for (int ss = 0; ss < nVectors; ++ss) {
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
        oss << "+" << test.indexVar << ss;
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
  kernelDump.close();
}

void addBenchmarkSpecOptions (OptionParser &op) {

}

template<class T>
void generation (cl_device_id id,
                 cl_context ctx,
                 cl_command_queue queue,
                 ResultDatabase &resultDB,
                 OptionParser &op,
                 char* type) {

	int npasses = 2;
	float repeatF = 5.0f;

  // Check for double precision support
  int hasDoubleFp = 0;
  string doublePragma = "";

	if (!strcmp (type, "double")) {
    if (checkExtension(id, "cl_khr_fp64")) {
      hasDoubleFp = 1;
      doublePragma = "#pragma OPENCL EXTENSION cl_khr_fp64: enable";
    }else if (checkExtension(id, "cl_amd_fp64")) {
      hasDoubleFp = 1;
      doublePragma = "#pragma OPENCL EXTENSION cl_amd_fp64: enable";
    }
  }


  // Set variables for the rest of the code

  char* typeName = NULL;
  char* precision = NULL;
  char* pragmaText = NULL;
  if (!strcmp(type, "double")) {
    typeName = (char *) "double"; precision = (char *) "-DP"; pragmaText = (char *) doublePragma.c_str();
  } else if (!strcmp(type, "float")){
    typeName = (char *) "float"; precision = (char *) "-SP"; pragmaText = (char *) "";
  }

  int aIdx = 0;
	while ((tests != 0) & (tests[aIdx].name != 0)) {
    ostringstream oss;
    struct _benchmark_type temp = tests[aIdx];
		if (temp.target == targetDevice)
			generateKernel (oss, temp, typeName, pragmaText, precision);

    aIdx++;
  }

}

template <class T>
void execution (cl_device_id id,
                cl_context ctx,
                cl_command_queue queue,
                ResultDatabase &resultDB,
                OptionParser &op,
                char* precision) {

  int verbose = false;
	int npasses = 3;
	float repeatF = 5.0f;

  // Check for double precision support
  int hasDoubleFp = 0;
  string doublePragma = "";

	int err;
  cl_mem mem1;
  char sizeStr[128];
  T *hostMem, *hostMem2;


	int aIdx = 0;
  while ((tests != 0) && (tests[aIdx].name != 0)) {
    struct _benchmark_type temp = tests[aIdx];

		if (temp.target != targetDevice) {
      aIdx += 1;
      continue;
    }

		cl_program program;
    cl_kernel kernel;

    int halfNumFloatsMax = temp.halfBufSizeMax*1024;
    int numFloatsMax = 2*halfNumFloatsMax;

    hostMem = new T[numFloatsMax];
    hostMem2 = new T[numFloatsMax];

		program = createProgram (ctx, id, (kernels_folder + temp.name + temp.nameExtension + precision + ".cl").c_str());
    if (program == NULL)
      exit (0);
		if (verbose) std::cout << "Program Created Successfully!" << std::endl;

		kernel = clCreateKernel (program, temp.name, &err);
    CL_CHECK_ERROR (err);
		if (verbose) std::cout << "Kernel created successfully!" << std::endl;

    // Just check for some simple attributes. Has nothing to do with the logic
    // of the code right now.
		printInfo (id);

    // Allocate device memory
		createMemObjects<T> (ctx, queue, &mem1, (int) sizeof (T), numFloatsMax, hostMem);
    CL_CHECK_ERROR (err);

		err= clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *)&mem1);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel, 1, sizeof (cl_int), (void *)&temp.numRepeats);
    CL_CHECK_ERROR (err);

	  std::cout << "Running kernel " << temp.name << std::endl;

		for (int halfNumFloats = temp.halfBufSizeMin * 1024;
         halfNumFloats <= temp.halfBufSizeMax * 1024;
         halfNumFloats += temp.halfBufSizeStride * 1024) {

      // Set up input memory, first half = second half
			int numFloats = 2 * halfNumFloats;
			for (int j = 0; j < halfNumFloats; ++j) {
				hostMem[j] = hostMem[numFloats - j-1] = (T)(drand48()*5.0);
      }

      size_t globalWorkSize[1] = {numFloats};
      size_t maxGroupSize = 1;
      maxGroupSize = getMaxWorkGroupSize (id);

      for (int wsBegin = temp.localWorkSizeMin; wsBegin <= temp.localWorkSizeMax; wsBegin *= temp.localWorkSizeStride) {
        size_t localWorkSize[1] = {1};
        localWorkSize[0] = wsBegin;
        char lwsString[10] = {'\0'};
				for (int pas = 0; pas < npasses; ++pas) {
       		refillMemObject<T> (ctx, queue, &mem1, (int) sizeof(T), numFloats, hostMem);

        	Event evKernel (temp.name);
        	err = clEnqueueNDRangeKernel (queue, kernel, 1, NULL,
        	                              globalWorkSize, localWorkSize,
        	                              0, NULL, &evKernel.CLEvent());
        	CL_CHECK_ERROR (err);
        	err = clWaitForEvents (1, &evKernel.CLEvent());
        	CL_CHECK_ERROR (err);

	        evKernel.FillTimingInfo ();
	        double flopCount = (double) numFloats *
	          													temp.flopCount *
	          													temp.numRepeats *
	          													temp.numUnrolls *
	          													temp.numStreams;
	        double gflop = flopCount / (double)(evKernel.SubmitEndRuntime());


					sprintf (sizeStr, "Size: %07d", numFloats);
          sprintf (lwsString, "%d", wsBegin);
	        resultDB.AddResult (string(temp.name) + string(temp.nameExtension) + string(temp.reportExtenstion) + string("-lws") + string(lwsString) + precision, sizeStr, "GFLOPS", gflop);

	        // Zero out the host memory
	        for (int j = 0; j < numFloats; j++) {
	          hostMem2[j] = 0.0;
	        }

	        // Read the result device memory back to the host
	        err = clEnqueueReadBuffer (queue, mem1, true, 0,
	                                   numFloats*sizeof(T), hostMem2,
	                                   0, NULL, NULL);
	        CL_CHECK_ERROR (err);

	        // Check the result -- At a minimum the first half of memory
	        // should match the second half
					for (int j = 0; j < halfNumFloats; ++j) {
	          if (hostMem2[j] != hostMem2[numFloats-j-1]) {
	            std::cout << "Error: hostMem2[" << j << "]=" << hostMem2[j]
	              				<< " is different from it's twin element hostMem2["
	              				<< (numFloats-j-1) << "]=" << hostMem2[numFloats-j-1]
	                      << "; stopping check\n";
	            break;
	          }
	        }
	      }
      }
    }

		err = clReleaseKernel (kernel);
    CL_CHECK_ERROR (err);
    err = clReleaseProgram (program);
    CL_CHECK_ERROR (err);
    err = clReleaseMemObject (mem1);
    CL_CHECK_ERROR (err);

    aIdx += 1;

    delete[] hostMem;
    delete[] hostMem2;
  }

}

void printInfo (cl_device_id id) {

	int verbose = false;

  cl_long maxAllocSizeBytes = 0;
  cl_long maxComputeUnits = 0;
  cl_long maxWorkGroupSize = 0;
  clGetDeviceInfo (id, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                   sizeof (cl_long), &maxAllocSizeBytes, NULL);
  clGetDeviceInfo (id, CL_DEVICE_MAX_COMPUTE_UNITS,
                   sizeof (cl_long), &maxComputeUnits, NULL);
  clGetDeviceInfo (id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                   sizeof (cl_long), &maxWorkGroupSize, NULL);

  if (verbose) std::cout << "Max allocation size is " << maxAllocSizeBytes << std::endl;
  if (verbose) std::cout << "Max compute unit is " << maxComputeUnits << std::endl;
  if (verbose) std::cout << "Max Work Group size is " << maxWorkGroupSize << std::endl;
}

void RunBenchmark (cl_device_id id,
                   cl_context ctx,
                   cl_command_queue queue,
                   ResultDatabase &resultDB,
                   OptionParser &op)
{
  if (executionMode == ExecutionMode::GENERATION) {
    generation<double> (id, ctx, queue, resultDB, op, (char *) "double");
    generation<float> (id, ctx, queue, resultDB, op, (char *) "float");
  } else {
    execution<double> (id, ctx, queue, resultDB, op, (char *) "-DP");
    execution<float> (id, ctx, queue, resultDB, op, (char *) "-SP");
  }
}

