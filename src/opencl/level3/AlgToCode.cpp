#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "support.h"
#include "Event.h"
#include "ResultDatabase.h"
#include "OptionParser.h"
#include "ProgressBar.h"
#include <time.h>
#include <stdlib.h>

using namespace std;

// GENERATION will only generate the set of kernels in the
// kernels folder. CALCULATION will use the generated kernels
// and run them one by one.
// Use should run the code in GENERATION mode first, and then
// change the mode into CALCULATION whether on FPGA or GPU.
class ExecutionMode {
public:
  enum executionMode {GENERATION, CALCULATION, ALL};
};
int executionMode = ExecutionMode::ALL;

// Defines whether we are going to run our code on FPGA or GPU
class TargetDevice {
public:
  enum targetDevice {FPGA, GPU};
};
int targetDevice = TargetDevice::GPU;

// Path to the folder where the generated kernels will reside. Change it effectively
// based on your own system.
std::string kernels_folder = "/home/users/saman/shoc/src/opencl/level3/Algs";

// All possible flags while running the CL kernel
//static const char *opts = "-cl-mad-enable -cl-no-signed-zeros "
// 												"-cl-unsafe-math-optimizations -cl-finite-math-only";

static const char *opts = "-cl-opt-disable";

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
};

struct _cl_info {
  char name[100] = {'\0'};		 					// Name of the info
 	char kernel_location[100] = {'\0'};		// mapping between kernel types and kernel locs
  int num_workitems;										// Number of work items required by the algorithm
  int flops;
};

vector<_cl_info> cl_metas; // all meta information for all cls

// NOTICE: For current implementation we will always assume we have
// only one foor loop. This is the first implementation assumption
// and gonna be changed in the next phase implementation

struct _algorithm_type tests[] = {
  {"Test2", 2, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), 1024, 1024, 1024, 32, 128, 2, 1, "float"},
  {"Test3", 4, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), 1024, 1024, 1024, 32, 128, 2, 1, "float"},
  {"Test4", 8, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), 1024, 1024, 1024, 32, 128, 2, 1, "float"},
  {"Test5", 16, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), 1024, 1024, 1024, 32, 128, 2, 1, "float"},
  {"Test7", 2, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), 1024, 1024, 1024, 32, 128, 2, 1, "double"},
  {"Test8", 4, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), 1024, 1024, 1024, 32, 128, 2, 1, "double"},
  {"Test9", 8, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), 1024, 1024, 1024, 32, 128, 2, 1, "double"},
  {"Test10", 16, 1, vector<int>({1048576}), vector<int>({500}), false, vector<int>({0}), 1024, 1024, 1024, 32, 128, 2, 1, "double"},
  {0, 0, 0, vector<int>(), vector<int>(), 0, vector<int>(), 0, 0, 0, 0, 0, 0, 0}
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

// Generate a single algorithm code based on the given test;
void generateSingleAlgorithm (ostringstream &oss, struct _algorithm_type &test);

// Generate Single-Threaded CPU version of code based on algorithm type struct
void generateAlgorithms ();

// Generate all opencl kernel plus the meta information related to opencl
// These information consists workgroup size, number of work items, and
// any other related information. These info are all stored in clInfo
// data structure
void generateCLs ();

// Generate a single opencl code and it's metadata baseed on benchmark type struct
void generateSingleCLCode (ostringstream &oss, struct _algorithm_type &temp, struct _cl_info &info);

// Generate all CL kernels meta information for execution phase
void generateCLsMetas ();

// Generate single cl meta information for execution phase
void generateSingleCLMeta (struct _algorithm_type &temp, struct _cl_info &info);

// Executing OpenCL codes
template <class T>
void execution (cl_device_id id,
           cl_context ctx,
           cl_command_queue queue,
           ResultDatabase &resultDB,
           OptionParser &op,
           char* precision);

void insertTab (ostringstream &oss, int numTabs);

// Printing benchmark info in a human-readable format
void print_benchmark ();

// Validating correctness of given benchmark meta information
void validate_benchmark ();

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
  if (err != 0) {
    char log[5000];
    size_t retsize = 0;
    err = clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG,
                                 5000*sizeof(char), log, &retsize);
    CL_CHECK_ERROR (err);

    cout << "Build Error!" << endl;
    cout << "retSize: " << retsize << endl;
    cout << "Log: " << log << endl;
    exit (0);
  }
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

void generateCLs () {

	int aIdx = 0;
  while ((tests != 0) && (tests[aIdx].name != 0)) {
    ostringstream oss;
    struct _cl_info test_cl_info;
    struct _algorithm_type temp = tests[aIdx];
		generateSingleCLCode (oss, temp, test_cl_info);
    aIdx++;
  }

}

void generateCLsMetas () {

	int aIdx = 0;
  while ((tests != 0) && (tests[aIdx].name != 0)) {
   	struct _cl_info test_cl_info;
    struct _algorithm_type temp = tests[aIdx];
		generateSingleCLMeta (temp, test_cl_info);
    cl_metas.push_back (test_cl_info);
    aIdx++;
  }

  for (int i = 0; i < cl_metas.size(); i++) {
    cout << "Kernel location is " << cl_metas[i].kernel_location << endl; 
  }

}

void generateSingleCLCode (ostringstream &oss, struct _algorithm_type &test, struct _cl_info &info) {

  if (test.loopCarriedDataDependency == false) {
  	ofstream codeDump;
  	string dumpFileName = kernels_folder + "/" + test.name + "-" + test.varType + ".cl";
  	codeDump.open (dumpFileName.c_str());

  	if (strcmp (test.varType, "double")) {
			oss << "#pragma OPENCL EXTENSION cl_khr_fp64: enable" << endl;
    	oss << endl;
  	}

		oss << "__kernel void " << test.name << "(__global " << test.varType << " *data, __global " << test.varType << " *rands, int index, int rand_max){" << endl;
    oss << endl;
		//insertTab (oss ,1); oss << test.varType << " temp[" << test.loopsDepth[0] << "] = {0};" << endl;
    insertTab (oss, 1); oss << test.varType << test.vectorSize << " temp = 0;" << endl;
		insertTab (oss ,1); oss << "int gid = get_global_id(0);" << endl;
  	oss << endl;
		//insertTab (oss, 1); oss << "temp[0] = data[gid];" << endl;
    insertTab (oss, 1); oss << "temp = data[gid];" << endl;

  	for (int i = 1; i < test.loopsDepth[0]; i++) {
      //    	insertTab (oss, 1); oss << "temp[" << i << "] = (rands[" << i << "]/((" << test.varType << ")rand_max/2)) * temp["
      //                          << i-1 << "];" << endl;
      //      insertTab (oss, 1); oss << "temp[" << i << "] = (" << rand() << "/((" << test.varType << ")rand_max/2)) * temp["
      //        											<< i-1 << "];" << endl;

      //insertTab (oss, 1); oss << "temp[" << i << "] = (" << test.varType << ") rands[" << i << "] * temp[" << i-1 << "];" << endl;
      insertTab (oss, 1); oss << "temp = (float) rands[" << i << "] * temp;" << endl;
  	}

  	//insertTab (oss, 1); oss << "data[gid] = temp[index];" << endl;
    insertTab (oss, 1); oss << "data[gid] = temp.s0;" << endl;
  	oss << endl;
  	oss << "}" << endl;
    codeDump << oss.str();
    codeDump.close ();
  }

  if (test.loopCarriedDataDependency == true) {
    //    ofstream codeDump;
    //    string dumpFileName = kernels_folder + "/" + test.name + "-" + varType + ".cl";
    //		codeDump.open (dumpFileName.c_str());

    //    if (strcmp (varType, "double")) {
    //      oss << "#pragma OPENCL EXTENSION cl_khr_fp64: enable" << endl;
    //      oss << endl;
		//    }

    //    oss << "__kernel void " << test.name << "(__global " << varType << " *data, " << varType
    //        << " *rands, int index, int rand_max){" << endl;
    //    insertTab (oss, 1); oss << varType << " temp[" << test.loopsDepth[0] << "] = {0};" << endl;
    //		insertTab (oss, 1); oss << "int gid = get_global_id();" << endl;
    //    oss << endl;
    //    insertTab (oss, 1); oss << "temp[0] = data[0];" << endl;

    //    for (int i = 0; i < test.loopsDepth[0] - test.loopCarriedDDLengths[0]; i++) {
    //      insertTab (oss, 1); oss << "temp[" << i << "] = (rands[i]/((" << varType << ")rand_max/2)) * temp["
    //                              << i-1 << "];" << endl;
    //    }

    //    for (int i = 0; test.loopCarriedDDLengths[0]; i++) {
    //      insertTab (oss, 1); oss << "temp[" << i << "] = (rands[i]/((" << varType << ")rand_max/2)) * temp["
    //                              << i-1 << "]; "
    //                            	<<
    //    }
    //    oss <<
  }
}


void generateSingleCLMeta (_algorithm_type &test, _cl_info &info) {

  memcpy (info.name, (char *)test.name, strlen((char *)test.name));
  memcpy (info.kernel_location, (char *)(string(kernels_folder + "/" + test.name + "-" + test.varType + ".cl").c_str()),
          strlen((char *)(string(kernels_folder + "/" + test.name + "-" + test.varType + ".cl").c_str())));
  info.num_workitems = test.loopsLengths[0];
	info.flops = test.flopCount;

  cout << info.name << " " << info.kernel_location << endl;
}

// Generate Single-Thread CPU code based on benchmark type struct
void generateAlgorithms ( ) {

  int aIdx = 0;
  while ((tests != 0) && (tests[aIdx].name != 0)) {
  	ostringstream oss;
    struct _algorithm_type temp = tests[aIdx];
		generateSingleAlgorithm (oss, temp);
    aIdx++;
  }
}

void generateSingleAlgorithm (ostringstream &oss, struct _algorithm_type &test) {

  ofstream codeDump;
 	string dumpFileName = kernels_folder + "/" + test.name + "-" + test.varType + ".cpp";
	codeDump.open (dumpFileName.c_str());

  oss << "// Generated by Saman Biookaghazadeh code" << endl;
  oss << "// This is the kernel created for test case with name" << test.name << endl;
	oss << endl << endl;
	oss << "#include <iostream>" << endl;
  oss << "#include <time.h>" << endl;
  oss << "#include <stdlib.h>" << endl;
  oss << endl;
  oss << test.varType << " f(" << test.varType << " x);" << endl;
  oss << test.varType << " generate_random();" << endl;
  oss << endl;
	oss << "int main () {" << endl;
	oss << endl;
  insertTab (oss, 1); oss << "srand (time(NULL));" << endl;
	insertTab (oss, 1); oss << "int D = " << test.loopsDepth[0] <<
                       			 "; // Depth of for loop" << endl;
  insertTab (oss, 1); oss << "int L = " << test.loopsLengths[0] <<
                        		 "; // Number of times loop is going to iterate" << endl;
	insertTab (oss, 1); oss << test.varType << "** temp;" << endl;
  insertTab (oss, 1); oss << "temp = new " << test.varType << "*[D];" << endl;
	insertTab (oss, 1); oss << "for (int j = 0; j < D; j++) {" << endl;
	insertTab (oss, 2); oss << "temp[j] = new " << test.varType << "[L];" << endl;
  insertTab (oss, 1); oss << "}" << endl;
  //insertTab (oss, 1); oss << varType << " temp[" << test.loopsDepth[0] <<
  //                      								"][" << test.loopsLengths[0] << "] = {0};" << endl;
  oss << endl;
	insertTab (oss, 1); oss << "for (int i = 0; i < L; i++ ){" << endl;
  oss << endl;
	insertTab (oss, 2); oss << "temp[0][i] = generate_random();" << endl;
  for (int i = 1; i < test.loopsDepth[0]; i++ ) {
    insertTab (oss, 2); oss << "temp[" << i << "][i] = f (temp["<< i-1 << "][i]);" << endl;
  }
  oss << endl;
  insertTab (oss, 1); oss << "}";
	oss << endl;
  insertTab (oss, 1); oss << test.varType << " final = temp[0][0] + temp[" << test.loopsDepth[0] - 1 << "][" << test.loopsLengths[0] - 1 << "];" << endl;
  insertTab (oss, 1); oss << "return 0;" << endl;
	oss << endl;
  oss << "}" << endl;
  oss << endl;

	oss << test.varType << " f (" << test.varType << " x) {" << endl;
  oss << endl;
  insertTab (oss, 1); oss << "return x * ((" << test.varType << ")rand()/(" << test.varType << ")(RAND_MAX/2));" << endl;
  oss << endl;
  oss << "}" << endl;
	oss << endl;
  oss << test.varType << " generate_random() {" << endl;
  oss << endl;
  insertTab (oss, 1); oss << "return (" << test.varType << ")rand()/(" << test.varType << ")(RAND_MAX/2);" << endl;
  oss << endl;
	oss << "}" << endl;

	codeDump << oss.str();
  codeDump.close();

}

void insertTab (ostringstream &oss, int numTabs) {

	for (int i = 0; i < numTabs; i++) {
    oss << "\t";
  }

}

void print_benchmark () {
  int aIdx = 0;
  while ((tests != 0) && (tests[aIdx].name != 0)) {
    struct _algorithm_type temp = tests[aIdx];

    cout << "[TEST] ";
    cout << "{name:" << temp.name << "},";
    cout << "{numLoops:" << temp.numLoops << "},";
    cout << "{loopsLengths:";
    for (int i = 0; i < temp.numLoops; i++) {
      cout << temp.loopsLengths[i] << ",";
    }
    cout << "},";
    cout << "{loopsDepths:";
    for (int i = 0; i < temp.numLoops; i++) {
      cout << temp.loopsDepth[i] << ",";
    }
    cout << "},";
    cout << "{LoopCarried:" << temp.loopCarriedDataDependency << "},";
    cout << "{loopCarriedDepths:";
    for (int i = 0; i < temp.numLoops; i++) {
      cout << temp.loopCarriedDDLengths[i] << ",";
    }
    cout << "},";
    cout << "{varType:" << temp.varType << "}";
    cout << endl;
    aIdx++;
  }
}

void validate_benchmark () {

	int aIdx = 0;
  
  while ((tests != 0) && (tests[aIdx].name != 0)) {
    struct _algorithm_type temp = tests[aIdx];

    if (temp.numLoops != 1) {
      cout << "WARNING: Number of loops cannot exceed 1 in current version!\n";
    	exit (0);
    }

    if (temp.loopsLengths.size() != temp.numLoops) {
      cout << "ERROR: Size of loops lengths vector should be similar to number of loops!\n";
      exit (0);
    }

    if (temp.loopsDepth.size() != temp.numLoops) {
      cout << "ERROR: Size of loops depths should be similar to number of loops!\n";
      exit (0);
    }

		if (temp.loopCarriedDDLengths.size() != temp.numLoops) {
      cout << "ERROR: Size of loop carried data depndency lengths should be similar to number of loops!\n";
      exit (0);
    }
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

	int verbose = true;
	int npasses = 3;
	int err;
  char sizeStr[128];

	T *hostMem_data;
  T *hostMem2;
  T *hostMem_rands;

  cl_mem mem_data;
  cl_mem mem_rands;

	if (verbose) cout << "start execution!" << endl;

  int aIdx = 0;
  while ((tests != 0) && (tests[aIdx].name != 0)) {
		if (strcmp(tests[aIdx].varType, precision)){
      aIdx++;
      continue;
    }

    struct _algorithm_type alg = tests[aIdx];
    struct _cl_info meta = cl_metas[aIdx];

		if (verbose) cout << "[Retrieved CL Meta] ";
    if (verbose) cout << "name=" << meta.name << ", kern_loc=" << meta.kernel_location << endl;

    cl_program program;
    cl_kernel kernel;

    int halfNumFloatsMax = alg.loopsLengths[0];
    int numFloatsMax = halfNumFloatsMax;

		if (verbose) cout << "numFloatsMax=" << numFloatsMax << ", depth=" << alg.loopsDepth[0] << endl;

		hostMem_data = new T[numFloatsMax];
    hostMem2 = new T[numFloatsMax];
    hostMem_rands = new T[alg.loopsDepth[0]];

    if (verbose) cout << "hostMem_data, hostMem2, and hostMem_rands are created successfully!" << endl;

		// Filling out the hostMem_rands array
    for (int length = 0; length < alg.loopsDepth[0]; length++) {
      hostMem_rands[length] = (float)rand() / ((float)RAND_MAX/2);
    }

    program = createProgram (ctx, id, meta.kernel_location);
    if (program == NULL)
      exit (0);
    if (verbose) std::cout << "Program Created Successfully!" << endl;

    kernel = clCreateKernel (program, alg.name, &err);
    CL_CHECK_ERROR (err);
    if (verbose) cout << "Kernel Created Successfully!" << endl;

		createMemObjects<T> (ctx, queue, &mem_data, (int) sizeof (T), numFloatsMax, hostMem_data);
    CL_CHECK_ERROR (err);

    createMemObjects<T> (ctx, queue, &mem_rands, (int) sizeof (T), alg.loopsDepth[0], hostMem_rands);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *)&mem_data);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void *)&mem_rands);
    CL_CHECK_ERROR (err);

    int random = rand() % alg.loopsDepth[0];
    err = clSetKernelArg (kernel, 2, sizeof (cl_int), (void *)&random);
    CL_CHECK_ERROR (err);

    int rand_max = RAND_MAX;
    err = clSetKernelArg (kernel, 3, sizeof (cl_int), (void *)&rand_max);
    CL_CHECK_ERROR (err);

		cout << "Running Kernel " << alg.name << endl;

    //    for (int halfNumFloats = alg.halfBufSizeMin * 1024;
    //     halfNumFloats <= alg.halfBufSizeMax * 1024;
    //     halfNumFloats += alg.halfBufSizeStride * 1024) {

    // Set up input memory for data, first half = second half
    int numFloats = alg.loopsLengths[0];
    for (int j = 0; j < alg.loopsLengths[0]/2; j++) {
    	hostMem_data[j] = hostMem_data[numFloats - j - 1] = (T)(drand48()*5.0);
    }


    size_t globalWorkSize[1] = {numFloats};
    size_t maxGroupSize = 1;
    maxGroupSize = getMaxWorkGroupSize (id);

    for (int wsBegin = alg.localWorkSizeMin; wsBegin <= alg.localWorkSizeMax; wsBegin *= alg.localWorkSizeStride) {

			size_t localWorkSize[1] = {1};
      localWorkSize[0] = wsBegin;
      char lwsString[10] = {'\0'};
      for (int pas = 0; pas < npasses; ++pas) {

        refillMemObject<T> (ctx, queue, &mem_data, (int) sizeof (T), numFloats, hostMem_data);
        refillMemObject<T> (ctx, queue, &mem_rands, (int) sizeof (T), alg.loopsDepth[0], hostMem_rands);

        Event evKernel (alg.name);
        err = clEnqueueNDRangeKernel (queue, kernel, 1, NULL,
                                      globalWorkSize, localWorkSize,
                                      0, NULL, &evKernel.CLEvent());
        CL_CHECK_ERROR (err);
        err = clWaitForEvents (1, &evKernel.CLEvent());
        CL_CHECK_ERROR (err);

        evKernel.FillTimingInfo ();
        double flopCount = (double) numFloats *
            												meta.flops *
            												alg.loopsDepth[0] *
          													alg.vectorSize;

        double gflop = flopCount / (double)(evKernel.SubmitEndRuntime());
				sprintf (sizeStr, "Size: %07d", numFloats);
        sprintf (lwsString, "%d", wsBegin);
        resultDB.AddResult (string(alg.name) + string("-lws") + string (lwsString) + string ("-") + string(precision), sizeStr, "GFLOPS", gflop);

        // Zero out the host memory
        for (int j = 0; j < numFloats; j++) {
          hostMem2[j] = 0.0;
        }

        // Read the result device memory back to the host
				err = clEnqueueReadBuffer (queue, mem_data, true, 0,
                                   numFloats*sizeof(T), hostMem2,
                                   0, NULL, NULL);
        CL_CHECK_ERROR (err);

      }
    }

    err = clReleaseKernel (kernel);
    CL_CHECK_ERROR (err);
    err = clReleaseProgram (program);
    CL_CHECK_ERROR (err);
    err = clReleaseMemObject (mem_data);
    CL_CHECK_ERROR (err);
    err = clReleaseMemObject (mem_rands);
    CL_CHECK_ERROR (err);

    aIdx += 1;

    delete[] hostMem_data;
    delete[] hostMem_rands;
	}

}

void RunBenchmark (cl_device_id id,
                   cl_context ctx,
                   cl_command_queue queue,
                   ResultDatabase &resultDB,
                   OptionParser &op) {

	srand (time(NULL));

	validate_benchmark();

	if (executionMode == ExecutionMode::GENERATION) {
		cout << "---- Benchmark Representation ----" << endl;
    print_benchmark ();
    cout << "---- ----" << endl;
    generateAlgorithms ();
    generateCLs ();
  } else if (executionMode == ExecutionMode::CALCULATION) {
    generateCLsMetas();
    execution<float> (id, ctx, queue, resultDB, op, (char *)"float");
    execution<double> (id, ctx, queue, resultDB, op, (char *)"double");
  } else if (executionMode == ExecutionMode::ALL) {
    cout << "---- Benchmark Representation ----" << endl;
    print_benchmark ();
    cout << "---- ----" << endl;
    generateAlgorithms();
    generateCLs();
    generateCLsMetas();
    execution<float> (id, ctx, queue, resultDB, op, (char *)"float");
    execution<double> (id, ctx, queue, resultDB, op, (char *)"double");
  }

}

void addBenchmarkSpecOptions (OptionParser &op) {

}
