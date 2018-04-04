#ifndef OPENCLENGINE_H_
#define OPENCLENGINE_H_

#include <string>
#include <vector>
#include "ResultDatabase.h"
#include "OptionParser.h"
#include "Event.h"
#include "support.h"
#include "aocl_utils.h"
#include "../common/ExecutionConfig.h"
#include "AlgorithmType.h"
#include "AlgorithmFactory.h"

using namespace std;

struct _cl_info {
  char name[100] = {'\0'};				// Name of the info
  char kernel_location[100] = {'\0'};	// Mapping between kernel types and kernel locs
  int num_workitems;	// Number of works items required by the algorithm
  int flops;
};

template<class T>
class OpenCLEngine {
public:

  OpenCLEngine(cl_context context, cl_device_id device,
               int executionMode, int targetDevice,
               struct _algorithm_type *tests);
  ~OpenCLEngine();

  // Creates the program object based on the platform,
	// whether it will be GPU or FPGA
  cl_program createProgram (const char* fileName);

  // Creates the memory objects which needs to be resided
  // on the target device
  bool createMemObjects (cl_command_queue queue,
                         cl_mem *memObjects, int singleElementSize,
                         const long long memDoublesSize, T *data);

  // Refilling the allocated memory on the device side
  bool refillMemObject (cl_command_queue queue,
                        cl_mem *memObject, int singleElementSize,
                        const long long memSize,
                        T* data);

  bool readbackMemObject (cl_command_queue queue,
                          cl_mem *buffer, int singleElementSize,
                          const long long memSize,
                          T* data);

  // Cleaning up the allocated resources
  void cleanup (cl_program, cl_kernel kernel, cl_mem memObject);

  // Generate a single algorithm code based on the given test
  // void generatingSingleAlgorithm (ostringstream &oss, struct _algorithm_type &test);

  // Generate Single-Threaded CPU version of code based on algorithm type struct
  // void generateAlgorithms ();

  // Generate all CL kernel plus the meta information related to CL
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
  void executionCL (cl_device_id id,
                    cl_context ctx,
                    cl_command_queue queue,
                    ResultDatabase &resultDB,
                    OptionParser &op,
                    char* precision,
                    AlgorithmFactory& algorithmFactory);

  // Executing Matrix Pipeline Example
	void executeMatrixPipeline (cl_device_id id,
                              cl_context ctx,
                              cl_command_queue queue,
                              ResultDatabase &resultDB,
                              OptionParser &op,
                              char* precision,
                              AlgorithmFactory& algorithmFactory,
                              int A_height, int A_width,
                              int B_height, int B_width,
                              int C_height, int C_width,
                              int batch_size);

  // Executing Matrix Pipeline Example
	void executeMatrixPipeline2 (cl_device_id id,
                              cl_context ctx,
                              cl_command_queue queue,
                              ResultDatabase &resultDB,
                              OptionParser &op,
                              char* precision,
                              AlgorithmFactory& algorithmFactory,
                              int A_height, int A_width,
                              int B_height, int B_width,
                              int C_height, int C_width,
                              int batch_size);

  // Executing Matrix Pipeline Example V3
	void executeMatrixPipelineV3 (cl_device_id id,
                                cl_context ctx,
                                cl_command_queue queue,
                                ResultDatabase &resultDB,
                                OptionParser &op,
                                char* precision,
                                AlgorithmFactory& algorithmFactory,
                                int A_height, int A_width,
                                int B_height, int B_width,
                                int C_height, int C_width,
                                int batch_size);

  // Validating correctness of given benchmark meta information
  void validate_benchmark ();

  // Printing benchmark info in a human-readable format
  void print_benchmark ();

private:

  void insertTab (ostringstream &oss, int numTabs);

  // replace $ inside varDeclFormula with depth
  string preparedVarDeclFormula (char *varDeclFormula, int depth);

  // replace $ inside varDeclFormula with depth
  string preparedVarDeclFormulaNonArray (char *varDeclFormula, int depth, bool lcdd);

  // Write the initialization section for the local memory utilization
  string preparedLocalMemoryInitialization (int depth, string varType, char** formula);

  // replace $, @, # inside the formula with appropriate variables
  string prepareOriginalFormula (char *formula, int index, char *variable);

  string prepareReturnOpCode (int streamSize, string returnOpCode);


  cl_context context;
  cl_device_id device;
  // All possible flags while running the CL kernels
  // static const char* opts = "-cl-mad-enable -cl-no-signed-zeros"
  //													"-cl-unsafe-math-optimization -cl-finite-math-only";
  //const char *opts = "-cl-opt-disable";
	const char *opts = "-cl-opt-disable";

  // Path to folder where the generated kernels will reside. Change it effectively
  std::string gpu_built_kernels_folder = "/home/users/saman/shoc/src/opencl/level3/Algs";
  std::string fpga_built_kernels_folder = "/home/Design/SHOC-Kernels/bin";

  vector<_cl_info> cl_metas;

  int executionMode;
  int targetDevice;
  struct _algorithm_type *tests;
  double offset = 0;
};

#endif // OPENCLENGINE_H_
