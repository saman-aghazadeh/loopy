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

std::string kernels_folder = "/home/users/saman/shoc/src/opencl/level3/FlopsFolder/";
std::string kernel_file = "flops.cl";

static const char *opts = "-cl-mad-enable -cl-no-signed-zeros "
  "-cl-unsafe-math-optimizations -cl-finite-math-only";

cl_program createProgram (cl_context context,
                          cl_device_id device,
                          const char* fileName) {
  cl_int errNum;
  cl_program program;

  std::ifstream kernelFile (fileName, std::ios::in);
  if (!kernelFile.is_open()) {
    std::cerr << "Failed to open file for reading: " << fileName << std::endl;
  }

  std::ostringstream oss;
  oss << kernelFile.rdbuf();

  std::string srcStdStr = oss.str();
  const char *srcStr = srcStdStr.c_str();
  program = clCreateProgramWithSource (context, 1, (const char **)&srcStr,
                                       NULL, &errNum);

	CL_CHECK_ERROR(errNum);

  errNum = clBuildProgram (program, 0, NULL, NULL, NULL, NULL);
  CL_CHECK_ERROR (errNum);

  return program;
}

bool createMemObjects (cl_context context, cl_command_queue queue,
                       cl_mem* memObject,
                       const int memFloatsSize, float *a) {

	cl_int err;
  *memObject = clCreateBuffer (context, CL_MEM_READ_WRITE,
                              memFloatsSize * sizeof(float), NULL, &err);
	CL_CHECK_ERROR(err);

  if (*memObject == NULL) {
    std::cerr << "Error creating memory objects. " << std::endl;
    return false;
  }

  Event evWrite("write");
	err = clEnqueueWriteBuffer (queue, *memObject, CL_FALSE, 0, memFloatsSize * sizeof(float),
                        a, 0, NULL, &evWrite.CLEvent());
  CL_CHECK_ERROR(err);
	err = clWaitForEvents (1, &evWrite.CLEvent());
  CL_CHECK_ERROR(err);

  return true;

}

void cleanup (cl_context context, cl_command_queue commandQueue,
              cl_program program, cl_kernel kernel, cl_mem memObject) {

  if (memObject != NULL)
		clReleaseMemObject (memObject);

  if (kernel != NULL)
    clReleaseKernel (kernel);

  if (program != NULL)
    clReleaseProgram (program);

}

void addBenchmarkSpecOptions(OptionParser &op) {

}

void RunBenchmark(cl_device_id id,
                  cl_context ctx,
                  cl_command_queue queue,
                  ResultDatabase &resultDB,
                  OptionParser &op)
{

  for (float i = 0.1; i <= 0.2; i+=0.1 ) {
    std::cout << "Deploying " << 100*i << "%" << std::endl;
		bool verbose = false;

		cl_int errNum;

  	cl_program program = 0;
  	cl_kernel kernel;
  	cl_mem memObject = 0;

  	char maxFloatsStr[128];
    char testStr[128];
		program = createProgram (ctx, id, (kernels_folder + kernel_file).c_str());
		if (program == NULL) {
    	exit (0);
  	}

  	if (verbose) std::cout << "Program created successfully!" << std::endl;

  	kernel = clCreateKernel (program, "flops", &errNum);
  	CL_CHECK_ERROR(errNum);

  	if (verbose) std::cout << "Kernel created successfully!" << std::endl;
  	// Identify maximum size of the global memory on the device side
		cl_long maxAllocSizeBytes = 0;
  	cl_long maxComputeUnits = 0;
  	cl_long maxWorkGroupSize = 0;
  	clGetDeviceInfo (id, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
  	                 sizeof(cl_long), &maxAllocSizeBytes, NULL);
  	clGetDeviceInfo (id, CL_DEVICE_MAX_COMPUTE_UNITS,
  	                 sizeof(cl_long), &maxComputeUnits, NULL);
  	clGetDeviceInfo (id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
  	                 sizeof(cl_long), &maxWorkGroupSize, NULL);

		// Let's use 80% of this memory for transferring data
  	cl_long maxFloatsUsageSize = ((maxAllocSizeBytes / 4) * 0.8);

  	if (verbose) std::cout << "Max floats usage size is " << maxFloatsUsageSize << std::endl;
  	if (verbose) std::cout << "Max compute unit is " << maxComputeUnits << std::endl;
  	if (verbose) std::cout << "Max Work Group size is " << maxWorkGroupSize << std::endl;

  	// Prepare buffer on the host side
  	float *a = new float[maxFloatsUsageSize];
  	for (int j = 0; j < maxFloatsUsageSize; j++) {
    	a[j] = (float) (j % 77);
  	}

  	if (verbose) std::cout << "Host buffer been prepared!" << std::endl;
  	// Creating buffer on the device side
  	if (!createMemObjects(ctx, queue, &memObject, maxFloatsUsageSize, a)) {
    	exit (0);
  	}

  	errNum = clSetKernelArg (kernel, 0, sizeof(cl_mem), &memObject);
		CL_CHECK_ERROR(errNum);

  	size_t wg_size, wg_multiple;
  	cl_ulong local_mem, private_usage, local_usage;
  	errNum = clGetKernelWorkGroupInfo (kernel, id,
    	                                 CL_KERNEL_WORK_GROUP_SIZE,
    	                                 sizeof (wg_size), &wg_size, NULL);
  	CL_CHECK_ERROR (errNum);
  	errNum = clGetKernelWorkGroupInfo (kernel, id,
    	                                 CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
    	                                 sizeof (wg_multiple), &wg_multiple, NULL);
  	CL_CHECK_ERROR (errNum);
  	errNum = clGetKernelWorkGroupInfo (kernel, id,
    	                                 CL_KERNEL_LOCAL_MEM_SIZE,
    	                                 sizeof (local_usage), &local_usage, NULL);
  	CL_CHECK_ERROR (errNum);
  	errNum = clGetKernelWorkGroupInfo (kernel, id,
    	                                 CL_KERNEL_PRIVATE_MEM_SIZE,
    	                                 sizeof (private_usage), &private_usage, NULL);
  	CL_CHECK_ERROR (errNum);
  	if (verbose) std::cout << "Work Group size is " << wg_size << std::endl;
  	if (verbose) std::cout << "Preferred Work Group size is " << wg_multiple << std::endl;
  	if (verbose) std::cout << "Local memory size is " << local_usage << std::endl;
 		if (verbose) std::cout << "Private memory size is " << private_usage << std::endl;


  	size_t globalWorkSize[1] = {maxFloatsUsageSize};
  	size_t localWorkSize[1] = {1};

  	Event evKernel("flops");
  	errNum = clEnqueueNDRangeKernel (queue, kernel, 1, NULL,
    	                               globalWorkSize, localWorkSize,
    	                               0, NULL, &evKernel.CLEvent());
 		CL_CHECK_ERROR (errNum);
  	if (verbose) cout << "Waiting for execution to finish ";
  	errNum = clWaitForEvents(1, &evKernel.CLEvent());
  	CL_CHECK_ERROR(errNum);
  	evKernel.FillTimingInfo();
  	if (verbose) cout << "Kernel execution terminated successfully!" << std::endl;
		delete[] a;

  	sprintf (maxFloatsStr, "Size: %d", maxFloatsUsageSize);
    sprintf (testStr, "Flops: %f\% Memory", 100*i);
  	double flopCount = maxFloatsUsageSize * 1;
  	double gflop = flopCount / (double)(evKernel.SubmitEndRuntime());
    cout << "SubmitEndRuntime = " << evKernel.SubmitEndRuntime();
		resultDB.AddResult (testStr, maxFloatsStr, "GFLOPS", gflop);

  	// Now it's time to read back the data
		a = new float[maxFloatsUsageSize];
		errNum = clEnqueueReadBuffer(queue, memObject, CL_TRUE, 0, maxFloatsUsageSize*sizeof(float), a, 0, NULL, NULL);
  	CL_CHECK_ERROR(errNum);
    if (verbose) {
			for (int j = 0; j < 10; j++) {
    		std::cout << a[j] << " ";
  		}
    }

    delete[] a;
    if (memObject != NULL)
      clReleaseMemObject (memObject);
    if (program != NULL)
      clReleaseProgram (program);
    if (kernel != NULL)
      clReleaseKernel (kernel);
  }
 	std::cout << "Program executed successfully!" << std::endl;

}
