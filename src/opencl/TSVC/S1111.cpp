#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cassert>

#include "OpenCLDeviceInfo.h"
#include "OptionParser.h"
#include "support.h"
#include "Event.h"
#include "ProgressBar.h"
#include "ResultDatabase.h"
#include "aocl_utils.h"

using namespace std;

#define CL_BAIL_ON_ERROR(err) \
{															\
 		CL_CHECK_ERROR(err);			\
		if (err != CL_SUCCESS)		\
      return;									\
}

// Forward declaration
template <class T> inline std::string toString (const T& t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}

void addBenchmarkSpecOptions (OptionParser &op) {
	op.addOption ("min_data_size", OPT_INT, "0", "minimum data size (in Kilobytes)");
  op.addOption ("max_data_size", OPT_INT, "0", "maximum data size (in Kilobytes)");
  op.addOption ("data_type", OPT_STRING, "", "data type (INT or SINGLE or DOUBLE)");
  op.addOption ("kern_loc", OPT_STRING, "", "path to the kernel");
  op.addOption ("kern_name", OPT_STRING, "", "name of the kernel function");
  op.addOption ("device_type", OPT_STRING, "", "device type (GPU or FPGA)");
  op.addOption ("fpga_op_type", OPT_STRING, "", "FPGA TYPE (NDRANGE or SINGLE)");
}

void RunBenchmark (cl_device_id dev,
                   cl_context ctx,
                   cl_command_queue queue,
                   ResultDatabase &resultDB,
                   OptionParser &op) {

	string int_precision = "-DINT_PRECISION ";
  string single_precision = "-DSINGLE_PRECISION ";
  string double_precision = "-DDOUBLE_PRECISION ";

	long long unsigned  minDataSize = op.getOptionInt("min_data_size") * 1024 * 1024;
	long long unsigned  maxDataSize = op.getOptionInt("max_data_size") * 1024 * 1024;

  int passes = op.getOptionInt("passes");

	string dataType = op.getOptionString("data_type");
	string kernel_location = op.getOptionString("kern_loc");
  string kernel_name = op.getOptionString("kern_name");
	string device_type = op.getOptionString("device_type");
  string fpga_op_type = op.getOptionString("fpga_op_type");
	string flags = "";

	int localX = 256;
  int globalX = 0;

	cl_int err;
  cl_program program;

  Event evKernel ("EventKernel");
  Event evRead ("EventRead");

	cout << "[INFO] Device Type is " << device_type << endl;
  cout << "[INFO] Data Type is " << dataType << endl;
  cout << "[INFO] Kernel Location is " << kernel_location << endl;
  cout << "[INFO] Minimum Data Size is " << minDataSize << endl;
  cout << "[INFO] Maximum Data Size is " << maxDataSize << endl;
	cout << "[INFO] number of passes is " << passes << endl;
  // First building the program

	if (device_type == "GPU") {
    flags += "-DGPU ";
  } else if (device_type == "FPGA") {
    if (fpga_op_type == "NDRANGE") {
      flags += "-DFPGA_NDRANGE ";
    } else if (fpga_op_type == "SINGLE") {
      flags += "-DFPGA_SINGLE ";
    }
  }

	if (dataType == "INT") {
    flags += int_precision;
  } else if (dataType == "SINGLE") {
    flags += single_precision;
  } else if (dataType == "DOUBLE") {
    flags += double_precision;
  }

	if (device_type == "GPU") {
  	ifstream kernelFile (kernel_location, ios::in);
    if (!kernelFile.is_open()) {
      cerr << "[ERROR] Failed to open file" << kernel_location << "for reading!" << endl;
      exit (0);
    }

    ostringstream oss;
    oss << kernelFile.rdbuf();

    string srcStdStr = oss.str();
    const char* srcStr = srcStdStr.c_str();

    program = clCreateProgramWithSource (ctx, 1, (const char **)&srcStr, NULL, &err);
    CL_CHECK_ERROR (err);
  } else if (device_type == "FPGA") {
    string binary_file = aocl_utils::getBoardBinaryFile (kernel_location.c_str(), dev);
    program = aocl_utils::createProgramFromBinary (ctx, binary_file.c_str(), &dev, 1);
  }
	cout << "[INFO] Program Created Successfully!" << endl;

  err = clBuildProgram (program, 1, &dev, flags.c_str(), NULL, NULL);
  cout << "[INFO] Kernel compiled with flags " << flags << endl;

  if (err != 0) {
    char log[5000];
    size_t retSize = 0;
    err = clGetProgramBuildInfo (program, dev, CL_PROGRAM_BUILD_LOG,
                                 5000 * sizeof (char), log, &retSize);

    cout << "[ERROR] Build Error!" << endl;
    cout << "[ERROR] Ret Size: " << retSize << endl;
    cout << "[ERROR] Log: " << log << endl;
    exit (0);
  }

  CL_CHECK_ERROR (err);

  for (unsigned long long dataSize = minDataSize; dataSize <= maxDataSize; dataSize *= 2) {

		cout << "[INFO] data size is " << dataSize << endl;

    void *A, *B, *C, *D;
    cl_mem clA, clB, clC, clD;

    cl_kernel kernel = clCreateKernel (program, kernel_name.c_str(), &err);
    CL_CHECK_ERROR (err);

    A = (void *) malloc (dataSize);
    B = (void *) malloc (dataSize);
    C = (void *) malloc (dataSize);
    D = (void *) malloc (dataSize);

    clA = clCreateBuffer (ctx, CL_MEM_READ_ONLY, dataSize, NULL, &err);
    CL_CHECK_ERROR (err);

    clB = clCreateBuffer (ctx, CL_MEM_READ_ONLY, dataSize, NULL, &err);
    CL_CHECK_ERROR (err);

    clC = clCreateBuffer (ctx, CL_MEM_READ_ONLY, dataSize, NULL, &err);
    CL_CHECK_ERROR (err);

		clD = clCreateBuffer (ctx, CL_MEM_WRITE_ONLY, dataSize, NULL, &err);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clA, CL_TRUE, 0, dataSize, A, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clB, CL_TRUE, 0, dataSize, B, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clC, CL_TRUE, 0, dataSize, C, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clD, CL_TRUE, 0, dataSize, D, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *) &clA);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void *) &clB);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel, 2, sizeof (cl_mem), (void *) &clC);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel, 3, sizeof (cl_mem), (void *) &clD);
    CL_CHECK_ERROR (err);

		if (dataType == "INT") {
      int alpha = 2;
      err = clSetKernelArg (kernel, 4, sizeof (int), (void *) &alpha);
    } else if (dataType == "SINGLE") {
      float alpha = 0.5;
      err = clSetKernelArg (kernel, 4, sizeof (float), (void *) &alpha);
    } else if (dataType == "DOUBLE") {
      double alpha = 0.5;
      err = clSetKernelArg (kernel, 4, sizeof (double), (void *) &alpha);
    }

    CL_CHECK_ERROR (err);

		if (device_type == "FPGA") {
      if (fpga_op_type == "SINGLE") {
        int numIterations = 0;
        err = clSetKernelArg (kernel, 5, sizeof (int), &numIterations);
        CL_CHECK_ERROR (err);
      }
    }

		clFinish (queue);
    CL_BAIL_ON_ERROR (err);

		size_t global_work_size[1];
    if (dataType == "INT")
    	global_work_size[0] = (size_t)(dataSize / sizeof (int));
    else if (dataType == "SINGLE")
      global_work_size[0] = (size_t)(dataSize / sizeof (float));
    else if (dataType == "DOUBLE")
      global_work_size[0] = (size_t)(dataSize / sizeof (double));
    global_work_size[0] = global_work_size[0]/2;
		cout << "[INFO] global work size is " << global_work_size[0] << endl;

    const size_t local_work_size[] = {(size_t)localX};
		cout << "[INFO] local work size is " << local_work_size[0] << endl;

    err = clEnqueueNDRangeKernel (queue, kernel, 1,
                                  NULL, global_work_size, local_work_size,
                                  0, NULL, &evKernel.CLEvent());

    clFinish (queue);
    CL_BAIL_ON_ERROR (err);

		for (int iter = 0; iter < passes; iter++) {

      err = clEnqueueNDRangeKernel (queue, kernel, 1,
                                    NULL, global_work_size, local_work_size,
                                    0, NULL, &evKernel.CLEvent());

      clFinish(queue);
      CL_BAIL_ON_ERROR (err);

      evKernel.FillTimingInfo();
      if (dataType == "INT")
      	resultDB.AddResult ("KernelINT" /*+ toString(dataSize) + "KiB"*/,
                          	toString(dataSize), "Bytes", evKernel.SubmitEndRuntime());
      else if (dataType == "SINGLE")
        resultDB.AddResult ("KernelSINGLE",
                            toString(dataSize), "Bytes", evKernel.SubmitEndRuntime());
      else if (dataType == "DOUBLE")
        resultDB.AddResult ("KernelDOUBLE",
                            toString(dataSize), "Bytes", evKernel.SubmitEndRuntime());

      err = clEnqueueReadBuffer (queue, clD, CL_TRUE, 0, dataSize, D, 0, 0, &evRead.CLEvent());

      clFinish (queue);
      CL_BAIL_ON_ERROR (err);

    }

    clReleaseMemObject (clA);
    clReleaseMemObject (clB);
    clReleaseKernel (kernel);
    free (A);
    free (B);
  }
}

void cleanup () {
  
}
