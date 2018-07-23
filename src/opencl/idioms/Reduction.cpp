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

void prepareIndexes (void* array, int size);

void addBenchmarkSpecOptions (OptionParser &op) {
	op.addOption ("min_data_size", OPT_INT, "0", "minimum data size (in Kilobytes)");
  op.addOption ("max_data_size", OPT_INT, "0", "maximum data size (in Kilobytes)");
  op.addOption ("data_type", OPT_STRING, "", "data type (INT or SINGLE or DOUBLE)");
  op.addOption ("kern_loc", OPT_STRING, "", "path to the kernel");
  op.addOption ("device_type", OPT_STRING, "", "device type (GPU or FPGA)");
  op.addOption ("fpga_op_type", OPT_STRING, "", "FPGA TYPE (NDRANGE or SINGLE)");
  op.addOption ("pack", OPT_INT, "1", "Packing granularity");
}

void RunBenchmark (cl_device_id dev,
                   cl_context ctx,
                   cl_command_queue queue,
                   ResultDatabase &resultDB,
                   OptionParser &op) {

	string int_precision = "-DINT_PRECISION";
  string single_precision = "-DSINGLE_PRECISION";
  string double_precision = "-DDOUBLE_PRECISION";

	long long unsigned  minDataSize = op.getOptionInt("min_data_size") * 1024 * 1024;
	long long unsigned  maxDataSize = op.getOptionInt("max_data_size") * 1024 * 1024;

	int localX = 256;
  int globalX = 0;
  int passes = op.getOptionInt("passes");
	int pack = op.getOptionInt("pack");

	string dataType = op.getOptionString("data_type");
	string kernel_location = op.getOptionString("kern_loc");
	string device_type = op.getOptionString("device_type");
  string fpga_op_type = op.getOptionString("fpga_op_type");
	string flags = "";

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

  flags += " -DPACK=" + toString(pack);
  // First building the program

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

    void *A, *B;
    cl_mem clA, clB, clC;

    cl_kernel streamKernel = clCreateKernel (program, "Reduction", &err);
    CL_CHECK_ERROR (err);

    A = (void *) malloc (dataSize);
    B = (void *) malloc (dataSize);

    clA = clCreateBuffer (ctx, CL_MEM_READ_ONLY, dataSize, NULL, &err);
    CL_CHECK_ERROR (err);

    clB = clCreateBuffer (ctx, CL_MEM_WRITE_ONLY, dataSize, NULL, &err);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clA, CL_TRUE, 0, dataSize, A, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clB, CL_TRUE, 0, dataSize, B, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (streamKernel, 0, sizeof (cl_mem), (void *) &clA);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (streamKernel, 1, sizeof (cl_mem), (void *) &clB);
    CL_CHECK_ERROR (err);

    if (device_type == "FPGA") {
      if (fpga_op_type == "SINGLE") {
        int lengthX = 0;

        err = clSetKernelArg (streamKernel, 3, sizeof (int), &lengthX);
        CL_CHECK_ERROR (err);
      }
    }

    clFinish (queue);
    CL_BAIL_ON_ERROR (err);

    size_t global_work_size[1];
    if (dataType == "INT") {
      global_work_size[0] = (size_t)(dataSize / sizeof(int));
    } else if (dataType == "SINGLE") {
      global_work_size[0] = (size_t)(dataSize / sizeof(float));
    } else if (dataType == "DOUBLE") {
      global_work_size[0] = (size_t)(dataSize / sizeof(double));
    }

    global_work_size[0] = global_work_size[0] / pack;
		cout << "[INFO] global work size is "
         << global_work_size[0]
         << endl;

    const size_t local_work_size[] = {(size_t)localX};
		cout << "[INFO] local work size is "
      	 << local_work_size[0]
         << endl;

    err = clEnqueueNDRangeKernel (queue, streamKernel, 1,
                                  NULL, global_work_size, local_work_size,
                                  0, NULL, &evKernel.CLEvent());
		clFinish (queue);
    CL_BAIL_ON_ERROR (err);

    for (int iter = 0; iter < passes; iter++) {

      err = clEnqueueNDRangeKernel (queue, streamKernel, 1,
                                    NULL, global_work_size, local_work_size,
                                    0, NULL, &evKernel.CLEvent());

      clFinish(queue);
      CL_BAIL_ON_ERROR (err);

      evKernel.FillTimingInfo();
      if (dataType == "INT")
      	resultDB.AddResult ("StreamINT" /*+ toString(dataSize) + "KiB"*/,
                          	toString(dataSize), "MB", evKernel.SubmitEndRuntime());
      else if (dataType == "SINGLE")
        resultDB.AddResult ("StreamSINGLE",
                            toString(dataSize), "MB", evKernel.SubmitEndRuntime());
      else if (dataType == "DOUBLE")
        resultDB.AddResult ("StreamDOUBLE",
                            toString(dataSize), "MB", evKernel.SubmitEndRuntime());

      err = clEnqueueReadBuffer (queue, clA, CL_TRUE, 0, dataSize, A, 0, 0, &evRead.CLEvent());

      clFinish (queue);
      CL_BAIL_ON_ERROR (err);
    }

		clReleaseMemObject (clA);
    clReleaseMemObject (clB);
		clReleaseKernel (streamKernel);
		free(A);
    free(B);

  }


 
}

void cleanup () {
  
}

void prepareIndexes (void* array, int size) {

  srand(time(0));

  for (int i = 0; i < size; i++) {
    ((unsigned int*)array)[i] = i;
  }

  for (int i = 0; i < size-1; i++) {
    int j = i + rand() / (RAND_MAX / (size - i) + 1);
    int t = ((unsigned int*)array)[j];
    ((unsigned int*)array)[j] = ((unsigned int*)array)[i];
    ((unsigned int*)array)[i] = t;
  }

}
