#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cassert>
#include <cmath>

#include "OpenCLDeviceInfo.h"
#include "OptionParser.h"
#include "support.h"
#include "Event.h"
#include "ProgressBar.h"
#include "ResultDatabase.h"
#include "aocl_utils.h"
#include "constants.h"

using namespace std;

#define GENERATE_PTX true

#define CL_BAIL_ON_ERROR(err)                   \
  {                                             \
    CL_CHECK_ERROR(err);                        \
    if (err != CL_SUCCESS)                      \
      return;                                   \
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
  op.addOption ("intensity", OPT_STRING, "1", "Intensity of the operations");
  op.addOption ("kernel_location", OPT_STRING, "", "Location of the kernel file.");
}

void RunBenchmark (cl_device_id dev,
                   cl_context ctx,
                   cl_command_queue queue,
                   ResultDatabase &resultDB,
                   OptionParser &op) {

  string int_precision = "-DINT_PRECISION ";
  string single_precision = "-DSINGLE_PRECISION ";
  string double_precision = "-DDOUBLE_PRECISION ";

  string kernel_controller_name = "controller";
  string kernel_memReadData_name = "memReadData";
  string kernel_memReadWeight_name = "memReadWeight";
  string kernel_memWrite_name = "memWrite";

  long long unsigned  minDataSize = op.getOptionInt("min_data_size") * 1024 * 1024;
  long long unsigned  maxDataSize = op.getOptionInt("max_data_size") * 1024 * 1024;

  int passes = op.getOptionInt("passes");
  string intensity = op.getOptionString("intensity");
  string kernel_location = op.getOptionString("kernel_location");

  string dataType = op.getOptionString("data_type");
  string flags = "";

  cl_int err;
  cl_program program;

  Event evKernelController ("EventKernelController");
  Event evKernelMemReadData ("EventKernelMemReadData");
  Event evKernelMemReadWeight ("EventKernelMemReadWeight");
  Event evKernelMemWrite ("EventKernelMemWrite");

  cout << "[INFO] Data Type is " << dataType << endl;
  cout << "[INFO] Minimum Data Size is " << minDataSize << endl;
  cout << "[INFO] Maximum Data Size is " << maxDataSize << endl;
  // First building the program

  setenv("CUDA_CACHE_DISABLE", "1", 1);

  flags += "-cl-opt-disable ";

  if (intensity == "1") {
    flags += "-DINTENSITY1 ";
  } else if (intensity == "2") {
    flags += "-DINTENSITY2 ";
  } else if (intensity == "3") {
    flags += "-DINTENSITY3 ";
  } else if (intensity == "4") {
    flags += "-DINTENSITY4 ";
  } else if (intensity == "5") {
    flags += "-DINTENSITY5 ";
  }

  if (dataType == "INT") {
    flags += int_precision;
  } else if (dataType == "SINGLE") {
    flags += single_precision;
  } else if (dataType == "DOUBLE") {
    flags += double_precision;
  }


  string binary_file = aocl_utils::getBoardBinaryFile (kernel_location.c_str(), dev);
  program = aocl_utils::createProgramFromBinary (ctx, binary_file.c_str(), &dev, 1);
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
    int max = 0;

    if (dataType == "INT") {
      max = dataSize / sizeof (int);
    } else if (dataType == "SINGLE") {
      max = dataSize / sizeof (float);
    } else if (dataType == "DOUBLE") {
      max = dataSize / sizeof (double);
    }

    for (int llly = 256; llly < max/4; llly *= 2) {
      if (llly < VEC_SIZE) {
	cout << "[WARN] llly=" << llly << " was less than " << VEC_SIZE << endl; 
        continue;
      }
      int lllx = max / llly;
      if (lllx < STAGE_SIZE) {
	cout << "[WARN] lllx=" << lllx << " was less than " << STAGE_SIZE << endl;
	continue;
      }
      cout << "[INFO] lllx is " << lllx << " and llly is " << llly << endl;
      cout << "[INFO] data size is " << dataSize << endl;

      void *input, *weight, *output;
      cl_mem clInput, clWeight, clOutput;

      cl_kernel kernelController = clCreateKernel (program, kernel_controller_name.c_str(), &err);
      CL_CHECK_ERROR (err);

      cl_kernel kernelMemReadData = clCreateKernel (program, kernel_memReadData_name.c_str(), &err);
      CL_CHECK_ERROR (err);

      cl_kernel kernelMemReadWeight = clCreateKernel (program, kernel_memReadData_name.c_str(), &err);
      CL_CHECK_ERROR (err);

      cl_kernel kernelMemWrite = clCreateKernel (program, kernel_memWrite_name.c_str(), &err);
      CL_CHECK_ERROR (err);

      input = (void *) malloc (dataSize);
      weight = (void *) malloc (dataSize);
      output = (void *) malloc (dataSize);

      // Initialization of AA and BB arrays
      int sizeX = 0;
      int sizeY = 0;
      if (lllx == 0 || llly == 0) {
        if (dataType == "INT") {
          sizeX = (int)(pow(2, ceil(log2l(dataSize / sizeof (int))/2)));
          sizeY = (int)(pow(2, floor(log2l(dataSize / sizeof (int))/2)));
        } else if (dataType == "SINGLE") {
          sizeX = (int)(pow(2, ceil(log2l(dataSize / sizeof (float))/2)));
          sizeY = (int)(pow(2, floor(log2l(dataSize / sizeof (float))/2)));
        } else if (dataType == "DOUBLE") {
          sizeX = (int)(pow(2, ceil(log2l(dataSize / sizeof (double))/2)));
          sizeY = (int)(pow(2, floor(log2l(dataSize / sizeof (double))/2)));
        }
      } else {
        if (lllx * llly != max) {
          cout << "[ERROR] Wrong spatial and temporal dimensions!" << endl;
          return ;
        } else {
          sizeX = lllx;
          sizeY = llly;
        }

      }

      if (dataType == "INT") {
        for (int i = 0; i < sizeX; i++) {
          for (int j = 0; j < sizeY; j++) {
            ((int *)input)[i*sizeY+j] = 1;
            ((int *)weight)[i*sizeY+j] = 1;
          }
        }
      } else if (dataType == "SINGLE") {
        for (int i = 0; i < sizeX; i++) {
          for (int j = 0; j < sizeY; j++) {
            ((float *)input)[i*sizeY+j] = 1;
            ((float *)weight)[i*sizeY+j] = 1;
          }
        }
      } else if (dataType == "DOUBLE") {
        for (int i = 0; i < sizeX; i++) {
          for (int j = 0; j < sizeY; j++) {
            ((double *)input)[i*sizeY+j] = 1;
            ((double *)weight)[i*sizeY+j] = 1;
          }
        }
      }

      // Initiating the opencl buffers
      clInput = clCreateBuffer (ctx, CL_MEM_READ_WRITE, dataSize, NULL, &err);
      CL_CHECK_ERROR (err);

      clWeight = clCreateBuffer (ctx, CL_MEM_READ_ONLY, dataSize, NULL, &err);
      CL_CHECK_ERROR (err);

      clOutput = clCreateBuffer (ctx, CL_MEM_WRITE_ONLY, dataSize, NULL, &err);
      CL_CHECK_ERROR (err);

      err = clEnqueueWriteBuffer (queue, clInput, CL_TRUE, 0, dataSize, input, 0, NULL, NULL);
      CL_CHECK_ERROR (err);

      err = clEnqueueWriteBuffer (queue, clWeight, CL_TRUE, 0, dataSize, weight, 0, NULL, NULL);
      CL_CHECK_ERROR (err);

      err = clEnqueueWriteBuffer (queue, clOutput, CL_TRUE, 0, dataSize, output, 0, NULL, NULL);
      CL_CHECK_ERROR (err);

      int num_vecs = llly / VEC_SIZE;
      int num_stages = lllx / STAGE_SIZE;

      err = clSetKernelArg (kernelController, 0, sizeof (int), (void *) &num_vecs);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernelController, 1, sizeof (int), (void *) &num_stages);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernelMemReadData, 0, sizeof (cl_mem), (void *) &clInput);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernelMemReadWeight, 0, sizeof (cl_mem), (void *) &clWeight);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernelMemWrite, 0, sizeof (cl_mem), (void *) &clOutput);
      CL_CHECK_ERROR (err);


      clFinish (queue);
      CL_BAIL_ON_ERROR (err);

      err = clEnqueueTask (queue, kernelController, 0, NULL, &evKernelController.CLEvent());
      CL_CHECK_ERROR (err);

      err = clEnqueueTask (queue, kernelMemReadData, 0, NULL, &evKernelMemReadData.CLEvent());
      CL_CHECK_ERROR (err);

      err = clEnqueueTask (queue, kernelMemReadWeight, 0, NULL, &evKernelMemReadWeight.CLEvent());
      CL_CHECK_ERROR (err);

      err = clEnqueueTask (queue, kernelMemWrite, 0, NULL, &evKernelMemWrite.CLEvent());
      CL_CHECK_ERROR (err);

      clFinish (queue);

      cout << "[INFO] Done with warmup" << endl;

      for (int iter = 0; iter < passes; iter++) {

        if (dataType == "INT") {
          for (int i = 0; i < sizeX; i++) {
            for (int j = 0; j < sizeY; j++) {
              ((int *)input)[i*sizeY+j] = 1;
              ((int *)weight)[i*sizeY+j] = 1;
            }
          }
        } else if (dataType == "SINGLE") {
          for (int i = 0; i < sizeX; i++) {
            for (int j = 0; j < sizeY; j++) {
              ((float *)input)[i*sizeY+j] = 1;
              ((float *)weight)[i*sizeY+j] = 1;
            }
          }
        } else if (dataType == "DOUBLE") {
          for (int i = 0; i < sizeX; i++) {
            for (int j = 0; j < sizeY; j++) {
              ((double *)input)[i*sizeY+j] = 1;
              ((double *)weight)[i*sizeY+j] = 1;
            }
          }
        }

        err = clEnqueueWriteBuffer (queue, clInput, CL_TRUE, 0, dataSize, input, 0, NULL, NULL);
        CL_CHECK_ERROR (err);

	err = clEnqueueWriteBuffer (queue, clWeight, CL_TRUE, 0, dataSize, weight, 0, NULL, NULL);
	CL_CHECK_ERROR (err);

	err = clEnqueueWriteBuffer (queue, clOutput, CL_TRUE, 0, dataSize, output, 0, NULL, NULL);
	CL_CHECK_ERROR (err);

        err = clEnqueueTask (queue, kernelController, 0, NULL, &evKernelController.CLEvent());
	CL_CHECK_ERROR (err);

	err = clEnqueueTask (queue, kernelMemReadData, 0, NULL, &evKernelMemReadData.CLEvent());
	CL_CHECK_ERROR (err);

	err = clEnqueueTask (queue, kernelMemReadWeight, 0, NULL, &evKernelMemReadWeight.CLEvent());
	CL_CHECK_ERROR (err);

	err = clEnqueueTask (queue, kernelMemWrite, 0, NULL, &evKernelMemWrite.CLEvent());
	CL_CHECK_ERROR (err);
	

        clFinish(queue);
        CL_BAIL_ON_ERROR (err);

        evKernelMemWrite.FillTimingInfo();
        if (dataType == "INT")
          resultDB.AddResult ("KernelINT" /*+ toString(dataSize) + "KiB"*/,
                              toString(dataSize)+"-"+toString(lllx)+"-"+toString(llly), "Bytes", evKernelMemWrite.SubmitEndRuntime());
        else if (dataType == "SINGLE")
          resultDB.AddResult ("KernelSINGLE",
                              toString(dataSize)+"-"+toString(lllx)+"-"+toString(llly), "Bytes", evKernelMemWrite.SubmitEndRuntime());
        else if (dataType == "DOUBLE")
          resultDB.AddResult ("KernelDOUBLE",
                              toString(dataSize)+"-"+toString(lllx)+"-"+toString(llly), "Bytes", evKernelMemWrite.SubmitEndRuntime());

        err = clEnqueueReadBuffer (queue, clOutput, CL_TRUE, 0, dataSize, output, 0, 0, NULL);

        clFinish (queue);
        CL_BAIL_ON_ERROR (err);

      }

      clReleaseMemObject (clInput);
      clReleaseMemObject (clWeight);
      clReleaseMemObject (clOutput);
      clReleaseKernel (kernelController);
      clReleaseKernel (kernelMemReadData);
      clReleaseKernel (kernelMemReadWeight);
      clReleaseKernel (kernelMemWrite);
      free (input);
      free (weight);
      free (output);
    }
  }
}

void cleanup () {

}
