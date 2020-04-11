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
  op.addOption ("passes", OPT_INT, "3", "Number of passes for each experiment");
}

void RunBenchmark (cl_device_id dev,
                   cl_context ctx,
                   cl_command_queue queue,
                   ResultDatabase &resultDB,
                   OptionParser &op) {

  string int_precision = "-DINT_PRECISION ";
  string single_precision = "-DSINGLE_PRECISION ";
  string double_precision = "-DDOUBLE_PRECISION ";

  string kernel_name = "IntraDimensionDependency";

  long long unsigned  minDataSize = op.getOptionInt("min_data_size") * 1024 * 1024;
  long long unsigned  maxDataSize = op.getOptionInt("max_data_size") * 1024 * 1024;

  int passes = op.getOptionInt("passes");
  string intensity = op.getOptionString("intensity");
  string kernel_location = op.getOptionString("kernel_location");

  string dataType = op.getOptionString("data_type");
  string flags = "";

  int localX = 256;
  int globalX = 0;

  cl_int err;
  cl_program program;

  Event evKernel ("EventKernel");

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

  ifstream kernelFile (kernel_location, ios::in);
  ifstream headerFile ("funcs.h", ios::in);
  if (!kernelFile.is_open()) {
    cerr << "[ERROR] Failed to open file" << kernel_location << "for reading!" << endl;
    exit (0);
  }

  if (!headerFile.is_open()) {
    cerr << "[ERROR] Failed to open file funcs.h for reading!" << endl;
    exit (0);
  }

  ostringstream oss;
  oss << kernelFile.rdbuf();
    
  string srcStdStr = oss.str();
  const char* srcStr = srcStdStr.c_str();

  program = clCreateProgramWithSource (ctx, 1, (const char **)&srcStr, NULL, &err);
  CL_CHECK_ERROR (err);

  cout << "[INFO] Program Created Successfully!" << endl;


  err = clBuildProgram (program, 1, &dev, flags.c_str(), NULL, NULL);
  cout << "[INFO] Kernel compiled with flags " << flags << endl << endl;

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

  if (GENERATE_PTX) {
                size_t bin_sz;
                err = clGetProgramInfo (program, CL_PROGRAM_BINARY_SIZES,
                            sizeof (size_t), &bin_sz, NULL);

                unsigned char* bin = (unsigned char*) malloc (bin_sz);
    err = clGetProgramInfo (program, CL_PROGRAM_BINARIES,
                            sizeof(unsigned char *), &bin, NULL);

    FILE* fp = fopen ("binary.ptx", "wb");
    fwrite (bin, sizeof(char), bin_sz, fp);
    fclose(fp);
    free (bin);

  }

  CL_CHECK_ERROR (err);

  for (unsigned long long dataSize = minDataSize; dataSize <= maxDataSize; dataSize *= 2) {

    cout << "======================================================" << endl;
    cout << "======================================================" << endl << endl;
    int max = 0;

    if (dataType == "INT") {
      max = dataSize / sizeof (int);
    } else if (dataType == "SINGLE") {
      max = dataSize / sizeof (float);
    } else if (dataType == "DOUBLE") {
      max = dataSize / sizeof (double);
    }

    for (int llly = 256; llly < max/4; llly *= 2) {
      cout << "======================================================" << endl;
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

      void *input, *weight, *output;
      cl_mem clInput, clWeight, clOutput;

      cl_kernel kernel = clCreateKernel (program, kernel_name.c_str(), &err);
      CL_CHECK_ERROR (err);

      cout << "[INFO] Kernel is created successfully!" << endl;

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

      cout << "[INFO] Host side buffer created and initialized successfully!" << endl;

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

      cout << "[INFO] OpenCL buffers created and enqueued successfully!" << endl;

      int num_vecs = llly / VEC_SIZE;
      int num_stages = lllx / STAGE_SIZE;

      cout << "[INFO] num_vecs=" << num_vecs << ", num_stages=" << num_stages << endl;

      err = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *) &clInput);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void *) &clWeight);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel, 2, sizeof (cl_mem), (void *) &clOutput);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel, 3, sizeof (int), (void *) &lllx);
      CL_CHECK_ERROR (err);

      cout << "[INFO] Kernel arguments set successfully!" << endl;	

      clFinish (queue);
      CL_BAIL_ON_ERROR (err);

      size_t global_work_size[1];
      if (dataType == "INT") {
        if (llly == 0 ) {
          global_work_size[0] = (size_t)(pow(2, floor(log2l(dataSize / sizeof (int))/2)));
        } else {
          global_work_size[0] = llly;
        }
      } else if (dataType == "SINGLE") {
        if (llly == 0) {
          global_work_size[0] = (size_t)(pow(2, floor(log2l(dataSize / sizeof (float))/2)));
        } else {
          global_work_size[0] = llly;
        }
      } else if (dataType == "DOUBLE") {
        if (llly == 0) {
          global_work_size[0] = (size_t)(pow(2, floor(log2l(dataSize / sizeof (double))/2)));
        } else {
          global_work_size[0] = llly;
        }
      }

      const size_t local_work_size[] = {(size_t) localX};

      err = clEnqueueNDRangeKernel (queue, kernel, 1,
				    NULL, global_work_size, local_work_size,
				    0, NULL, &evKernel.CLEvent());
      CL_CHECK_ERROR (err);
      clFinish (queue);

      cout << "[INFO] Task is enqueued successfully!" << endl;
      cout << "[INFO] Done with warmup" << endl;

      for (int iter = 0; iter < passes; iter++) {

	cout << "[INFO] Pass #" << iter << endl;

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

        err = clEnqueueNDRangeKernel (queue, kernel, 1, 
				NULL, global_work_size, local_work_size,
				0, NULL, &evKernel.CLEvent());
	CL_CHECK_ERROR (err);

        clFinish(queue);
        CL_BAIL_ON_ERROR (err);

        evKernel.FillTimingInfo();
        if (dataType == "INT")
          resultDB.AddResult ("KernelINT" /*+ toString(dataSize) + "KiB"*/,
                              toString(dataSize)+"-"+toString(lllx)+"-"+toString(llly), "Bytes", evKernel.SubmitEndRuntime());
        else if (dataType == "SINGLE")
          resultDB.AddResult ("KernelSINGLE",
                              toString(dataSize)+"-"+toString(lllx)+"-"+toString(llly), "Bytes", evKernel.SubmitEndRuntime());
        else if (dataType == "DOUBLE")
          resultDB.AddResult ("KernelDOUBLE",
                              toString(dataSize)+"-"+toString(lllx)+"-"+toString(llly), "Bytes", evKernel.SubmitEndRuntime());

        err = clEnqueueReadBuffer (queue, clOutput, CL_TRUE, 0, dataSize, output, 0, 0, NULL);

        clFinish (queue);
        CL_BAIL_ON_ERROR (err);

      }

      clReleaseMemObject (clInput);
      clReleaseMemObject (clWeight);
      clReleaseMemObject (clOutput);
      clReleaseKernel (kernel);
      free (input);
      free (weight);
      free (output);
    }
  }

  cout << "***************************************************" << endl;

}

void cleanup () {

}
