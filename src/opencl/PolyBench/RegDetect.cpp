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

#define GENERATE_PTX true

#define CL_BAIL_ON_ERROR(err) \
{															\
 		CL_CHECK_ERROR (err);			\
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
  op.addOption ("intensity", OPT_STRING, "", "intensity of the kernel");
  op.addOption ("use_channel", OPT_STRING, "0", "Whether using channel or not!");
  op.addOption ("num_fmas", OPT_STRING, "1", "Number of FMA operations");
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
  string intensity = op.getOptionString("intensity");
	string numfmas = op.getOptionString("num_fmas");


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
	cout << "[INFO] Number of passes is " << passes << endl;
  cout << "[INFO] Number of fmas are " << numfmas << endl;
  // First building the program

	setenv ("CUDA_CACHE_DISABLE", "1", 1);

  flags += "-cl-opt-disable";
  flags += ("-DNUMFMAS=" + numfmas + " ");

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

  for (unsigned long long dataSize = minDataSize; dataSize <= maxDataSize; dataSize *= 2) {

		cout << "[INFO] data size is " << dataSize << endl;

		int size = 0;
		if (dataType == "INT") {
      size = (int)(dataSize / sizeof(int));
      int sizeSqrt = sqrt(size);
      if (sizeSqrt * sizeSqrt != size) continue;
    } else if (dataType == "SINGLE") {
      size = (int)(dataSize / sizeof(float));
      int sizeSqrt = sqrt(size);
      if (sizeSqrt * sizeSqrt != size) continue;
    } else if (dataType == "DOUBLE") {
      size = (int)(dataSize / sizeof(double));
      int sizeSqrt  = sqrt(size);
    }

    void *path, *mean;
    cl_mem clPath, clMean;

    cl_kernel kernel = clCreateKernel (program, kernel_name.c_str(), &err);
    CL_CHECK_ERROR (err);

    path = (void *) malloc (dataSize);
   	mean = (void *) malloc (dataSize);

    if (dataType == "INT") {
      for (int i = 0; i < size; i++) {
      	((int *)path)[i] = 1.5;
        ((int *)mean)[i] = 2.5;
      }
    } else if (dataType == "SINGLE") {
      for (int i = 0; i < size; i++) {
        ((float *)path)[i] = 1.5;
        ((float *)mean)[i] = 2.5;
      }
    } else if (dataType == "DOUBLE") {
      for (int i = 0; i < size; i++) {
        ((double *)path)[i] = 1.5;
        ((double *)mean)[i] = 2.5;
      }
    }

    clPath = clCreateBuffer (ctx, CL_MEM_READ_ONLY, dataSize, NULL, &err);
    CL_CHECK_ERROR (err);

    clMean = clCreateBuffer (ctx, CL_MEM_READ_ONLY, dataSize, NULL, &err);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clPath, CL_TRUE, 0, dataSize, path, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clMean, CL_TRUE, 0, dataSize, mean, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *) &clPath);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void *) &clMean);
    CL_CHECK_ERROR (err);

    CL_CHECK_ERROR (err);

    int numIterations = 0;
    if (dataType == "INT") {
    	numIterations = dataSize / 4;
    } else if (dataType == "SINGLE") {
      numIterations = dataSize / 4;
    } else if (dataType == "DOUBLE") {
      numIterations = dataSize / 8;
    }

		int lll = sqrt(numIterations);

    err = clSetKernelArg (kernel, 2, sizeof (int), &lll);
    CL_CHECK_ERROR (err);

		clFinish (queue);
    CL_BAIL_ON_ERROR (err);


		size_t global_work_size[1];
    if (dataType == "INT") {
      global_work_size[0] = dataSize/4;
    } else if (dataType == "SINGLE") {
      global_work_size[0] = dataSize/4;
    } else if (dataType == "DOUBLE") {
      global_work_size[0] = dataSize/8;
    }
		cout << "[INFO] global work size is " << global_work_size[0] << endl;

    const size_t local_work_size[] = {(size_t)localX};
		cout << "[INFO] local work size is " << local_work_size[0] << endl;

    if (device_type == "FPGA" && fpga_op_type == "SINGLE") {
      err = clEnqueueTask (queue, kernel, 0, NULL, &evKernel.CLEvent());
    } else {
      err = clEnqueueNDRangeKernel (queue, kernel, 1,
                                      NULL, global_work_size, local_work_size,
                                      0, NULL, &evKernel.CLEvent());
    }

    clFinish (queue);
    CL_BAIL_ON_ERROR (err);

		cout << "[INFO] Done with warmup" << endl;

		for (int iter = 0; iter < passes; iter++) {

			if (dataType == "INT") {
        for (int i = 0; i < size; i++) {
          ((int *)path)[i] = 1.5;
          ((int *)mean)[i] = 2.5;
        }
      } else if (dataType == "SINGLE") {
        for (int i = 0; i < size; i++) {
          ((float *)path)[i] = 1.5;
          ((float *)mean)[i] = 2.5;
        }
      } else if (dataType == "DOUBLE") {
        for (int i = 0; i < size; i++) {
          ((double *)path)[i] = 1.5;
          ((double *)mean)[i] = 2.5;
        }
      }

      if (device_type == "FPGA" && fpga_op_type == "SINGLE") {
        err = clEnqueueTask (queue, kernel, 0, NULL, &evKernel.CLEvent());
      } else {
       	err = clEnqueueNDRangeKernel (queue, kernel, 1,
                                      NULL, global_work_size, local_work_size,
                                      0, NULL, &evKernel.CLEvent());
      }

      clFinish(queue);
      CL_BAIL_ON_ERROR (err);

      evKernel.FillTimingInfo();
      if (dataType == "INT")
      	resultDB.AddResult ("KernelINT",
                          	toString(dataSize), "Bytes", evKernel.SubmitEndRuntime());
      else if (dataType == "SINGLE")
        resultDB.AddResult ("KernelSINGLE",
                            toString(dataSize), "Bytes", evKernel.SubmitEndRuntime());
      else if (dataType == "DOUBLE")
        resultDB.AddResult ("KernelDOUBLE",
                            toString(dataSize), "Bytes", evKernel.SubmitEndRuntime());


      clFinish (queue);
      CL_BAIL_ON_ERROR (err);

    }

    clReleaseMemObject (clPath);
    clReleaseMemObject (clMean);
    clReleaseKernel (kernel);
    free (path);
    free (mean);
  }
}

void cleanup () {
  
}
