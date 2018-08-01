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

	int localX = 16;
  int localY = 16;
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

    void *AA, *BB;
    cl_mem clAA, clBB;

    cl_kernel kernel = clCreateKernel (program, kernel_name.c_str(), &err);
    CL_CHECK_ERROR (err);

    AA = (void *) malloc (dataSize);
    BB = (void *) malloc (dataSize);

    // Initialization of AA and BB arrays
    int sizeX = 0;
    int sizeY = 0;
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

    if (dataType == "INT") {
    	for (int i = 0; i < sizeX; i++) {
      	for (int j = 0; j < sizeY; j++) {
        	((int *)AA)[i*sizeY+j] = 1;
        	((int *)BB)[i*sizeY+j] = 1;
      	}
    	}
    } else if (dataType == "SINGLE") {
      for (int i = 0; i < sizeX; i++) {
        for (int j = 0; j < sizeY; j++) {
          ((float *)AA)[i*sizeY+j] = 1;
          ((float *)BB)[i*sizeY+j] = 1;
        }
      }
    } else if (dataType == "DOUBLE") {
      for (int i = 0; i < sizeX; i++) {
        for (int j = 0; j < sizeY; j++) {
          ((double *)AA)[i*sizeY+j] = 1;
          ((double *)BB)[i*sizeY+j] = 1;
        }
      }
    }

		// Initiating the opencl buffers
    clAA = clCreateBuffer (ctx, CL_MEM_READ_WRITE, dataSize, NULL, &err);
    CL_CHECK_ERROR (err);

    clBB = clCreateBuffer (ctx, CL_MEM_READ_ONLY, dataSize, NULL, &err);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clAA, CL_TRUE, 0, dataSize, AA, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clBB, CL_TRUE, 0, dataSize, BB, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *) &clAA);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void *) &clBB);
    CL_CHECK_ERROR (err);

		if (device_type == "FPGA") {
      if (fpga_op_type == "SINGLE") {
        int numIterationsX = 0;
        int numIterationsY = 0;
        if (dataType == "INT") {
          numIterationsX = (int)(pow(2, ceil(log2l(dataSize / sizeof (int))/2)));
          numIterationsY = (int)(pow(2, floor(log2l(dataSize / sizeof (int))/2)));
        } else if (dataType == "SINGLE") {
          numIterationsX = (int)(pow(2, ceil(log2l(dataSize / sizeof (float))/2)));
          numIterationsY = (int)(pow(2, floor(log2l(dataSize / sizeof (float))/2)));
        } else if (dataType == "DOUBLE") {
          numIterationsX = (int)(pow(2, ceil(log2l(dataSize / sizeof (double))/2)));
          numIterationsY = (int)(pow(2, floor(log2l(dataSize / sizeof (double))/2)));
        }

	cout << "[INFO] dimensions are " << numIterationsX << " " << numIterationsY << endl; 

        err = clSetKernelArg (kernel, 2, sizeof (int), &numIterationsX);
        CL_CHECK_ERROR (err);
        err = clSetKernelArg (kernel, 3, sizeof (int), &numIterationsY);
        CL_CHECK_ERROR (err);
      }
    }

		clFinish (queue);
    CL_BAIL_ON_ERROR (err);

		size_t global_work_size[2];
    if (dataType == "INT") {
    	global_work_size[0] = (size_t)(pow(2, ceil(log2l(dataSize / sizeof (int))/2)));
    	global_work_size[1] = (size_t)(pow(2, floor(log2l(dataSize / sizeof (int))/2)));
    } else if (dataType == "SINGLE") {
    	global_work_size[0] = (size_t)(pow(2, ceil(log2l(dataSize / sizeof (float))/2)));
    	global_work_size[1] = (size_t)(pow(2, floor(log2l(dataSize / sizeof (float))/2)));
    } else if (dataType == "DOUBLE") {
    	global_work_size[0] = (size_t)(pow(2, ceil(log2l(dataSize / sizeof (double))/2)));
    	global_work_size[1] = (size_t)(pow(2, floor(log2l(dataSize / sizeof (double))/2)));
    }
    if (!(device_type == "FPGA" && fpga_op_type == "SINGLE"))
			cout << "[INFO] global work size is " << global_work_size[0] << " " << global_work_size[1] << endl;

    const size_t local_work_size[] = {(size_t)localX, (size_t)localY};
    if (!(device_type == "FPGA" && fpga_op_type == "SINGLE"))
			cout << "[INFO] local work size is " << local_work_size[0] << " " << local_work_size[1] << endl;

    if (device_type == "FPGA" && fpga_op_type == "SINGLE") {
      err = clEnqueueTask (queue, kernel, 0, NULL, &evKernel.CLEvent());
    } else {
    	err = clEnqueueNDRangeKernel (queue, kernel, 2,
                                  NULL, global_work_size, local_work_size,
                                  0, NULL, &evKernel.CLEvent());
    }
    clFinish (queue);
    CL_BAIL_ON_ERROR (err);

		cout << "[INFO] Done with warmup" << endl;

		for (int iter = 0; iter < passes; iter++) {

      if (dataType == "INT") {
        for (int i = 0; i < sizeX; i++) {
          for (int j = 0; j < sizeY; j++) {
            ((int *)AA)[i*sizeY+j] = 1;
            ((int *)BB)[i*sizeY+j] = 1;
          }
        }
      } else if (dataType == "SINGLE") {
        for (int i = 0; i < sizeX; i++) {
          for (int j = 0; j < sizeY; j++) {
            ((float *)AA)[i*sizeY+j] = 1;
            ((float *)BB)[i*sizeY+j] = 1;
          }
        }
      } else if (dataType == "DOUBLE") {
        for (int i = 0; i < sizeX; i++) {
          for (int j = 0; j < sizeY; j++) {
            ((double *)AA)[i*sizeY+j] = 1;
            ((double *)BB)[i*sizeY+j] = 1;
          }
        }
      }

      err = clEnqueueWriteBuffer (queue, clAA, CL_TRUE, 0, dataSize, AA, 0, NULL, NULL);
      CL_CHECK_ERROR (err);

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
      	resultDB.AddResult ("KernelINT" /*+ toString(dataSize) + "KiB"*/,
                          	toString(dataSize), "Bytes", evKernel.SubmitEndRuntime());
      else if (dataType == "SINGLE")
        resultDB.AddResult ("KernelSINGLE",
                            toString(dataSize), "Bytes", evKernel.SubmitEndRuntime());
      else if (dataType == "DOUBLE")
        resultDB.AddResult ("KernelDOUBLE",
                            toString(dataSize), "Bytes", evKernel.SubmitEndRuntime());

      err = clEnqueueReadBuffer (queue, clAA, CL_TRUE, 0, dataSize, AA, 0, 0, &evRead.CLEvent());

      clFinish (queue);
      CL_BAIL_ON_ERROR (err);

			// Testing the same operation on CPU and do the verification
      void* AACPU = (void *) (malloc (dataSize));
      void* BBCPU = (void *) (malloc (dataSize));

	    if (dataType == "INT") {
		cout << "Sizes are " << sizeX << " " << sizeY << endl;
        for (int i = 0; i < sizeX; i++) {
          for (int j = 0; j < sizeY; j++) {
            ((int *)AACPU)[i*sizeY+j] = 1;
            ((int *)BBCPU)[i*sizeY+j] = 1;
          }
        }
        for (int i = 1; i < sizeX; i++) {
          for (int j = 1; j < sizeY; j++) {
            ((int *)AACPU)[i*sizeY+j] = ((int *)AACPU)[(i-1)*sizeY+(j-1)] + ((int *)BBCPU)[i*sizeY+j];
          }
        }
        int wrong = 0;
        for (int i = 0; i < sizeX; i++) {
          for (int j = 0; j < sizeY; j++) {
            if (((int *)BBCPU)[i*sizeY+j] != ((int *)BB)[i*sizeY+j]) {
              wrong = 1;
              break;
            }
	    cout << "[INFO] AACPU[" << i*sizeY+j << "]=" 
		 << ((int *)AACPU)[i*sizeY+j] << " AA[" 
		 << i*sizeY+j << "]=" << ((int *)AA)[i*sizeY+j] << endl;
            if (((int *)AACPU)[i*sizeY+j] != ((int *)AA)[i*sizeY+j]) {
              wrong = 2;
              break;
            }
          }
          if (wrong) break;
        }
        if (wrong == 1) cout << "[ERROR] BB does not match!" << endl;
        else if (wrong == 2) cout << "[ERROR] AA does not match!" << endl;
        else cout << "[INFO] Data matches perfectly!" << endl;
      } else if (dataType == "SINGLE") {
        for (int i = 0; i < sizeX; i++) {
          for (int j = 0; j < sizeY; j++) {
            ((float *)AACPU)[i*sizeY+j] = 1;
            ((float *)BBCPU)[i*sizeY+j] = 1;
          }
        }
        for (int i = 1; i < sizeX; i++) {
          for (int j = 1; j < sizeY; j++) {
				  	((float *)AACPU)[i*sizeY+j] = ((float *)AACPU)[(i-1)*sizeY+(j-1)] + ((float *)BBCPU)[i*sizeY+j];
          }
        }
				int wrong = 0;
        for (int i = 0; i < sizeX; i++) {
          for (int j = 0; j < sizeY; j++) {
            if (abs(((float *)BBCPU)[i*sizeY+j] - ((float *)BB)[i*sizeY+j]) > 0.01) {
              wrong = 1;
              break;
            }
            if (abs(((float *)AACPU)[i*sizeY+j] - ((float *)AA)[i*sizeY+j]) > 0.01) {
              wrong = 2;
              break;
            }

          }
          if (wrong) break;
        }

        if (wrong == 1) cout << "[ERROR] BB does not match!" << endl;
        else if (wrong == 2) cout << "[ERROR] AA does not match!" << endl;
        else cout << "[INFO] Data matches perfectly!" << endl;

      } else if (dataType == "DOUBLE") {
        for (int i = 0; i < sizeX; i++) {
          for (int j = 0; j < sizeY; j++) {
            ((double *)AACPU)[i*sizeY+j] = 1;
            ((double *)BBCPU)[i*sizeY+j] = 1;
          }
        }
        for (int i = 1; i < sizeX; i++) {
          for (int j = 1; j < sizeY; j++) {
            ((double *)AACPU)[i*sizeY+j] = ((double *)AACPU)[(i-1)*sizeY+(j-1)] + ((double *)BBCPU)[i*sizeY+j];
          }
        }
        int wrong = 0;
        for (int i = 0; i < sizeX; i++) {
          for (int j = 0; j < sizeY; j++) {
            if (abs(((double *)BBCPU)[i*sizeY+j] - ((double *)BB)[i*sizeY+j]) > 0.01) {
              wrong = 1;
              break;
            }
            if (abs(((double *)AACPU)[i*sizeY+j] - ((double *)AA)[i*sizeY+j]) > 0.01) {
              wrong = 2;
              break;
            }
          }
          if (wrong) break;
        }

        if (wrong == 1) cout << "[ERROR] BB does not match!" << endl;
        else if (wrong == 2) cout << "[ERROR] AA does not match!" << endl;
        else cout << "[INFO] Data matches perfectly!" << endl;
      }

        free (AACPU);
        free (BBCPU);

    }

    clReleaseMemObject (clAA);
    clReleaseMemObject (clBB);
    clReleaseKernel (kernel);
    free (AA);
    free (BB);
  }
}

void cleanup () {

}
