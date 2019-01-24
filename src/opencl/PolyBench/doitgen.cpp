#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cassert>
#include <chrono>

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

#define ALPHA 1.5
#define BETA  1.5
#define GENERATE_PTX false

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
  op.addOption ("kern1_name", OPT_STRING, "", "name of the first kernel function");
  op.addOption ("kern2_name", OPT_STRING, "", "name of the second kernel function");
  op.addOption ("device_type", OPT_STRING, "", "device type (GPU or FPGA)");
  op.addOption ("fpga_op_type", OPT_STRING, "", "FPGA TYPE (NDRANGE or SINGLE)");
  op.addOption ("intensity", OPT_STRING, "", "Setting intensity of the computation");
  op.addOption ("use_channel", OPT_INT, "0", "Whether using channel or not");
  op.addOption ("num_fmas", OPT_STRING, "1", "Number of FMA operations");
}

void RunBenchmark (cl_device_id dev,
                   cl_context ctx,
                   cl_command_queue queue,
                   ResultDatabase &resultDB,
                   OptionParser &op) {

	cl_int status;
  cl_command_queue second_queue;

  second_queue = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &status);
	CL_CHECK_ERROR (status);

  string int_precision = "-DINT_PRECISION ";
  string single_precision = "-DSINGLE_PRECISION ";
  string double_precision = "-DDOUBLE_PRECISION ";

  long long unsigned  minDataSize = op.getOptionInt("min_data_size") * 1024 * 1024;
  long long unsigned  maxDataSize = op.getOptionInt("max_data_size") * 1024 * 1024;

  int passes = op.getOptionInt("passes");

  string dataType = op.getOptionString("data_type");
  string kernel_location = op.getOptionString("kern_loc");
  string kern1_name = op.getOptionString("kern1_name");
  string kern2_name = op.getOptionString("kern2_name");
  string device_type = op.getOptionString("device_type");
  string fpga_op_type = op.getOptionString("fpga_op_type");
  string intensity = op.getOptionString("intensity");
  string flags = "";
	int block_size = op.getOptionInt("block_size");
  string numfmas = op.getOptionString("num_fmas");

  int localX = 32;
  int localY = 32;


  cl_int err;
  cl_program program;

  Event evKernel1 ("EventKernel1");
  Event evKernel2 ("EventKernel2");
  Event evRead ("EventRead");

  cout << "[INFO] Device Type is " << device_type << endl;
  cout << "[INFO] Data Type is " << dataType << endl;
  cout << "[INFO] Kernel Location is " << kernel_location << endl;
  cout << "[INFO] Minimum Data Size is " << minDataSize << endl;
  cout << "[INFO] Maximum Data Size is " << maxDataSize << endl;
  cout << "[INFO] number of passes is " << passes << endl;
  cout << "[INFO] number of fmas are " << numfmas << endl;
  // First building the program

	setenv("CUDA_CACHE_DISABLE", "1", 1);

	flags += "-cl-opt-disable ";
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
		//oss << headerFile.rdbuf();
    //oss << "\n\n\n";
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

    for (int lllY = 128; lllY < max/128; lllY *= 2) {
      int lllX = max / lllY;
      cout << "[INFO] lllX is " << lllX << " and lllY is " << lllY << endl;
      cout << "[INFO] data size is " << dataSize << endl;
      cout << "[INFO] Transformed lllX and lllY are " << lllX << " " << lllY << endl;

			int c4Size = 0;
      if (dataType == "INT") {
        c4Size = lllY * lllY * sizeof (int);
      } else if (dataType == "SINGLE") {
        c4Size = lllY * lllY * sizeof (float);
      } else if (dataType == "DOUBLE") {
        c4Size = lllY * lllY * sizeof (double);
      }

      void *A, *AA, *C4, *sum;
      cl_mem clA, clAA, clC4, clSum;

      cl_kernel kernel1 = clCreateKernel (program, kern1_name.c_str(), &err);
      cl_kernel kernel2 = clCreateKernel (program, kern2_name.c_str(), &err);
      CL_CHECK_ERROR (err);

			cout << "[INFO] both kernels are compiled successfully!" << endl;

      A = (void *) malloc (dataSize);
      AA = (void *) malloc (dataSize);
      sum = (void *) malloc (dataSize);
			C4 = (void *) malloc (c4Size);

      // Initialization of A and B arrays
      int sizeX = 0;
      int sizeY = 0;
      if (lllX * lllY != max) {
        cout << "[ERROR] Wrong spatial and temporal dimensions!" << endl;
        //return ;
      } else {
        sizeX = lllX;
        sizeY = lllY;
      }

      if (dataType == "INT") {
        for (int i = 0; i < sizeX; i++) {
          for (int j = 0; j < sizeY; j++) {
            ((int *)A)[i*sizeY+j] = 1.1;
          }
        }
        for (int i = 0; i < sizeY; i++) {
          for (int j = 0; j < sizeY; j++) {
            ((int *)C4)[i*sizeY+j] = 1.1;
          }
        }
      } else if (dataType == "SINGLE") {
        for (int i = 0; i < sizeX; i++) {
          for (int j = 0; j < sizeY; j++) {
            ((float *)A)[i*sizeY+j] = 1.1;
          }
        }
        for (int i = 0; i < sizeY; i++) {
          for (int j = 0; j < sizeY; j++) {
            ((float *)C4)[i*sizeY+j] = 1.1;
          }
        }
      } else if (dataType == "DOUBLE") {
        for (int i = 0; i < sizeX; i++) {
          for (int j = 0; j < sizeY; j++) {
            ((double *)A)[i*sizeY+j] = 1.1;
          }
        }
        for (int i = 0; i < sizeY; i++) {
          for (int j = 0; j < sizeY; j++) {
            ((double *)C4)[i*sizeY+j] = 1.1;
          }
        }
      }

			cout << "[INFO] Start creating the buffers!" << endl;

      // Initiating the opencl buffers
      clA = clCreateBuffer (ctx, CL_MEM_READ_ONLY, dataSize, NULL, &err);
      CL_CHECK_ERROR (err);

      clAA = clCreateBuffer (ctx, CL_MEM_READ_WRITE, dataSize, NULL, &err);
      CL_CHECK_ERROR (err);

			clC4 = clCreateBuffer (ctx, CL_MEM_READ_ONLY, c4Size, NULL, &err);
      CL_CHECK_ERROR (err);

      clSum = clCreateBuffer (ctx, CL_MEM_READ_WRITE, dataSize, NULL, &err);
      CL_CHECK_ERROR (err);

      err = clEnqueueWriteBuffer (queue, clA, CL_TRUE, 0, dataSize, A, 0, NULL, NULL);
      CL_CHECK_ERROR (err);

      err = clEnqueueWriteBuffer (queue, clC4, CL_TRUE, 0, c4Size, C4, 0, NULL, NULL);
      CL_CHECK_ERROR (err);

      err = clEnqueueWriteBuffer (queue, clSum, CL_TRUE, 0, dataSize, sum, 0, NULL, NULL);
      CL_CHECK_ERROR (err);

			err = clEnqueueWriteBuffer (second_queue, clAA, CL_TRUE, 0, dataSize, AA, 0, NULL, NULL);
      CL_CHECK_ERROR (err);

      err = clEnqueueWriteBuffer (second_queue, clSum, CL_TRUE, 0, dataSize, sum, 0, NULL, NULL);
			CL_CHECK_ERROR (err);

      cout << "[INFO] Buffer for the first and second queues are created and enqueued successfully!" << endl;

      err = clSetKernelArg (kernel1, 0, sizeof (cl_mem), (void *) &clA);
      CL_CHECK_ERROR (err);
      cout << "[INFO] clA is set successfully for kernel1" << endl;

      err = clSetKernelArg (kernel1, 1, sizeof (cl_mem), (void *) &clC4);
      CL_CHECK_ERROR (err);
			cout << "[INFO] clC4 is set successfully for kernel1" << endl;

      err = clSetKernelArg (kernel1, 2, sizeof (cl_mem), (void *) &clSum);
      CL_CHECK_ERROR (err);
      cout << "[INFO] clSum is set successfully for kernel1" << endl;

      err = clSetKernelArg (kernel1, 3, sizeof (int), (void *) &(lllX));
      CL_CHECK_ERROR (err);
      cout << "[INFO] lllX is set successfully for kernel1" << endl;

			err = clSetKernelArg (kernel1, 4, sizeof (int), (void *) &(lllY));
      CL_CHECK_ERROR (err);
      cout << "[INFO] lllX is set successfully for kernel1" << endl;

      err = clSetKernelArg (kernel2, 0, sizeof (cl_mem), (void *) &clAA);
      CL_CHECK_ERROR (err);
      cout << "[INFO] clAA is set successfully for kernel2" << endl;

      err = clSetKernelArg (kernel2, 1, sizeof (cl_mem), (void *) &clSum);
      CL_CHECK_ERROR (err);
      cout << "[INFO] clSum is set successfully for kernel2" << endl;

      err = clSetKernelArg (kernel2, 2, sizeof (int), (void *) &lllX);
      CL_CHECK_ERROR (err);
      cout << "[INFO] lllX is set successfully for kernel2" << endl;

      err = clSetKernelArg (kernel2, 3, sizeof (int), (void *) &lllY);
      CL_CHECK_ERROR (err);
      cout << "[INFO] lllY is set successfully for kernel2" << endl;

      cout << "[INFO] Kernel args are set successfully!" << endl;

      clFinish (queue);
      clFinish (second_queue);

      cout << "Finishing jobs on both queues!" << endl;
      CL_BAIL_ON_ERROR (err);

      const size_t global_work_size[] = {lllY, lllX};

      if (!(device_type == "FPGA" && fpga_op_type == "SINGLE")) {
        cout << "[INFO] global work size is " << global_work_size[0] << " and " << global_work_size[1] << endl;
      }

      const size_t local_work_size[] = {localY, localX};

      if (!(device_type == "FPGA" && fpga_op_type == "SINGLE")) {
        cout << "[INFO] local work size is " << local_work_size[0] << " and " << local_work_size[1] << endl;
			}

      if (device_type == "FPGA" && fpga_op_type == "SINGLE") {
        err = clEnqueueTask (second_queue, kernel2, 0, NULL, &evKernel2.CLEvent());
        err = clEnqueueTask (queue, kernel1, 0, NULL, &evKernel1.CLEvent());

        cout << "[INFO] Both jobs are enqueued successfully!" << endl;
      } else {
        err = clEnqueueNDRangeKernel (queue, kernel1, 2,
                                      NULL, global_work_size, local_work_size,
                                      0, NULL, &evKernel1.CLEvent());
        err = clEnqueueNDRangeKernel (second_queue, kernel2, 2,
                                      NULL, global_work_size, local_work_size,
                                      1, &evKernel1.CLEvent(), &evKernel2.CLEvent());
      }
      clFinish (queue);
      clFinish(second_queue);
      CL_BAIL_ON_ERROR (err);

      cout << "[INFO] Done with warmup" << endl;

      for (int iter = 0; iter < passes; iter++) {
      	if (dataType == "INT") {
        	for (int i = 0; i < sizeX; i++) {
          	for (int j = 0; j < sizeY; j++) {
            	((int *)A)[i*sizeY+j] = 1.1;
          	}
        	}
        	for (int i = 0; i < sizeY; i++) {
          	for (int j = 0; j < sizeY; j++) {
            	((int *)C4)[i*sizeY+j] = 1.1;
          	}
        	}
      	} else if (dataType == "SINGLE") {
        	for (int i = 0; i < sizeX; i++) {
          	for (int j = 0; j < sizeY; j++) {
            	((float *)A)[i*sizeY+j] = 1.1;
          	}
        	}
        	for (int i = 0; i < sizeY; i++) {
          	for (int j = 0; j < sizeY; j++) {
            	((float *)C4)[i*sizeY+j] = 1.1;
          	}
        	}
      	} else if (dataType == "DOUBLE") {
        	for (int i = 0; i < sizeX; i++) {
          	for (int j = 0; j < sizeY; j++) {
            	((double *)A)[i*sizeY+j] = 1.1;
          	}
       		}
        	for (int i = 0; i < sizeY; i++) {
          	for (int j = 0; j < sizeY; j++) {
            	((double *)C4)[i*sizeY+j] = 1.1;
          	}
        	}
      	}

        if (device_type == "FPGA" && fpga_op_type == "SINGLE") {
          err = clEnqueueTask (second_queue, kernel2, 0, NULL, &evKernel2.CLEvent());
          err = clEnqueueTask (queue, kernel1, 0, NULL, &evKernel1.CLEvent());
        } else {
          err = clEnqueueNDRangeKernel (queue, kernel1, 2,
                                        NULL, global_work_size, local_work_size,
                                        0, NULL, &evKernel1.CLEvent());
          err = clEnqueueNDRangeKernel (second_queue, kernel2, 2,
                                        NULL, global_work_size, local_work_size,
                                        1, &evKernel1.CLEvent(), &evKernel2.CLEvent());
        }
        clFinish (queue);
        clFinish (second_queue);
        CL_BAIL_ON_ERROR (err);

				cl_ulong totalTime = 0;

        evKernel1.FillTimingInfo();
        evKernel2.FillTimingInfo();

				if (device_type == "FPGA" && fpga_op_type == "SINGLE") {
          cl_ulong start1 = evKernel1.SubmitTime();
          cl_ulong start2 = evKernel2.SubmitTime();

          cl_ulong end1 = evKernel1.EndTime();
          cl_ulong end2 = evKernel2.EndTime();

          cl_ulong start = (start1 > start2) ? start2 : start1;
          cl_ulong end   = (end1 > end2) ? end1 : end2;

          totalTime = end - start;
        } else {
          totalTime = evKernel1.SubmitEndRuntime() + evKernel2.SubmitEndRuntime();
        }

        if (dataType == "INT")
          resultDB.AddResult ("KernelINT" /*+ toString(dataSize) + "KiB"*/,
                              toString(dataSize)+"-"+toString(lllX)+"-"+toString(lllY), "Bytes", totalTime);
        else if (dataType == "SINGLE")
          resultDB.AddResult ("KernelSINGLE",
                              toString(dataSize)+"-"+toString(lllX)+"-"+toString(lllY), "Bytes", totalTime);
        else if (dataType == "DOUBLE")
          resultDB.AddResult ("KernelDOUBLE",
                              toString(dataSize)+"-"+toString(lllX)+"-"+toString(lllY), "Bytes", totalTime);

        clFinish (queue);
        clFinish (second_queue);
        CL_BAIL_ON_ERROR (err);

      }


      clReleaseMemObject (clA);
      clReleaseMemObject (clAA);
      clReleaseMemObject (clC4);
      clReleaseMemObject (clSum);
      clReleaseKernel (kernel1);
      clReleaseKernel (kernel2);
      free (A);
      free (AA);
      free (C4);
      free (sum);
    }
  }


}

void cleanup () {
  
}

