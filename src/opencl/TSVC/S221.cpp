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


void Afunction (float* A, float* B, float* C, float* D) {

	float tempA = 0;
  float tempB = *B;
  float tempC = *C;
  float tempD = *D;

	tempA = cos(tempB) * tempC * tempD;
  tempA = cos(tempA) * tempC;

  *A = tempA;

}

void Afunction2 (float* A, float* B, float* C, float* D) {

	float tempA = 0;
  float tempB = *B;
  float tempC = *C;
  float tempD = *D;

	tempA = cos(tempB) * tempC * tempD;
  tempA = cos(tempA) * tempC;
  tempA = cos(tempA) * tempC;

  *A = tempA;

}

void Afunction3 (float* A, float* B, float* C, float* D) {

	float tempA = 0;
  float tempB = *B;
  float tempC = *C;
  float tempD = *D;

	tempA = cos(tempB) * tempC * tempD;
  tempA = cos(tempA) * tempC;
  tempA = cos(tempA) * tempC;
	tempA = cos(tempA) * tempC;

  *A = tempA;

}

void Afunction4 (float* A, float* B, float* C, float* D) {

	float tempA = 0;
  float tempB = *B;
  float tempC = *C;
  float tempD = *D;

	tempA = cos(tempB) * tempC * tempD;
  tempA = cos(tempA) * tempC;
  tempA = cos(tempA) * tempC;
	tempA = cos(tempA) * tempC;
  tempA = cos(tempA) * tempC;

  *A = tempA;

}

void Afunction5 (float* A, float* B, float* C, float* D) {

	float tempA = 0;
  float tempB = *B;
  float tempC = *C;
  float tempD = *D;

	tempA = cos(tempB) * tempC * tempD;
  tempA = cos(tempA) * tempC;
  tempA = cos(tempA) * tempC;
	tempA = cos(tempA) * tempC;
	tempA = cos(tempA) * tempC;

  *A = tempA;

}


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
  op.addOption ("kern1_name", OPT_STRING, "", "name of the first kernel function");
  op.addOption ("kern2_name", OPT_STRING, "", "name of the second kernel function");
  op.addOption ("device_type", OPT_STRING, "", "device type (GPU or FPGA)");
  op.addOption ("fpga_op_type", OPT_STRING, "", "FPGA TYPE (NDRANGE or SINGLE)");
  op.addOption ("intensity", OPT_STRING, "", "Setting intensity of the computation");
  op.addOption ("use_channel", OPT_INT, "0", "Whether using channel or not");
}

void RunBenchmark (cl_device_id dev,
                   cl_context ctx,
                   cl_command_queue queue,
                   ResultDatabase &resultDB,
                   OptionParser &op) {

	cl_int status;
  cl_command_queue second_queue;

  second_queue = clCreateCommandQueue (ctx, dev, CL_QUEUE_PROFILING_ENABLE, &status);
  CL_CHECK_ERROR (status);

	string int_precision = "-DINT_PRECISION ";
  string single_precision = "-DSINGLE_PRECISION ";
  string double_precision = "-DDOUBLE_PRECISION ";

	long long unsigned  minDataSize = op.getOptionInt("min_data_size") * 1024 * 1024;
	long long unsigned  maxDataSize = op.getOptionInt("max_data_size") * 1024 * 1024;

  int passes = op.getOptionInt("passes");

	string dataType = op.getOptionString("data_type");
	string kernel_location = op.getOptionString("kern_loc");
  string kernel1_name = op.getOptionString("kern1_name");
  string kernel2_name = op.getOptionString("kern2_name");
	string device_type = op.getOptionString("device_type");
  string fpga_op_type = op.getOptionString("fpga_op_type");
	string flags = "";
  string intensity = op.getOptionString("intensity");
  int use_channel = op.getOptionInt("use_channel");

	int localX = 256;
  int globalX = 0;

	cl_int err;
  cl_program program;

  Event evKernel1 ("EventKernel1");
  Event evKernel2 ("EventKernel2");
  Event evRead ("EventRead");

	cout << "[INFO] Device Type is " << device_type << endl;
  cout << "[INFO] Data Type is " << dataType << endl;
  cout << "[INFO] Kernel Location is " << kernel_location << endl;
  cout << "[INFO] kernel one is " << kernel1_name << endl;
	cout << "[INFO] kernel two is " << kernel2_name << endl;
  cout << "[INFO] Minimum Data Size is " << minDataSize << endl;
  cout << "[INFO] Maximum Data Size is " << maxDataSize << endl;
	cout << "[INFO] Number of passes is " << passes << endl;
	cout << "[INFO] use channel is " << use_channel << endl;

  // First building the program

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
  } else if (intensity == "6") {
    flags += "-DINTENSITY6 ";
  } else if (intensity == "7") {
    flags += "-DINTENSITY7 ";
  } else if (intensity == "8") {
    flags += "-DINTENSITY8 ";
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

  CL_CHECK_ERROR (err);

  for (unsigned long long dataSize = minDataSize; dataSize <= maxDataSize; dataSize *= 2) {

		cout << "[INFO] data size is " << dataSize << endl;

    void *A, *B, *C;
    cl_mem clA, clB, clC;

    cl_kernel kernel1 = clCreateKernel (program, kernel1_name.c_str(), &err);
    CL_CHECK_ERROR (err);

		cl_kernel kernel2;
    if (use_channel == 1) {
    	kernel2 = clCreateKernel (program, kernel2_name.c_str(), &err);
    	CL_CHECK_ERROR (err);
  	}
    A = (void *) malloc (dataSize);
    B = (void *) malloc (dataSize);
    C = (void *) malloc (dataSize);

		int size = 0;
		if (dataType == "INT")
      size = (int)(dataSize / sizeof(int));
    else if (dataType == "SINGLE")
      size = (int)(dataSize / sizeof(float));
    else if (dataType == "DOUBLE")
      size = (int)(dataSize / sizeof(double));

    if (dataType == "INT") {
      for (int i = 0; i < size; i++) {
      	((int *)A)[i] = 10;
        ((int *)B)[i] = 10;
        ((int *)C)[i] = 10;
      }
    } else if (dataType == "SINGLE") {
      for (int i = 0; i < size; i++) {
        ((float *)A)[i] = 10;
        ((float *)B)[i] = 10;
        ((float *)C)[i] = 10;
      }
    } else if (dataType == "DOUBLE") {
      for (int i = 0; i < size; i++) {
        ((double *)A)[i] = 10;
        ((double *)B)[i] = 10;
        ((double *)C)[i] = 10;
      }
    }

    clA = clCreateBuffer (ctx, CL_MEM_WRITE_ONLY, dataSize, NULL, &err);
    CL_CHECK_ERROR (err);

    clB = clCreateBuffer (ctx, CL_MEM_READ_WRITE, dataSize, NULL, &err);
    CL_CHECK_ERROR (err);

    clC = clCreateBuffer (ctx, CL_MEM_READ_ONLY, dataSize, NULL, &err);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clA, CL_TRUE, 0, dataSize, A, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clB, CL_TRUE, 0, dataSize, B, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clC, CL_TRUE, 0, dataSize, C, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (second_queue, clA, CL_TRUE, 0, dataSize, A, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (second_queue, clB, CL_TRUE, 0, dataSize, B, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (second_queue, clC, CL_TRUE, 0, dataSize, C, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel1, 0, sizeof (cl_mem), (void *) &clA);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel1, 1, sizeof (cl_mem), (void *) &clB);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel1, 2, sizeof (cl_mem), (void *) &clC);
    CL_CHECK_ERROR (err);

    if (use_channel == 1) {
    	err = clSetKernelArg (kernel2, 0, sizeof (cl_mem), (void *) &clA);
    	CL_CHECK_ERROR (err);

    	err = clSetKernelArg (kernel2, 1, sizeof (cl_mem), (void *) &clB);
    	CL_CHECK_ERROR (err);

    	err = clSetKernelArg (kernel2, 2, sizeof (cl_mem), (void *) &clC);
    	CL_CHECK_ERROR (err);
    }


    if (device_type == "FPGA") {
      if (fpga_op_type == "SINGLE") {
        int numIterations = 0;
        if (dataType == "INT") {
          numIterations = dataSize / 4;
        } else if (dataType == "SINGLE") {
          numIterations = dataSize / 4;
        } else if (dataType == "DOUBLE") {
          numIterations = dataSize / 8;
        }

        numIterations = numIterations;

        err = clSetKernelArg (kernel1, 3, sizeof (int), &numIterations);
        CL_CHECK_ERROR (err);

        if (use_channel == 1) {
        	err = clSetKernelArg (kernel2, 3, sizeof (int), &numIterations);
        	CL_CHECK_ERROR (err);
        }

      }
    }

		clFinish (queue);
    if (use_channel == 1) {
      clFinish (queue);
    }
    CL_BAIL_ON_ERROR (err);


		size_t global_work_size[1];
    if (dataType == "INT") {
      global_work_size[0] = dataSize/4;
    } else if (dataType == "SINGLE") {
      global_work_size[0] = dataSize/4;
    } else if (dataType == "DOUBLE") {
      global_work_size[0] = dataSize/8;
    }

    // global_work_size[0] = global_work_size[0] / 2;
		cout << "[INFO] global work size is " << global_work_size[0] << endl;

    const size_t local_work_size[] = {(size_t)localX};
		cout << "[INFO] local work size is " << local_work_size[0] << endl;

		Event evKernel1 ("KernelEvent1");
    Event evKernel2 ("KernelEvent2");
    long count;

    if (device_type == "FPGA" && fpga_op_type == "SINGLE") {
      if (use_channel == 1) {
      	err = clEnqueueTask (queue, kernel1, 0, NULL, &evKernel1.CLEvent());
      	err = clEnqueueTask (second_queue, kernel2, 0, NULL, &evKernel2.CLEvent());
      } else {
        err = clEnqueueTask (queue, kernel1, 0, NULL, &evKernel1.CLEvent());
      }
    } else {

      auto start = std::chrono::high_resolution_clock::now();
       	err = clEnqueueNDRangeKernel (queue, kernel1, 1,
                                      NULL, global_work_size, local_work_size,
                                      0, NULL, &evKernel1.CLEvent());
        err = clEnqueueReadBuffer (queue, clA, CL_TRUE, 0, dataSize, A, 1,
                                   &evKernel1.CLEvent(), &evRead.CLEvent());
        err = clWaitForEvents (1, &evRead.CLEvent());

        float multiplier = 1.5;
				for (int i = 1; i < global_work_size[0]; i++) {
          if (intensity == "1") {
            Afunction (&(((float *)B)[i]), &(((float *)B)[i-1]), &multiplier, &(((float *)A)[i]));
          } else if (intensity == "2") {
          	Afunction2 (&(((float *)B)[i]), &(((float *)B)[i-1]), &multiplier, &(((float *)A)[i]));
          } else if (intensity == "3") {
        		Afunction3 (&(((float *)B)[i]), &(((float *)B)[i-1]), &multiplier, &(((float *)A)[i]));
          } else if (intensity == "4") {
      			Afunction4 (&(((float *)B)[i]), &(((float *)B)[i-1]), &multiplier, &(((float *)A)[i]));
          } else if (intensity == "5") {
    				Afunction5 (&(((float *)B)[i]), &(((float *)B)[i-1]), &multiplier, &(((float *)A)[i]));
          }
        }

        auto finish = std::chrono::high_resolution_clock::now();
				count = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();

    }

    clFinish (queue);
    CL_BAIL_ON_ERROR (err);

		cout << "[INFO] Done with warmup" << endl;

		for (int iter = 0; iter < passes; iter++) {

			if (dataType == "INT") {
        for (int i = 0; i < size; i++) {
          ((int *)A)[i] = 10;
          ((int *)B)[i] = 10;
          ((int *)C)[i] = 10;
        }
      } else if (dataType == "SINGLE") {
        for (int i = 0; i < size; i++) {
          ((float *)A)[i] = 10;
          ((float *)B)[i] = 10;
          ((float *)C)[i] = 10;
        }
      } else if (dataType == "DOUBLE") {
        for (int i = 0; i < size; i++) {
          ((double *)A)[i] = 10;
          ((double *)B)[i] = 10;
          ((double *)C)[i] = 10;
        }
      }

      long count = 0;
      if (device_type == "FPGA" && fpga_op_type == "SINGLE") {
        if (use_channel == 1) {
        	err = clEnqueueTask (queue, kernel1, 0, NULL, &evKernel1.CLEvent());
        	err = clEnqueueTask (second_queue, kernel2, 0, NULL, &evKernel2.CLEvent());
        } else {
          err = clEnqueueTask (queue, kernel1, 0, NULL, &evKernel1.CLEvent());
        }
      } else {
        auto start = std::chrono::high_resolution_clock::now();
       	err = clEnqueueNDRangeKernel (queue, kernel1, 1,
                                      NULL, global_work_size, local_work_size,
                                      0, NULL, &evKernel1.CLEvent());
        err = clEnqueueReadBuffer (queue, clA, CL_TRUE, 0, dataSize, A, 1,
                                   &evKernel1.CLEvent(), &evRead.CLEvent());
        err = clWaitForEvents (1, &evRead.CLEvent());

        float multiplier = 1.5;
				for (int i = 1; i < global_work_size[0]; i++) {
          if (intensity == "1") {
            Afunction (&(((float *)B)[i]), &(((float *)B)[i-1]), &multiplier, &(((float *)A)[i]));
          } else if (intensity == "2") {
          	Afunction2 (&(((float *)B)[i]), &(((float *)B)[i-1]), &multiplier, &(((float *)A)[i]));
          } else if (intensity == "3") {
        		Afunction3 (&(((float *)B)[i]), &(((float *)B)[i-1]), &multiplier, &(((float *)A)[i]));
          } else if (intensity == "4") {
      			Afunction4 (&(((float *)B)[i]), &(((float *)B)[i-1]), &multiplier, &(((float *)A)[i]));
          } else if (intensity == "5") {
    				Afunction5 (&(((float *)B)[i]), &(((float *)B)[i-1]), &multiplier, &(((float *)A)[i]));
          }
        }

        auto finish = std::chrono::high_resolution_clock::now();
				count = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
      }

      clFinish(queue);
      CL_BAIL_ON_ERROR (err);

			cl_ulong totalTime = 0;

			evKernel1.FillTimingInfo();
      if (use_channel == 1) {
        evKernel2.FillTimingInfo();
      }

      if (device_type == "FPGA" && fpga_op_type == "SINGLE") {
        if (use_channel == 1) {
        	cl_ulong start1 = evKernel1.SubmitEndRuntime();
       		cl_ulong start2 = evKernel2.SubmitEndRuntime();

        	cl_ulong end1 = evKernel1.EndTime();
        	cl_ulong end2 = evKernel2.EndTime();

        	cl_ulong start = (start1 > start2) ? start2 : start1;
        	cl_ulong end = (end1 > end2) ? end2 : end1;

        	totalTime = end - start;
        } else {
          totalTime = evKernel1.SubmitEndRuntime();
        }
      } else {
				totalTime = count;
      }

      if (dataType == "INT")
      	resultDB.AddResult ("KernelINT" /*+ toString(dataSize) + "KiB"*/,
                          	toString(dataSize), "Bytes", totalTime);
      else if (dataType == "SINGLE")
        resultDB.AddResult ("KernelSINGLE",
                            toString(dataSize), "Bytes", totalTime);

      err = clEnqueueReadBuffer (queue, clA, CL_TRUE, 0, dataSize, A, 0, 0, &evRead.CLEvent());

      clFinish (queue);
      CL_BAIL_ON_ERROR (err);

			continue;

			void* ACPU = (void *) (malloc (dataSize));
      void* BCPU = (void *) (malloc (dataSize));
      void* CCPU = (void *) (malloc (dataSize));
      void* DCPU = (void *) (malloc (dataSize));
      void* ECPU = (void *) (malloc (dataSize));
			void* BPrimeCPU = (void *) (malloc(dataSize));

      if (dataType == "INT") {
        for (int i = 0; i < size+2; i++) {
          ((int *)ACPU)[i] = 1;
          ((int *)BCPU)[i] = 1;
          ((int *)CCPU)[i] = 1;
        }

        for (int j = 0; j < size; j++) {
          ((int *)BPrimeCPU)[j+1] = ((int *)BCPU)[j+2] - ((int *)ECPU)[j+1] * ((int *)DCPU)[j+1];
        }

        for (int j = 0; j < size; j++) {
          ((int *)ACPU)[j+1] = ((int *)BPrimeCPU)[j] - ((int *)ECPU)[j+1] * ((int *)DCPU)[j+1];
        }

        int wrong = 0;

        for (int i = 0; i < size; i++) {
          if (((int *)ACPU)[i] != ((int *)A)[i]) {
            wrong = 2;
            break;
          }
        }

        if (wrong == 1) cout << "[ERROR] B does not match!" << endl;
        else if (wrong == 2) cout << "[ERROR] A does not match!" << endl;
        else cout << "[INFO] Data matches perfectly!" << endl;
      } else if (dataType == "SINGLE") {
        for (int i = 0; i < size; i++) {
          ((float *)ACPU)[i] = 1;
          ((float *)BCPU)[i] = 1;
          ((float *)CCPU)[i] = 1;
        }

        for (int j = 0; j < size; j++) {
          ((float *)BPrimeCPU)[j+1] = ((float *)BCPU)[j+2] - ((float *)ECPU)[j+1] * ((float *)DCPU)[j+1];
        }

        for (int j = 0; j < size; j++) {
          ((float *)ACPU)[j+1] = ((float *)BPrimeCPU)[j] - ((float *)ECPU)[j+1] * ((float *)DCPU)[j+1];
        }

        int wrong = 0;

        for (int i = 0; i < size; i++) {
          if (((int *)ACPU)[i] != ((int *)A)[i]) {
            wrong = 2;
            break;
          }
        }

        if (wrong == 1) cout << "[ERROR] B does not match!" << endl;
        else if (wrong == 2) cout << "[ERROR] A does not match!" << endl;
        else cout << "[INFO] Data matches perfectly!" << endl;

      } else if (dataType == "DOUBLE") {
        for (int i = 0; i < size; i++) {
          ((double *)ACPU)[i] = 1;
          ((double *)BCPU)[i] = 1;
          ((double *)BPrimeCPU)[i] = 1;
          ((double *)CCPU)[i] = 1;
          ((double *)DCPU)[i] = 1;
          ((double *)ECPU)[i] = 1;
        }

        for (int j = 0; j < size; j++) {
          ((double *)BPrimeCPU)[j+1] = ((double *)BCPU)[j+2] - ((double *)ECPU)[j+1] * ((double *)DCPU)[j+1];
        }

        for (int j = 0; j < size; j++) {
          ((double *)ACPU)[j+1] = ((double *)BPrimeCPU)[j] - ((double *)ECPU)[j+1] * ((double *)DCPU)[j+1];
        }

        int wrong = 0;

        for (int i = 0; i < size; i++) {
          if (((double *)ACPU)[i] != ((double *)A)[i]) {
            wrong = 2;
            break;
          }
        }

        if (wrong == 1) cout << "[ERROR] B does not match!" << endl;
        else if (wrong == 2) cout << "[ERROR]AA does not match!" << endl;
        else cout << "[INFO] Data matches perfectly!" << endl;

      }

      free (ACPU);
      free (BCPU);
			free (CCPU);
			free (DCPU);
			free (ECPU);
			free (BPrimeCPU);

    }

    clReleaseMemObject (clA);
    clReleaseMemObject (clB);
    clReleaseMemObject (clC);
    clReleaseKernel (kernel1);
    clReleaseKernel (kernel2);
    free (A);
    free (B);
    free (C);
  }
}

void cleanup () {
  
}
