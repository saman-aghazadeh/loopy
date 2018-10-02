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
  op.addOption ("kern1_name", OPT_STRING, "", "name of the kernel 1 function");
	op.addOption ("kern2_name", OPT_STRING, "", "name of the kernel 2 function");
  op.addOption ("device_type", OPT_STRING, "", "device type (GPU or FPGA)");
  op.addOption ("fpga_op_type", OPT_STRING, "", "FPGA TYPE (NDRANGE or SINGLE)");
  op.addOption ("intensity", OPT_STRING, "", "Setting intensity of the computation");
  op.addOption ("use_channel", OPT_INT, "0", "WHETHER the code is utilizing the channels or not");
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
  cout << "[INFO] First kernel is " << kernel1_name << endl;
  cout << "[INFO] Second kernel is " << kernel2_name << endl;
  cout << "[INFO] Minimum Data Size is " << minDataSize << endl;
  cout << "[INFO] Maximum Data Size is " << maxDataSize << endl;
	cout << "[INFO] Number of passes is " << passes << endl;
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

		int extra = 0;
		if (dataType == "DOUBLE") {
      extra = 40; // 4 bytes
    } else {
			extra = 20; // 2 bytes
    }

		cout << "[INFO] data size is " << dataSize << endl;

    void *A, *B, *BPrime, *C, *D, *E;
    cl_mem clA, clB, clBPrime, clC, clD, clE;

    cl_kernel kernel1 = clCreateKernel (program, kernel1_name.c_str(), &err);
    CL_CHECK_ERROR (err);

    cl_kernel kernel2 = clCreateKernel (program, kernel2_name.c_str(), &err);
    CL_CHECK_ERROR (err);

    A = (void *) malloc (dataSize + extra);
    BPrime = (void *) malloc (dataSize + extra);
    B = (void *) malloc (dataSize + extra);
    C = (void *) malloc (dataSize + extra);
    D = (void *) malloc (dataSize + extra);
    E = (void *) malloc (dataSize + extra);

		int size = 0;
		if (dataType == "INT")
      size = (int)(dataSize / sizeof(int));
    else if (dataType == "SINGLE")
      size = (int)(dataSize / sizeof(float));
    else if (dataType == "DOUBLE")
      size = (int)(dataSize / sizeof(double));

    if (dataType == "INT") {
      for (int i = 0; i < size+2; i++) {
      	((int *)A)[i] = 10;
        ((int *)B)[i] = 10;
        ((int *)BPrime)[i] = 10;
        ((int *)C)[i] = 10;
        ((int *)D)[i] = 10;
        ((int *)E)[i] = 10;
      }
    } else if (dataType == "SINGLE") {
      for (int i = 0; i < size+2; i++) {
        ((float *)A)[i] = 10;
        ((float *)B)[i] = 10;
        ((float *)BPrime)[i] = 10;
        ((float *)C)[i] = 10;
        ((float *)D)[i] = 10;
        ((float *)E)[i] = 10;
      }
    } else if (dataType == "DOUBLE") {
      for (int i = 0; i < size+2; i++) {
        ((double *)A)[i] = 10;
        ((double *)B)[i] = 10;
        ((double *)BPrime)[i] = 10;
        ((double *)C)[i] = 10;
        ((double *)D)[i] = 10;
        ((double *)E)[i] = 10;
      }
    }

    clA = clCreateBuffer (ctx, CL_MEM_WRITE_ONLY, dataSize+extra, NULL, &err);
    CL_CHECK_ERROR (err);

    clB = clCreateBuffer (ctx, CL_MEM_READ_WRITE, dataSize+extra, NULL, &err);
    CL_CHECK_ERROR (err);

    clBPrime = clCreateBuffer (ctx, CL_MEM_READ_WRITE, dataSize+extra, NULL, &err);
    CL_CHECK_ERROR (err);

    clC = clCreateBuffer (ctx, CL_MEM_READ_ONLY, dataSize+extra, NULL, &err);
    CL_CHECK_ERROR (err);

    clD = clCreateBuffer (ctx, CL_MEM_READ_ONLY, dataSize+extra, NULL, &err);
    CL_CHECK_ERROR (err);

    clE = clCreateBuffer (ctx, CL_MEM_READ_ONLY, dataSize+extra, NULL, &err);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clA, CL_TRUE, 0, dataSize+extra, A, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clB, CL_TRUE, 0, dataSize+extra, B, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clBPrime, CL_TRUE, 0, dataSize+extra, BPrime, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clC, CL_TRUE, 0, dataSize+extra, C, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clD, CL_TRUE, 0, dataSize+extra, D, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (queue, clE, CL_TRUE, 0, dataSize+extra, E, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (second_queue, clA, CL_TRUE, 0, dataSize+extra, A, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (second_queue, clB, CL_TRUE, 0, dataSize+extra, B, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (second_queue, clBPrime, CL_TRUE, 0, dataSize+extra, BPrime, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (second_queue, clC, CL_TRUE, 0, dataSize+extra, C, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (second_queue, clD, CL_TRUE, 0, dataSize+extra, D, 0, NULL, NULL);
    CL_CHECK_ERROR (err);

    err = clEnqueueWriteBuffer (second_queue, clE, CL_TRUE, 0, dataSize+extra, E, 0, NULL, NULL);
    CL_CHECK_ERROR (err);


    err = clSetKernelArg (kernel1, 0, sizeof (cl_mem), (void *) &clA);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel1, 1, sizeof (cl_mem), (void *) &clB);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel1, 2, sizeof (cl_mem), (void *) &clBPrime);
    CL_CHECK_ERROR (err);

		err = clSetKernelArg (kernel1, 3, sizeof (cl_mem), (void *) &clC);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel1, 4, sizeof (cl_mem), (void *) &clD);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel1, 5, sizeof (cl_mem), (void *) &clE);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel2, 0, sizeof (cl_mem), (void *) &clA);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel2, 1, sizeof (cl_mem), (void *) &clB);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel2, 2, sizeof (cl_mem), (void *) &clBPrime);
    CL_CHECK_ERROR (err);

		err = clSetKernelArg (kernel2, 3, sizeof (cl_mem), (void *) &clC);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel2, 4, sizeof (cl_mem), (void *) &clD);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel2, 5, sizeof (cl_mem), (void *) &clE);
    CL_CHECK_ERROR (err);

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

        err = clSetKernelArg (kernel1, 6, sizeof (int), &numIterations);
        CL_CHECK_ERROR (err);

        err = clSetKernelArg (kernel2, 6, sizeof (int), &numIterations);
        CL_CHECK_ERROR (err);
      }
    }

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

    // global_work_size[0] = global_work_size[0] / 2;
		cout << "[INFO] global work size is " << global_work_size[0] << endl;

    const size_t local_work_size[] = {(size_t)localX};
		cout << "[INFO] local work size is " << local_work_size[0] << endl;

		Event evKernel1 ("KernelEvent1");
		Event evKernel2 ("KernelEvent2");
    Event evKernel3 ("KernelEvent3");
    Event evKernel4 ("KernelEvent4");

    if (device_type == "FPGA" && fpga_op_type == "SINGLE") {
      if (use_channel == 0) {
      	err = clEnqueueTask (queue, kernel1, 0, NULL, &evKernel1.CLEvent());
      	err = clEnqueueTask (queue, kernel2, 1, &evKernel1.CLEvent(), &evKernel2.CLEvent());
      } else if (use_channel == 1) {
      	err = clEnqueueTask (second_queue, kernel2, 0, NULL, &evKernel2.CLEvent());
       	err = clEnqueueTask (queue, kernel1, 0, NULL, &evKernel1.CLEvent());

      }
    } else {
      err = clEnqueueNDRangeKernel (queue, kernel1, 1,
                                      NULL, global_work_size, local_work_size,
                                      0, NULL, &evKernel1.CLEvent());
      err = clEnqueueNDRangeKernel (queue, kernel2, 1,
                                    	NULL, global_work_size, local_work_size,
                                    	1, &evKernel1.CLEvent(), &evKernel2.CLEvent());
    }

    clFinish (queue);
    CL_BAIL_ON_ERROR (err);

		cout << "[INFO] Done with warmup" << endl;

		for (int iter = 0; iter < passes; iter++) {

			if (dataType == "INT") {
        for (int i = 0; i < size; i++) {
          ((int *)A)[i] = 10;
          ((int *)B)[i] = 10;
          ((int *)BPrime)[i] = 10;
          ((int *)C)[i] = 10;
          ((int *)D)[i] = 10;
          ((int *)E)[i] = 10;
        }
      } else if (dataType == "SINGLE") {
        for (int i = 0; i < size; i++) {
          ((float *)A)[i] = 10;
          ((float *)B)[i] = 10;
          ((float *)BPrime)[i] = 10;
          ((float *)C)[i] = 10;
          ((float *)D)[i] = 10;
          ((float *)E)[i] = 10;
        }
      } else if (dataType == "DOUBLE") {
        for (int i = 0; i < size; i++) {
          ((double *)A)[i] = 10;
          ((double *)B)[i] = 10;
          ((double *)BPrime)[i] = 10;
          ((double *)C)[i] = 10;
          ((double *)D)[i] = 10;
          ((double *)E)[i] = 10;
        }
      }

      err = clEnqueueWriteBuffer (queue, clE, CL_TRUE, 0, dataSize+extra, E, 0, NULL, NULL);
			CL_CHECK_ERROR (err);

      err = clEnqueueWriteBuffer (queue, clC, CL_TRUE, 0, dataSize+extra, C, 0, NULL, NULL);
      CL_CHECK_ERROR (err);

      err = clEnqueueWriteBuffer (queue, clD, CL_TRUE, 0, dataSize+extra, D, 0, NULL, NULL);
      CL_CHECK_ERROR (err);

      err = clEnqueueWriteBuffer (queue, clB, CL_TRUE, 0, dataSize+extra, B, 0, NULL, NULL);
      CL_CHECK_ERROR (err);



      if (device_type == "FPGA" && fpga_op_type == "SINGLE") {
        if (use_channel == false) {
          err = clEnqueueTask (queue, kernel1, 0, NULL, &evKernel1.CLEvent());
          err = clEnqueueTask (queue, kernel2, 1, &evKernel1.CLEvent(), &evKernel2.CLEvent());
        } else if (use_channel == true) {
          err = clEnqueueTask (second_queue, kernel2, 0, NULL, &evKernel2.CLEvent());
          err = clEnqueueTask (queue, kernel1, 0, NULL, &evKernel1.CLEvent());
        }
      } else {
       	err = clEnqueueNDRangeKernel (queue, kernel1, 1,
                                      NULL, global_work_size, local_work_size,
                                      0, NULL, &evKernel1.CLEvent());
        err = clEnqueueNDRangeKernel (queue, kernel2, 1,
                                      NULL, global_work_size, local_work_size,
                                      1, &evKernel1.CLEvent(), &evKernel2.CLEvent());
      }

      clFinish(queue);
      CL_BAIL_ON_ERROR (err);

			cl_ulong totalTime = 0;

			evKernel1.FillTimingInfo();
      evKernel2.FillTimingInfo();

			if (device_type == "FPGA" && fpga_op_type == "SINGLE" && use_channel == 1) {
        cl_ulong start1 = evKernel1.SubmitTime();
        cl_ulong start2 = evKernel2.SubmitTime();

        cl_ulong end1 = evKernel1.EndTime();
        cl_ulong end2 = evKernel2.EndTime();

        cl_ulong start = (start1 > start2) ? start2 : start1;
        cl_ulong end = (end1 > end2) ? end1 : end2;

				totalTime = end - start;

      } else {
        totalTime = evKernel1.SubmitEndRuntime() + evKernel2.SubmitEndRuntime();
      }
      if (dataType == "INT")
      	resultDB.AddResult ("KernelINT" /*+ toString(dataSize) + "KiB"*/,
                          	toString(dataSize), "Bytes", totalTime);
      else if (dataType == "SINGLE")
        resultDB.AddResult ("KernelSINGLE",
                            toString(dataSize), "Bytes", totalTime);
      else if (dataType == "DOUBLE")
        resultDB.AddResult ("KernelDOUBLE",
                            toString(dataSize), "Bytes", evKernel1.SubmitEndRuntime() + evKernel2.SubmitEndRuntime());

      err = clEnqueueReadBuffer (queue, clA, CL_TRUE, 0, dataSize+extra, A, 0, 0, &evRead.CLEvent());

      clFinish (queue);
      CL_BAIL_ON_ERROR (err);

			continue;

			void* ACPU = (void *) (malloc (dataSize+extra));
      void* BPrimeCPU = (void *) (malloc (dataSize+extra));
      void* BCPU = (void *) (malloc (dataSize+extra));
      void* CCPU = (void *) (malloc (dataSize+extra));
      void* DCPU = (void *) (malloc (dataSize+extra));
      void* ECPU = (void *) (malloc (dataSize+extra));

      if (dataType == "INT") {
        for (int i = 0; i < size+2; i++) {
          ((int *)ACPU)[i] = 1;
          ((int *)BCPU)[i] = 1;
          ((int *)BPrimeCPU)[i] = 1;
          ((int *)CCPU)[i] = 1;
          ((int *)DCPU)[i] = 1;
          ((int *)ECPU)[i] = 1;
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
          ((float *)BPrimeCPU)[i] = 1;
          ((float *)CCPU)[i] = 1;
          ((float *)DCPU)[i] = 1;
          ((float *)ECPU)[i] = 1;
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
      free (BPrimeCPU);
			free (CCPU);
      free (DCPU);
      free (ECPU);

    }

    clReleaseMemObject (clA);
    clReleaseMemObject (clB);
    clReleaseMemObject (clBPrime);
    clReleaseMemObject (clC);
    clReleaseMemObject (clD);
    clReleaseMemObject (clE);
    clReleaseKernel (kernel1);
    clReleaseKernel (kernel2);
    clReleaseKernel (kernel3);
    clReleaseKernel (kernel4);
    free (A);
    free (B);
    free (BPrime);
    free (C);
    free (D);
    free (E);
  }
}

void cleanup () {
  
}
