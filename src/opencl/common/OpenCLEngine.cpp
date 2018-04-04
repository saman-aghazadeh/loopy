#include "OpenCLEngine.h"
#include "NumberGenerator.h"
#include <sys/time.h>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <math.h>

using namespace std;
using namespace std::chrono;

#define VERBOSE true
#define PRIVATE_VECTOR_SIZE 5
#define TEMP_INIT_VALUE 1.0
#define VERIFICATION false
#define GENERATE_PTX true

#define SWI_MODE true

const std::string currentDateTime() {
  char            fmt[64], buf[64];
  struct timeval  tv;
  struct tm       *tm;

  gettimeofday(&tv, NULL);
  tm = localtime(&tv.tv_sec);
  strftime(fmt, sizeof fmt, "%Y-%m-%d %H:%M:%S.%%06u", tm);
  snprintf(buf, sizeof buf, fmt, tv.tv_usec);
  return buf;
}

template class OpenCLEngine<float>;

template <class T>
OpenCLEngine<T>::OpenCLEngine (cl_context context, cl_device_id device,
                               int executionMode, int targetDevice,
                               struct _algorithm_type *tests) {

	this->context = context;
  this->device = device;

  this->executionMode = executionMode;
  this->targetDevice = targetDevice;

  this->tests = tests;
}

template <class T>
OpenCLEngine<T>::~OpenCLEngine () {}

template <class T>
cl_program OpenCLEngine<T>::createProgram (const char* fileName) {

	cl_int err;
  cl_program program;

  if (targetDevice == TargetDevice::GPU) {
  	// Open kernel file and check whether exists or not
    cout << "Filename is " << fileName << endl;
  	std::ifstream kernelFile (fileName, std::ios::in);
  	if (!kernelFile.is_open()) {
    	std::cerr << "Failed to open file for reading!" << fileName << std::endl;
    	exit (0);
  	}

  	std::ostringstream oss;
  	oss << kernelFile.rdbuf();

  	// Create and build a program
  	std::string srcStdStr = oss.str();
		const char *srcStr = srcStdStr.c_str();
  	program = clCreateProgramWithSource (this->context, 1, (const char **)&srcStr,
                                       NULL, &err);
  	CL_CHECK_ERROR (err);
  } else if (targetDevice == TargetDevice::FPGA) {
		string binary_file = aocl_utils::getBoardBinaryFile (fileName, this->device);
    program = aocl_utils::createProgramFromBinary (this->context, binary_file.c_str(), &(this->device), 1);
	}
	cout << "Before cl build program!" << endl;
  err = clBuildProgram (program, 0, NULL, opts, NULL, NULL);
  cout << "Error is " << err << endl;
  if (err != 0) {
    char log[5000];
    size_t retsize = 0;
    err = clGetProgramBuildInfo (program, this->device, CL_PROGRAM_BUILD_LOG,
                                5000*sizeof(char), log, &retsize);
    //CL_CHECK_ERROR (err);

    cout << "Build Error!" << endl;
    cout << "retSize: " << retsize << endl;
    cout << "Log: " << log << endl;
    exit (0);
  }
  CL_CHECK_ERROR (err);

  return program;

}

template <class T>
bool OpenCLEngine<T>::createMemObjects (cl_command_queue queue, cl_mem *memObjects,
                                     int singleElementSize, const long long memSize,
                                     T *data) {

	cl_int err;


  *memObjects = clCreateBuffer (this->context, CL_MEM_READ_WRITE,
                                memSize * singleElementSize, NULL, &err);
  CL_CHECK_ERROR (err);

  if (*memObjects == NULL) {
    std::cerr << "Error creating memory objects. " << std::endl;
    return false;
  }

  // Enqueue data buffer
  Event evWriteData ("write-data");
  err = clEnqueueWriteBuffer (queue, *memObjects, CL_TRUE, 0, memSize * singleElementSize,
                              data, 0, NULL, &evWriteData.CLEvent());
  CL_CHECK_ERROR (err);
  err = clWaitForEvents (1, &evWriteData.CLEvent());
  CL_CHECK_ERROR (err);

  return true;
}

template <class T>
bool OpenCLEngine<T>::refillMemObject (cl_command_queue queue,
                      cl_mem* memObject, int singleElementSize,
                      const long long memSize, T *data) {

  cl_int err;

  Event evWriteData ("write-data");
  err = clEnqueueWriteBuffer (queue, *memObject, true, 0,
                              memSize * singleElementSize, data,
                              0, NULL, &evWriteData.CLEvent());
  CL_CHECK_ERROR (err);
  err = clWaitForEvents (1, &evWriteData.CLEvent());
  CL_CHECK_ERROR (err);

  return true;
}

template <class T>
bool OpenCLEngine<T>::readbackMemObject (cl_command_queue queue,
                                         cl_mem *memObject, int singleElementSize,
                                         const long long memSize, T* data) {
  cl_int err;

  Event evReadbackData ("readback_data");
  err = clEnqueueReadBuffer (queue, *memObject, true, 0,
                             memSize * singleElementSize, data,
                             0, NULL, &evReadbackData.CLEvent());
  err = clFinish (queue);
  CL_CHECK_ERROR (err);
  err = clWaitForEvents (1, &evReadbackData.CLEvent());
  CL_CHECK_ERROR (err);

  return true;
}

template <class T>
void OpenCLEngine<T>::generateCLs () {

	int aIdx = 0;
  while ((tests != 0) && (tests[aIdx].name != 0)) {
    ostringstream oss;
    struct _cl_info test_cl_info;
    struct _algorithm_type temp = tests[aIdx];
		generateSingleCLCode (oss, temp, test_cl_info);
    aIdx++;
  }

}

template <class T>
void OpenCLEngine<T>::generateCLsMetas () {

	int aIdx = 0;
  while ((tests != 0) && (tests[aIdx].name != 0)) {
   	struct _cl_info test_cl_info;
    struct _algorithm_type temp = tests[aIdx];
		generateSingleCLMeta (temp, test_cl_info);
    cl_metas.push_back (test_cl_info);
    aIdx++;
  }

}

template <class T>
void OpenCLEngine<T>::generateSingleCLCode (ostringstream &oss, struct _algorithm_type &test, struct _cl_info &info) {

	if (VERBOSE) cout << "[VERBOSE] Generate Single CL Code for " << test.name << endl;

  if (test.loopCarriedDataDependency == false) {
  	ofstream codeDump;
  	string dumpFileName = gpu_built_kernels_folder + "/" + test.name + ".cl";
  	codeDump.open (dumpFileName.c_str());
		if (!codeDump.is_open()) {
      cout << "[ERROR] Dump File cannot be created or opened!" << endl;
    }

    //  	if (strcmp (test.varType, "double")) {
		//	oss << "#pragma OPENCL EXTENSION cl_khr_fp64: enable" << endl;
    //	oss << endl;
  	//}

		oss << "__kernel void " << test.name << "(__global " << test.varType << " *data, __global " << test.varType << " *rands, int index, int rand_max){" << endl;
    string declFormula = preparedVarDeclFormulaNonArray ((char *)test.varDeclFormula, PRIVATE_VECTOR_SIZE, false);
    insertTab (oss, 1); oss << declFormula << ";" << endl;
    if (!test.doLocalMemory) {
			insertTab (oss ,1); oss << "int gid = get_global_id(0);" << endl;
    } else {
			string localMemoryInit = preparedLocalMemoryInitialization (test.loopsDepth[0], test.varType, (char **)&(test.formula));
      oss << localMemoryInit << endl;
    }
    oss << endl;
    insertTab (oss, 1); oss << test.varInitFormula << ";" << endl;
		if (VERBOSE) cout << "[VERBOSE] Init Formula been inserted successfully!" << endl;

    if (test.doManualUnroll == true ) {
			NumberGenerator numberGenerator (test.loopsDepth[0]);
			if (VERBOSE) cout << "[VERBOSE] Manually Unrolling is True!" << endl;
	  	for (int i = 1; i < test.loopsDepth[0]; i++) {
        string origFormula;
        if (test.randomAccessType == RandomAccessType::SEQUENTIAL)
					origFormula = prepareOriginalFormula ((char *)test.formula, numberGenerator.getNextSequential()+1, (char *) test.variable);
        else if (test.randomAccessType == RandomAccessType::RANDOM)
          origFormula = prepareOriginalFormula ((char *)test.formula, numberGenerator.getNextRandom()+1, (char *) test.variable);
	      insertTab (oss, 1); oss << origFormula << ";" << endl;
	  	}
    } else {
      if (VERBOSE) cout << "[VERBOSE] Manually Unrolling is False!" << endl;
      if (test.randomAccessType != RandomAccessType::SEQUENTIAL)
        cout << "[WARNING] random access type cannot be generated for automatically unrolled kernels!" << endl
             << "\tPlease re-design your \"test.h\" file!" << endl;
      if (test.unrollFactor != 0) {
      	insertTab (oss, 1); oss << "#pragma unroll " << test.unrollFactor << endl;
      }
      insertTab (oss, 1); oss << "for (int i = 0; i < " << test.loopsDepth[0] << "; i++){" << endl;
    	string origFormula = prepareOriginalFormula ((char *)test.formula, 0, (char *) test.variable);
      insertTab (oss, 2); oss << origFormula << ";" << endl;
      insertTab (oss, 1); oss << "}" << endl;
    }

    int pos = -1;
	  char returnBuf[32];

		string returnOpCode = prepareReturnOpCode (test.vectorSize, test.returnFormula);

		//string returnOpCode = string (test.returnFormula);
    //if ((pos = returnOpCode.find("$")) != (-1) )
    //returnOpCode.replace (pos, 1, string("0"));
      //returnOpCode.replace (pos, 1, string("index"));
		if (VERBOSE) cout << "[VERBOSE] Return Op Code been prepared!" << endl;

    insertTab (oss, 1); oss << returnOpCode << ";" << endl;

  	oss << endl;
  	oss << "}" << endl;
    codeDump << oss.str();
    codeDump.close ();

    if (VERBOSE) cout << "[VERBOSE] CL Code for " << test.name << " been created successfully!" << endl;

  }

  else if (test.loopCarriedDataDependency == true) {

		cout << "in lcdd" << endl;

		ofstream codeDump;
    string dumpFileName = gpu_built_kernels_folder + "/" + test.name + ".cl";
		codeDump.open (dumpFileName.c_str());

    oss << "__kernel void " << test.name << "(global " << test.varType << " *data, __global " << test.varType << " *rands, int index, int rand_max){" << endl << endl;

		string declFormula = preparedVarDeclFormulaNonArray ((char *)test.varDeclFormula, test.loopsDepth[0], true);

    insertTab (oss, 1); oss << declFormula << ";" << endl;
    if (!test.doLocalMemory) {
      insertTab (oss, 1); oss << "int depth = " << test.loopsDepth[0] << ";" << endl;
    	insertTab (oss, 1); oss << "int gid = get_global_id(0);" << endl;
    } else {
      string localMemoryInit = preparedLocalMemoryInitialization (test.loopsDepth[0], test.varType, (char **)&(test.variable));
      oss << localMemoryInit << endl;
    }

    oss << endl;
    insertTab (oss, 1); oss << test.varInitFormula << ";" << endl;
		oss << endl;

		if (test.doManualUnroll == true) {
    	for (int i = 1; i < test.loopsDepth[0]; i++) {
      	string origFormula = prepareOriginalFormula ((char *)test.formula, i, (char *)test.variable);
				insertTab (oss, 1); oss << origFormula << ";" << endl;
    	}
    } else {
      insertTab (oss, 1); oss << "for (int j = 0; j < " << test.loopsLengths[0] << "; j++){" << endl;
      insertTab (oss, 2); oss << "for (int i = 0; i < " << test.loopsDepth[0] << "; i++){" << endl;
      string origFormula = prepareOriginalFormula ((char *) test.formula, 0, (char *) test.variable);
      insertTab (oss, 3); oss << origFormula << ";" << endl;
      insertTab (oss, 2); oss << "}" << endl;
			insertTab (oss, 2); oss << "tempbefore = tempnow * rands[j%depth];" << endl;
      insertTab (oss, 1); oss << "}" << endl;
    }

    int pos = -1;
    char returnBuf[32];
    string returnOpCode = string (test.returnFormula);
    if ((pos = returnOpCode.find("$")) != (-1))
      returnOpCode.replace (pos, 1, string("0"));

    insertTab (oss, 1); oss << returnOpCode << ";" << endl;

    oss << endl;
    oss << "}" << endl;
    codeDump << oss.str();
    codeDump.close();

  }
}

template <class T>
void OpenCLEngine<T>::generateSingleCLMeta (_algorithm_type &test, _cl_info &info) {

  memcpy (info.name, (char *)test.name, strlen((char *)test.name));
  if (targetDevice == TargetDevice::GPU)
  	memcpy (info.kernel_location, (char *)(string(gpu_built_kernels_folder + "/" + test.name + ".cl").c_str()),
            strlen((char *)(string(gpu_built_kernels_folder + "/" + test.name + ".cl").c_str())));
  else
    memcpy (info.kernel_location, (char *)(string(fpga_built_kernels_folder + "/" + test.name).c_str()),
            strlen((char *)(string(fpga_built_kernels_folder + "/" + test.name).c_str())));
  info.num_workitems = test.loopsLengths[0];
	info.flops = test.flopCount;

}

// Generate Single-Thread CPU code based on benchmark type struct
//template <class T>
//void OpenCLEngine<T>::generateAlgorithms ( ) {

//  int aIdx = 0;
//  while ((tests != 0) && (tests[aIdx].name != 0)) {
//  	ostringstream oss;
//    struct _algorithm_type temp = tests[aIdx];
//		generateSingleAlgorithm (oss, temp);
//    aIdx++;
//  }
//}

template <class T>
void OpenCLEngine<T>::insertTab (ostringstream &oss, int numTabs) {

	for (int i = 0; i < numTabs; i++) {
    oss << "\t";
  }

}

template <class T>
void OpenCLEngine<T>::print_benchmark () {
  int aIdx = 0;
  while ((tests != 0) && (tests[aIdx].name != 0)) {
    struct _algorithm_type temp = tests[aIdx];

    cout << "[TEST] ";
    cout << "{name:" << temp.name << "},";
    cout << "{numLoops:" << temp.numLoops << "},";
    cout << "{loopsLengths:";
    for (int i = 0; i < temp.numLoops; i++) {
      cout << temp.loopsLengths[i] << ",";
    }
    cout << "},";
    cout << "{loopsDepths:";
    for (int i = 0; i < temp.numLoops; i++) {
      cout << temp.loopsDepth[i] << ",";
    }
    cout << "},";
    cout << "{LoopCarried:" << temp.loopCarriedDataDependency << "},";
    cout << "{loopCarriedDepths:";
    for (int i = 0; i < temp.numLoops; i++) {
      cout << temp.loopCarriedDDLengths[i] << ",";
    }
    cout << "},";
    cout << "{varType:" << temp.varType << "}";
    cout << endl;
    aIdx++;
  }
}

template <class T>
void OpenCLEngine<T>::validate_benchmark () {

	int aIdx = 0;

  while ((tests != 0) && (tests[aIdx].name != 0)) {
    struct _algorithm_type temp = tests[aIdx];

    if (temp.numLoops != 1) {
      cout << "WARNING: Number of loops cannot exceed 1 in current version!\n";
    	exit (0);
    }

    if (temp.loopsLengths.size() != temp.numLoops) {
      cout << "ERROR: Size of loops lengths vector should be similar to number of loops!\n";
      exit (0);
    }

    if (temp.loopsDepth.size() != temp.numLoops) {
      cout << "ERROR: Size of loops depths should be similar to number of loops!\n";
      exit (0);
    }

		if (temp.loopCarriedDDLengths.size() != temp.numLoops) {
      cout << "ERROR: Size of loop carried data depndency lengths should be similar to number of loops!\n";
      exit (0);
    }
    aIdx++;
  }

}

template <class T>
void OpenCLEngine<T>::executionCL (cl_device_id id,
                cl_context ctx,
                cl_command_queue queue,
                ResultDatabase &resultDB,
                OptionParser &op,
                char* precision,
                AlgorithmFactory& algorithmFactory) {

	int verbose = true;
	int npasses = 5;
	int err;
  char sizeStr[128];

	T *hostMem_GIn;
  T *hostMem_GOut;
	T *verification_GOut;
  cl_mem mem_GIn;
  cl_mem mem_GOut;

	if (verbose) cout << "start execution!" << endl;

  int aIdx = 0;

  while (true) {
    Algorithm* algorithm = algorithmFactory.nextAlgorithm ();
    if (algorithm == NULL) break;

    int loopLength = algorithm->getForLoop()->at(0).getNumOfHomogenousWorkItems ();
    if (verbose) cout << "current algorithm name for execution is: " << algorithm->getKernelName () << endl;

    cl_program program;
    cl_kernel kernel;

    long long GInSize = algorithm->getGInSize ();
    long long GOutSize = algorithm->getGOutSize ();

		hostMem_GIn = new T[GInSize];
    hostMem_GOut = new T[GOutSize];
    verification_GOut = new T[GOutSize];

		int targetDevice = algorithm->getAlgorithmTargetDevice (); 
		if (targetDevice == AlgorithmTargetDevice::GPU) {
    	program = createProgram ((algorithm->getKernelLocation ()).c_str());
    } else if (targetDevice == AlgorithmTargetDevice::FPGA) {
      string kernelLoc = algorithm->getKernelLocation ();
      program = createProgram (kernelLoc.substr(0, kernelLoc.size()-3).c_str());
    }

    if (program == NULL)
      exit (0);
    if (verbose) std::cout << "Program Created Successfully!" << endl;

    kernel = clCreateKernel (program, (algorithm->getKernelName ()).c_str(), &err);
    CL_CHECK_ERROR (err);
    if (verbose) cout << "Kernel Created Successfully!" << endl;

		if (GENERATE_PTX) {
      size_t bin_sz;
      err = clGetProgramInfo (program, CL_PROGRAM_BINARY_SIZES,
                              sizeof(size_t), &bin_sz, NULL);
      // Read binary (PTX file) to memory buffer
      unsigned char* bin = (unsigned char*) malloc (bin_sz);
      err = clGetProgramInfo (program, CL_PROGRAM_BINARIES,
                              sizeof(unsigned char *), &bin, NULL);

      FILE* fp = fopen ("binary.ptx", "wb");
      fwrite (bin, sizeof(char), bin_sz, fp);
      fclose (fp);
      free (bin);
    }

		createMemObjects (queue, &mem_GIn, (int) sizeof (T), GInSize, hostMem_GIn);
    CL_CHECK_ERROR (err);

    createMemObjects (queue, &mem_GOut, (int) sizeof (T), GOutSize, hostMem_GOut);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *)&mem_GIn);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void *)&mem_GOut);
    CL_CHECK_ERROR (err);

    // It's time to fill values M, N, and P in version 2 of generator
    if (algorithm->getIsV2 ()) {
      float *M = new float;
			float *N = new float;
      float *P = new float;
      M[0] = algorithm->getM(); N[0] = algorithm->getN(); P[0] = algorithm->getP();
#if SWI_MODE==false
      err = clSetKernelArg (kernel, 2, sizeof (float), (void *)M);
#else
      M[0] = loopLength;
      err = clSetKernelArg (kernel, 2, sizeof (float), (void *)M);
#endif
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel, 3, sizeof (float), (void *)N);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel, 4, sizeof (float), (void *)P);
      CL_CHECK_ERROR (err);
    }


    // Set up input memory for data, first half = second half
    for (int j = 0; j < GInSize; j++)
      hostMem_GIn[j] = (T)(drand48()*5.0);

    int* wsBegin;
    while ((wsBegin = algorithm->nextLocalWorkSize ()) != NULL) {

      char lwsString[10] = {'\0'};
      sprintf (lwsString, "%d", *wsBegin);
      cout << "wsBegin: " << *wsBegin << endl;
			//cout << algorithm->getName() << "-lws" << lwsString << "-" << precision << "-" << "wallclock" << " ";

      for (int pas = 0; pas < npasses; ++pas) {

				//struct timeval timeStart;
        //struct timeval timeEnd;

        //gettimeofday (&timeStart, NULL);

        //high_resolution_clock::time_point t1 = high_resolution_clock::now();
        refillMemObject (queue, &mem_GIn, (int) sizeof (T), GInSize, hostMem_GIn);
        refillMemObject (queue, &mem_GOut, (int) sizeof (T), GOutSize, hostMem_GOut);

        err = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *)&mem_GIn);
        CL_CHECK_ERROR (err);

        err = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void *)&mem_GOut);
        CL_CHECK_ERROR (err);

        // It's time to fill values M, N, and P in version 2 of generator
        if (algorithm->getIsV2 ()) {
          float *M = new float;
          float *N = new float;
          float *P = new float;
          // TODO: Should be un-commented. This is only for testing
          //M[0] = algorithm->getM(); N[0] = algorithm->getN(); P[0] = algorithm->getP();
          M[0] = 0.05; N[0] = 0.05; P[0] = 0.05;
#if SWI_MODE==true
          M[0] = loopLength;
#endif
          err = clSetKernelArg (kernel, 2, sizeof (float), (void *)M);
          CL_CHECK_ERROR (err);

          err = clSetKernelArg (kernel, 3, sizeof (float), (void *)N);
          CL_CHECK_ERROR (err);

          err = clSetKernelArg (kernel, 4, sizeof (float), (void *)P);
          CL_CHECK_ERROR (err);
        }


        size_t* globalWorkSize = new size_t[algorithm->getWorkDim()];
        size_t* localMemSize = new size_t[algorithm->getWorkDim()];

				int skip = 0;

        for (int i = 0; i < algorithm->getWorkDim(); i++) {
          globalWorkSize[i] = (size_t) algorithm->getGlobalWorkSize()[i];
          if (globalWorkSize[i] < 256) skip = 1;
          else localMemSize[i] = wsBegin[i];
        }
        if (skip == 1) break;
        //size_t localWorkSize[1];
        //globalWorkSize[0] = algorithm->getGlobalWorkSize ();
        //localWorkSize[0] = wsBegin;
        //if (verbose)
          //cout << "--Executing kernel with global work size " << globalWorkSize[0] << endl
          //     << "--and local work size " << localWorkSize[0] << endl
          //     << "--and total number of floating point operations as " << algorithm->getTotalNumFlops ()
          //     << endl;
        Event evKernel (algorithm->getKernelName ());
#if SWI_MODE==false
        //struct timeval timeStart;
        //struct timeval timeEnd;
        //gettimeofday (&timeStart, NULL);
        clFinish (queue);
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        err = clEnqueueNDRangeKernel (queue, kernel, algorithm->getWorkDim(),
	                                    NULL,
                                      globalWorkSize,
                                      localMemSize,
                                      0, NULL, &evKernel.CLEvent());

#else
        clFinish (queue);
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        err = clEnqueueTask (queue, kernel, 0, NULL, &evKernel.CLEvent());
#endif
        CL_CHECK_ERROR (err);
        err = clFinish (queue);
        CL_CHECK_ERROR (err);
        err = clWaitForEvents (1, &evKernel.CLEvent());
        CL_CHECK_ERROR (err);
        //cout << "millisEnd: " << millisEnd << endl;
        //gettimeofday (&timeEnd, NULL);
        //long millisStart = timeStart.tv_usec;
				//long millisStart = (timeStart.tv_sec * 1000) + (timeStart.tv_usec/1000);
        //cout << "millisStart: " << millisStart << "," << timeStart.tv_sec << endl;
        //long millisEnd = (timeEnd.tv_sec * 1000) + (timeEnd.tv_usec/1000);
        //long millisEnd = timeEnd.tv_usec;
        //cout << "millisEnd: " << millisEnd << "," << timeEnd.tv_sec << endl;
                                                                      //long totalTime = (millisEnd - millisStart);
        //cout << "err is " << err << endl;
        //CL_CHECK_ERROR (err);

        //usleep (100000);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        evKernel.FillTimingInfo ();
        readbackMemObject (queue, &mem_GOut, (int) sizeof (T), GOutSize, hostMem_GOut);
        //cout << "Start Time: " << evKernel.StartTime() << endl;
        //cout << "End Time: " << evKernel.EndTime() << endl;
        //cout << "Total Time: " << evKernel.StartEnRuntime () << endl;
        //cout << "Total Time: " << totalTime << endl;
        //cout << "millisEnd: " << millisEnd << endl;
        //double TNF = algorithm->getTotalNumFlops ();
        //double time = evKernel.StartEndRuntime ();
        //gettimeofday (&timeEnd, NULL);
        //high_resolution_clock::time_point t2 = high_resolution_clock::now();
        cout << algorithm->getKernelLocation() << "-" << algorithm->getName() << "-lws" << lwsString << "-" << precision << " " << (double)(evKernel.StartEndRuntime()) << endl;
        cout << "Calculating gflops" << endl;
      	double gflop = (double)algorithm->getTotalNumFlops () / (double)(evKernel.StartEndRuntime());
        //double gflop = (double)algorithm->getTotalNumFlops () / (double)(duration_cast<nanoseconds>(t2-t1).count());
				//sprintf (sizeStr, "Size: %07d", algorithm->getGlobalWorkSize ());
        cout << "Adding to database" << endl;
        resultDB.AddResult (string(algorithm->getKernelLocation ()) + string("-") + string(algorithm->getName()) + string("-ll") + to_string(loopLength) + string("-lws") + string (lwsString) + string ("-") + string(precision), sizeStr, "GFLOPS", gflop);

				if (VERIFICATION) {
					//algorithm->verify(hostMem_GOut, GOutSize, algorithm->getM(), algorithm->getN(), algorithm->getP(),
          //                  algorithm->getGlobalWorkSize()[0], wsBegin[0]);
          algorithm->verify(hostMem_GOut, GOutSize, 0.05, 0.05, 0.05, algorithm->getGlobalWorkSize()[0], wsBegin[0]);
        }

        CL_CHECK_ERROR (err);
      }
      cout << endl;
    }

    cout << "Releasing" << endl;

    err = clReleaseKernel (kernel);
    CL_CHECK_ERROR (err);
    err = clReleaseProgram (program);
    CL_CHECK_ERROR (err);
    err = clReleaseMemObject (mem_GIn);
    CL_CHECK_ERROR (err);
    err = clReleaseMemObject (mem_GOut);
    CL_CHECK_ERROR (err);

		cout << "deleting" << endl;

    delete[] hostMem_GIn;
    cout << "Delete Gin" << endl;
    delete[] hostMem_GOut;
    cout << "Delete GOut" << endl;
    delete[] verification_GOut;
    cout << "Delete verification" << endl;
	}

}

template <class T>
void OpenCLEngine<T>::executeMatrixPipeline (cl_device_id id,
                                             cl_context ctx,
                                             cl_command_queue queue,
                                             ResultDatabase &resultDB,
                                             OptionParser &op,
                                             char* precision,
                                             AlgorithmFactory& algorithmFactory,
                                             int A_height, int A_width,
                                             int B_height, int B_width,
                                             int C_height, int C_width,
                                             int batch_size) {

	int verbose = true;
  int npasses = 10;
  double totalOffset = 0;
  int err;
  cl_int clErr;
  char sizeStr[128];

  T *hostMem_A;
  T *hostMem_B;
  T *hostMem_I;
  T *hostMem_C;
  T *hostMem_R;
  T *verification_R;

  cl_mem mem_A;
  cl_mem mem_B;
  cl_mem mem_I;
  cl_mem mem_C;
  cl_mem mem_R;

	if (verbose) cout << "Creating queue for the second kernel" << endl;

  cl_command_queue second_queue = clCreateCommandQueue (ctx, id, CL_QUEUE_PROFILING_ENABLE, &clErr);
  CL_CHECK_ERROR (clErr);


  if (verbose) cout << "Start Execution!" << endl;

  Algorithm* algorithm = algorithmFactory.nextAlgorithm ();


	cl_program program;
  cl_kernel kernel1;
  cl_kernel kernel2;

  long long ASize = A_height * A_width * batch_size;
  long long BSize = B_height * B_width * batch_size;
  long long ISize = A_height * B_width * batch_size;
  long long CSize = C_height * C_width * batch_size;
  long long RSize = A_height * C_width * batch_size;

	hostMem_A = new T[ASize];
  hostMem_B = new T[BSize];
  hostMem_I = new T[ISize];
  hostMem_C = new T[CSize];
  hostMem_R = new T[RSize];
	verification_R = new T[RSize];


	int targetDevice = algorithm->getAlgorithmTargetDevice ();
	if (targetDevice == AlgorithmTargetDevice::GPU) {
    if (verbose) cout << algorithm->getKernelLocation() << endl;
    program = createProgram ((algorithm->getKernelLocation ()).c_str());
  } else if (targetDevice == AlgorithmTargetDevice::FPGA) {
    string kernelLoc = algorithm->getKernelLocation ();
    program = createProgram (kernelLoc.substr(0, kernelLoc.size()-3).c_str());
  }

	if (program == NULL) exit (0);
  if (verbose) cout << "Program Created Successfully!" << endl;

  string kernel1Name;
	if (targetDevice == AlgorithmTargetDevice::FPGA) {
    kernel1Name = algorithm->getKernelName () + string("1Size8");
  } else {
    kernel1Name = algorithm->getKernelName ();
  }

  string kernel2Name;
  if (targetDevice == AlgorithmTargetDevice::FPGA) {
    kernel2Name = algorithm->getKernelName () + string("2Size8");
  } else {
    kernel2Name = algorithm->getKernelName ();
  }

  kernel1 = clCreateKernel (program, kernel1Name.c_str(), &err);
  if (verbose) cout << "Kernel name is " << kernel1Name << endl;
  CL_CHECK_ERROR (err);
	kernel2 = clCreateKernel (program, kernel2Name.c_str(), &err);
  if (verbose) cout << "Kernel name is " << kernel2Name << endl;
  CL_CHECK_ERROR (err);
  if (verbose) cout << "Kernel Created Successfully!" << endl;

  if (GENERATE_PTX) {
    size_t bin_sz;
    err = clGetProgramInfo (program, CL_PROGRAM_BINARY_SIZES,
                            sizeof(size_t), &bin_sz, NULL);

    // Read binary (PTX File) to memory buffer
    unsigned char* bin = (unsigned char*) malloc (bin_sz);
    err = clGetProgramInfo (program, CL_PROGRAM_BINARIES,
                            sizeof(unsigned char *), &bin, NULL);

    FILE* fp = fopen ("binary.ptx", "wb");
    fwrite (bin, sizeof(char), bin_sz, fp);
    fclose (fp);
    free (bin);
  }

  if (verbose) cout << "Create mem object for A matrix" << endl;
  createMemObjects (queue, &mem_A, (int) sizeof (T), ASize, hostMem_A);
  CL_CHECK_ERROR (err);

  if (verbose) cout << "Create mem object for B matrix" << endl;
  createMemObjects (queue, &mem_B, (int) sizeof (T), BSize, hostMem_B);
  CL_CHECK_ERROR (err);

  if (targetDevice == TargetDevice::GPU) {
  	if (verbose) cout << "Create mem object for I matrix" << endl;
		createMemObjects (queue, &mem_I, (int) sizeof (T), ISize, hostMem_I);
  }

  if (targetDevice == TargetDevice::GPU) {
  	if (verbose) cout << "Create mem object for C matrix" << endl;
  	createMemObjects (queue, &mem_C, (int) sizeof (T), CSize, hostMem_C);
  	CL_CHECK_ERROR (err);

  	if (verbose) cout << "Create mem object for R matrix" << endl;
		createMemObjects (queue, &mem_R, (int) sizeof (T), RSize, hostMem_R);
  	CL_CHECK_ERROR (err);
  } else {
  	if (verbose) cout << "Create mem object for C matrix" << endl;
  	createMemObjects (second_queue, &mem_C, (int) sizeof (T), CSize, hostMem_C);
  	CL_CHECK_ERROR (err);

  	if (verbose) cout << "Create mem object for R matrix" << endl;
		createMemObjects (second_queue, &mem_R, (int) sizeof (T), RSize, hostMem_R);
  	CL_CHECK_ERROR (err);
  }

	if (verbose) cout << "Start setting arguments for kernel 1" << endl;

  err = clSetKernelArg (kernel1, 0, sizeof (cl_mem), (void *)&mem_A);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel1, 1, sizeof (cl_mem), (void *)&mem_B);
  CL_CHECK_ERROR (err);

  if (targetDevice == AlgorithmTargetDevice::GPU) {
  	err = clSetKernelArg (kernel1, 2, sizeof (cl_mem), (void *)&mem_I);
  	CL_CHECK_ERROR (err);

  	err = clSetKernelArg (kernel1, 3, sizeof (int), (void *)&A_height);
		CL_CHECK_ERROR (err);

  	err = clSetKernelArg (kernel1, 4, sizeof (int), (void *)&A_width);
  	CL_CHECK_ERROR (err);

  	err = clSetKernelArg (kernel1, 5, sizeof (int), (void *)&B_height);
  	CL_CHECK_ERROR (err);

  	err = clSetKernelArg (kernel1, 6, sizeof (int), (void *)&B_width);
  	CL_CHECK_ERROR (err);

  	err = clSetKernelArg (kernel1, 7, sizeof (int), (void *)&A_height);
  	CL_CHECK_ERROR (err);

  	err = clSetKernelArg (kernel1, 8, sizeof (int), (void *)&B_width);
  	CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel1, 9, sizeof (int), (void *)&batch_size);
    CL_CHECK_ERROR (err);
  } else {
    err = clSetKernelArg (kernel1, 2, sizeof (int), (void *)&A_height);
    CL_CHECK_ERROR (err);

		err = clSetKernelArg (kernel1, 3, sizeof (int), (void *)&A_width);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel1, 4, sizeof (int), (void *)&B_height);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel1, 5, sizeof (int), (void *)&B_width);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel1, 6, sizeof (int), (void *)&batch_size);
    CL_CHECK_ERROR (err);
  }

	//err = clSetKernelArg (kernel1, 9, sizeof (int), (void *)batch_size);
  //CL_CHECK_ERROR (err);

	if (verbose) cout << "Setting arguments for kernel 1 is completed!" << endl;

	if (verbose) cout << "Start setting arguments for kernel 2" << endl;

  if (targetDevice == AlgorithmTargetDevice::GPU) {
  	err = clSetKernelArg (kernel2, 0, sizeof (cl_mem), (void *)&mem_I);
  	CL_CHECK_ERROR (err);

  	err = clSetKernelArg (kernel2, 1, sizeof (cl_mem), (void *)&mem_C);
  	CL_CHECK_ERROR (err);

  	err = clSetKernelArg (kernel2, 2, sizeof (cl_mem), (void *)&mem_R);
  	CL_CHECK_ERROR (err);

  	err = clSetKernelArg (kernel2, 3, sizeof (int), (void *)&A_height);
  	CL_CHECK_ERROR (err);

  	err = clSetKernelArg (kernel2, 4, sizeof (int), (void *)&B_width);
  	CL_CHECK_ERROR (err);

  	err = clSetKernelArg (kernel2, 5, sizeof (int), (void *)&C_height);
  	CL_CHECK_ERROR (err);

  	err = clSetKernelArg (kernel2, 6, sizeof (int), (void *)&C_width);
  	CL_CHECK_ERROR (err);

  	err = clSetKernelArg (kernel2, 7, sizeof (int), (void *)&A_height);
  	CL_CHECK_ERROR (err);

  	err = clSetKernelArg (kernel2, 8, sizeof (int), (void *)&C_width);
  	CL_CHECK_ERROR (err);
  } else {
    err = clSetKernelArg (kernel2, 0, sizeof (cl_mem), (void *)&mem_C);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel2, 1, sizeof (cl_mem), (void *)&mem_R);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel2, 2, sizeof (int), (void *)&A_height);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel2, 3, sizeof (int), (void *)&B_width);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel2, 4, sizeof (int), (void *)&C_height);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel2, 5, sizeof (int), (void *)&C_width);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel2, 6, sizeof (int), (void *)&batch_size);
    CL_CHECK_ERROR (err);
  }

    //err = clSetKernelArg (kernel2, 3, sizeof (int), (void *)batch_size);
    //CL_CHECK_ERROR (err);

  if (verbose) cout << "Setting arguments for kernel 2 is completed!" << endl;

  // Setup input memories for A, B, and C
  for (int i = 0; i < ASize; i++) {
    hostMem_A[i] = (T)(drand48()*5.0);
  }
  for (int i = 0; i < BSize; i++) {
    hostMem_B[i] = (T)(drand48()*5.0);
  }
  for (int i = 0; i < CSize; i++) {
    hostMem_C[i] = (T)(drand48()*5.0);
  }

  for (int pas = 0; pas < npasses; pas++) {
  	refillMemObject (queue, &mem_A, (int) sizeof (T), ASize, hostMem_A);
    refillMemObject (queue, &mem_B, (int) sizeof (T), BSize, hostMem_B);
    if (targetDevice == TargetDevice::GPU)
    	refillMemObject (queue, &mem_C, (int) sizeof (T), CSize, hostMem_C);
    else
			refillMemObject (second_queue, &mem_C, (int) sizeof (T), CSize, hostMem_C);

		cout << "Start setting arguments for kernel 1" << endl;

    err = clSetKernelArg (kernel1, 0, sizeof (cl_mem), (void *)&mem_A);
    CL_CHECK_ERROR (err);

    err = clSetKernelArg (kernel1, 1, sizeof (cl_mem), (void *)&mem_B);
		CL_CHECK_ERROR (err);

    if (targetDevice == TargetDevice::GPU) {
    	err = clSetKernelArg (kernel1, 2, sizeof (cl_mem), (void *)&mem_I);
    	CL_CHECK_ERROR (err);

			err = clSetKernelArg (kernel1, 3, sizeof (int), (void *)&A_height);
    	CL_CHECK_ERROR (err);

    	err = clSetKernelArg (kernel1, 4, sizeof (int), (void *)&A_width);
    	CL_CHECK_ERROR (err);

    	err = clSetKernelArg (kernel1, 5, sizeof (int), (void *)&B_height);
    	CL_CHECK_ERROR (err);

    	err = clSetKernelArg (kernel1, 6, sizeof (int), (void *)&B_width);
  		CL_CHECK_ERROR (err);

    	err = clSetKernelArg (kernel1, 7, sizeof (int), (void *)&A_height);
    	CL_CHECK_ERROR (err);

    	err = clSetKernelArg (kernel1, 8, sizeof (int), (void *)&B_width);
    	CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel1, 9, sizeof (int), (void *)&batch_size);
      CL_CHECK_ERROR (err);
    } else {
      err = clSetKernelArg (kernel1, 2, sizeof (int), (void *)&A_height);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel1, 3, sizeof (int), (void *)&A_width);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel1, 4, sizeof (int), (void *)&B_height);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel1, 5, sizeof (int), (void *)&B_width);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel1, 6, sizeof (int), (void *)&batch_size);
      CL_CHECK_ERROR (err);
    }

		//err = clSetKernelArg (kernel1, 3, sizeof (int), (void *)batch_size);
    //CL_CHECK_ERROR (err);

		cout << "Start setting argument for kernel 2" << endl;

    if (targetDevice == TargetDevice::GPU) {
    	err = clSetKernelArg (kernel2, 0, sizeof (cl_mem), (void *)&mem_I);
			CL_CHECK_ERROR (err);

    	err = clSetKernelArg (kernel2, 1, sizeof (cl_mem), (void *)&mem_C);
			CL_CHECK_ERROR (err);

    	err = clSetKernelArg (kernel2, 2, sizeof (cl_mem), (void *)&mem_R);
			CL_CHECK_ERROR (err);

			err = clSetKernelArg (kernel2, 3, sizeof (int), (void *)&A_height);
    	CL_CHECK_ERROR (err);

   		err = clSetKernelArg (kernel2, 4, sizeof (int), (void *)&B_width);
    	CL_CHECK_ERROR (err);

    	err = clSetKernelArg (kernel2, 5, sizeof (int), (void *)&C_height);
    	CL_CHECK_ERROR (err);

    	err = clSetKernelArg (kernel2, 6, sizeof (int), (void *)&C_width);
    	CL_CHECK_ERROR (err);

    	err = clSetKernelArg (kernel2, 7, sizeof (int), (void *)&A_height);
    	CL_CHECK_ERROR (err);

    	err = clSetKernelArg (kernel2, 8, sizeof (int), (void *)&C_width);
    	CL_CHECK_ERROR (err);

    	err = clSetKernelArg (kernel2, 3, sizeof (int), (void *)batch_size);
			CL_CHECK_ERROR (err);
    } else {
      err = clSetKernelArg (kernel2, 0, sizeof (cl_mem), (void *)&mem_C);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel2, 1, sizeof (cl_mem), (void*)&mem_R);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel2, 2, sizeof (int), (void *)&A_height);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel2, 3, sizeof (int), (void *)&B_width);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel2, 4, sizeof (int), (void *)&C_height);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel2, 5, sizeof (int), (void *)&C_width);
      CL_CHECK_ERROR (err);

      err = clSetKernelArg (kernel2, 6, sizeof (int), (void *)&batch_size);
      CL_CHECK_ERROR (err);
    }
		const size_t GWS1[3] = {A_height, B_width, batch_size};
    const size_t LWS1[3] = {8, 8, 1};


    const size_t GWS2[3] = {A_height, C_width, batch_size};
    const size_t LWS2[3] = {8, 8, 1};

		Event evKernel1 ("Kernel1");
    Event evKernel2 ("Kernel2");

    clFinish (queue);
    clFinish (second_queue);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    if (targetDevice == TargetDevice::GPU) {
			err = clEnqueueNDRangeKernel (queue,
                                  	kernel1,
                                  	3,
                                  	NULL,
                                  	GWS1,
                                  	LWS1,
                                  	0,
                                  	NULL,
                                  	&evKernel1.CLEvent());
    } else {
      err = clEnqueueTask (queue, kernel1, 0, NULL, &evKernel1.CLEvent());
    }

		//clFinish (queue);
    //CL_CHECK_ERROR (err);
    //err = clFinish (queue);
    //CL_CHECK_ERROR (err);
    //err = clWaitForEvents (1, &evKernel1.CLEvent());
    CL_CHECK_ERROR (err);
    //evKernel1.FillTimingInfo ();

    if (targetDevice == TargetDevice::GPU) {
			err = clEnqueueNDRangeKernel (queue,
                                  	kernel2,
                                  	3,
                                  	NULL,
                                  	GWS2,
                                  	LWS2,
                                  	0,
                                  	NULL,
                                  	&evKernel2.CLEvent());
    } else {
      err = clEnqueueTask (second_queue, kernel2, 0, NULL, &evKernel2.CLEvent());
    }

    //clFinish (queue);
    //CL_CHECK_ERROR (err);
    //err =clFinish (queue);
    CL_CHECK_ERROR (err);
    err = clWaitForEvents (1, &evKernel2.CLEvent());
    err = clWaitForEvents (1, &evKernel1.CLEvent());
    CL_CHECK_ERROR (err);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

		clFinish (queue);
    clFinish (second_queue);

		evKernel2.FillTimingInfo ();

		cout << "Reading back mem object" << endl;

    if (pas >= 2)
			//totalOffset += (duration_cast<nanoseconds>(t2-t1).count());
      totalOffset += evKernel2.StartEndRuntime ();

		readbackMemObject (queue, &mem_R, (int) sizeof (T), RSize, hostMem_R);

		cout << algorithm->getKernelLocation () << "-" << algorithm->getName() << "-" << batch_size << " " << (double)(duration_cast<nanoseconds>(t2-t1).count()) << endl;

		double gflops = (((A_height * B_width) * 2 * A_width) * batch_size + ((A_height * C_width) * 2 * B_width) * batch_size) / (double)(duration_cast<nanoseconds>(t2-t1).count());

    if (pas >= 2) {
    	resultDB.AddResult (string(algorithm->getKernelLocation ()) + string("-") + string(algorithm->getName()) + string("-") + string("b") + to_string(batch_size), string(), "GFLOPS", gflops );
 			
    	//resultDB.AddResult (string(algorithm->getKernelLocation ()) + string("-") + string(algorithm->getName()) + string("-") + string("b") + to_string(batch_size) + string("-") + string("outRate"), string(), "Matrix/ms", batch_size / (((double)(duration_cast<nanoseconds>(t2-t1).count()) - this->offset) / 1000000.0));
    	resultDB.AddResult (string(algorithm->getKernelLocation ()) + string("-") + string(algorithm->getName()) + string("-") + string("b") + to_string(batch_size) + string("-") + string("outRate"), string(), "Matrix/ms", batch_size / ((evKernel2.StartEndRuntime() - this->offset) / 1000000.0));
    	//resultDB.AddResult (string(algorithm->getKernelLocation ()) + string("-") + string(algorithm->getName()) + string("-") + string("b") + to_string(batch_size) + string("-") + string("outRate"), string(), "Matrix/ms", batch_size / (((double)(duration_cast<nanoseconds>(t2-t1).count())) / 1000000.0));
    }

    if (VERIFICATION) {
      algorithm->verifyMatrixPipeline (hostMem_A, hostMem_B, hostMem_C, verification_R,
                                       A_height, A_width,
                                       B_height, B_width,
                                       C_height, C_width,
                                       batch_size);

			if (verbose) cout << "Verification is done, now we get into comparison!" << endl;

      for (int element = 0; element < A_height*C_width*batch_size; element++) {
        float diff = verification_R[element] - hostMem_R[element];
        if (diff > 0.1 || diff < -0.1) {
          cout << "[ERROR] Verification Failed!" << endl;
          break;
        }
      }

     	cout << "Verification Passed!" << endl;
    }

    CL_CHECK_ERROR (err);
  }

  if (batch_size == 1)
  	this->offset = (totalOffset / (npasses-2)) * 0.85;

  cout << "offset is " << this->offset << endl;
  cout << endl;

  cout << "time to release" << endl;

  err = clReleaseKernel (kernel1);
  CL_CHECK_ERROR (err);
  err = clReleaseKernel (kernel2);
  CL_CHECK_ERROR (err);
  err = clReleaseMemObject (mem_A);
  CL_CHECK_ERROR (err);
  err = clReleaseMemObject (mem_B);
  CL_CHECK_ERROR (err);
  err = clReleaseMemObject (mem_C);
  CL_CHECK_ERROR (err);
  if (targetDevice == TargetDevice::GPU) {
		err = clReleaseMemObject (mem_I);
  	CL_CHECK_ERROR (err);
  }
  err = clReleaseMemObject (mem_R);
  CL_CHECK_ERROR (err);

  clReleaseCommandQueue (second_queue);


  delete[] hostMem_A;
  delete[] hostMem_B;
  delete[] hostMem_C;
  delete[] hostMem_R;
  delete[] hostMem_I;
  delete[] verification_R;

}

template <class T>
void OpenCLEngine<T>::executeMatrixPipeline2 (cl_device_id id,
                                             cl_context ctx,
                                             cl_command_queue queue,
                                             ResultDatabase &resultDB,
                                             OptionParser &op,
                                             char* precision,
                                             AlgorithmFactory& algorithmFactory,
                                             int A_height, int A_width,
                                             int B_height, int B_width,
                                             int C_height, int C_width,
                                             int batch_size) {

	int verbose = true;
  int npasses = 10;
  double totalOffset = 0;
  int err;
  cl_int clErr;
  char sizeStr[128];

  T *hostMem_A;
  T *hostMem_B;
  T *hostMem_I;
  T *hostMem_C;
  T *hostMem_R;
  T *verification_R;

  cl_mem mem_A;
  cl_mem mem_B;
  cl_mem mem_I;
  cl_mem mem_C;
  cl_mem mem_R;

	//if (verbose) cout << "Creating queue for the second kernel" << endl;

  //cl_command_queue second_queue = clCreateCommandQueue (ctx, id, CL_QUEUE_PROFILING_ENABLE, &clErr);
  //CL_CHECK_ERROR (clErr);


  if (verbose) cout << "Start Execution!" << endl;

  Algorithm* algorithm = algorithmFactory.nextAlgorithm ();


	cl_program program;
  cl_kernel kernel;

  long long ASize = A_height * A_width * batch_size;
  long long BSize = B_height * B_width * batch_size;
  long long ISize = A_height * B_width * batch_size;
  long long CSize = C_height * C_width * batch_size;
  long long RSize = A_height * C_width * batch_size;

	hostMem_A = new T[ASize];
  hostMem_B = new T[BSize];
  hostMem_I = new T[ISize];
  hostMem_C = new T[CSize];
  hostMem_R = new T[RSize];
	verification_R = new T[RSize];


	int targetDevice = algorithm->getAlgorithmTargetDevice ();
	if (targetDevice == AlgorithmTargetDevice::GPU) {
    if (verbose) cout << algorithm->getKernelLocation() << endl;
    program = createProgram ((algorithm->getKernelLocation ()).c_str());
  } else if (targetDevice == AlgorithmTargetDevice::FPGA) {
    string kernelLoc = algorithm->getKernelLocation ();
    program = createProgram (kernelLoc.substr(0, kernelLoc.size()-3).c_str());
  }

	if (program == NULL) exit (0);
  if (verbose) cout << "Program Created Successfully!" << endl;

  string kernelName;
	//if (targetDevice == AlgorithmTargetDevice::FPGA) {
    kernelName = algorithm->getKernelName ();
    //} else {
    //kernel1Name = algorithm->getKernelName ();
    //}

    //string kernel2Name;
    //if (targetDevice == AlgorithmTargetDevice::FPGA) {
    //kernel2Name = algorithm->getKernelName () + string("2Size8");
    //} else {
    //kernel2Name = algorithm->getKernelName ();
    //}

  kernel = clCreateKernel (program, kernelName.c_str(), &err);
  if (verbose) cout << "Kernel name is " << kernelName << endl;
  CL_CHECK_ERROR (err);
	//kernel2 = clCreateKernel (program, kernel2Name.c_str(), &err);
  //if (verbose) cout << "Kernel name is " << kernel2Name << endl;
  //CL_CHECK_ERROR (err);
  if (verbose) cout << "Kernel Created Successfully!" << endl;

  if (GENERATE_PTX) {
    size_t bin_sz;
    err = clGetProgramInfo (program, CL_PROGRAM_BINARY_SIZES,
                            sizeof(size_t), &bin_sz, NULL);

    // Read binary (PTX File) to memory buffer
    unsigned char* bin = (unsigned char*) malloc (bin_sz);
    err = clGetProgramInfo (program, CL_PROGRAM_BINARIES,
                            sizeof(unsigned char *), &bin, NULL);

    FILE* fp = fopen ("binary.ptx", "wb");
    fwrite (bin, sizeof(char), bin_sz, fp);
    fclose (fp);
    free (bin);
  }

  if (verbose) cout << "Create mem object for A matrix" << endl;
  createMemObjects (queue, &mem_A, (int) sizeof (T), ASize, hostMem_A);
  CL_CHECK_ERROR (err);

  if (verbose) cout << "Create mem object for B matrix" << endl;
  createMemObjects (queue, &mem_B, (int) sizeof (T), BSize, hostMem_B);
  CL_CHECK_ERROR (err);

  if (verbose) cout << "Create mem object for C matrix" << endl;
  createMemObjects (queue, &mem_C, (int) sizeof (T), CSize, hostMem_C);
  CL_CHECK_ERROR (err);

  if (verbose) cout << "Create mem object for R matrix" << endl;
	createMemObjects (queue, &mem_R, (int) sizeof (T), RSize, hostMem_R);
  CL_CHECK_ERROR (err);

	if (verbose) cout << "Start setting arguments for kernel 1" << endl;

  err = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *)&mem_A);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void *)&mem_B);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 2, sizeof (cl_mem), (void *)&mem_C);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 3, sizeof (cl_mem), (void *)&mem_R);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 4, sizeof (int), (void *)&A_height);
  CL_CHECK_ERROR (err);

	err = clSetKernelArg (kernel, 5, sizeof (int), (void *)&A_width);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 6, sizeof (int), (void *)&B_height);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 7, sizeof (int), (void *)&B_width);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 8, sizeof (int), (void *)&C_height);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 9, sizeof (int), (void *)&C_width);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 10, sizeof (int), (void *)&batch_size);
  CL_CHECK_ERROR (err);

	//err = clSetKernelArg (kernel1, 9, sizeof (int), (void *)batch_size);
  //CL_CHECK_ERROR (err);

	if (verbose) cout << "Setting arguments for kernel  is completed!" << endl;

    //err = clSetKernelArg (kernel2, 3, sizeof (int), (void *)batch_size);
    //CL_CHECK_ERROR (err);

  if (verbose) cout << "Setting arguments for kernel 2 is completed!" << endl;

  // Setup input memories for A, B, and C
  for (int i = 0; i < ASize; i++) {
    hostMem_A[i] = (T)(drand48()*5.0);
  }
  for (int i = 0; i < BSize; i++) {
    hostMem_B[i] = (T)(drand48()*5.0);
  }
  for (int i = 0; i < CSize; i++) {
    hostMem_C[i] = (T)(drand48()*5.0);
  }

  for (int pas = 0; pas < npasses; pas++) {
  	refillMemObject (queue, &mem_A, (int) sizeof (T), ASize, hostMem_A);
    refillMemObject (queue, &mem_B, (int) sizeof (T), BSize, hostMem_B);
    refillMemObject (queue, &mem_C, (int) sizeof (T), CSize, hostMem_C);

		cout << "Start setting arguments for kernel" << endl;

  err = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *)&mem_A);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void *)&mem_B);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 2, sizeof (cl_mem), (void *)&mem_C);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 3, sizeof (cl_mem), (void *)&mem_R);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 4, sizeof (int), (void *)&A_height);
  CL_CHECK_ERROR (err);

	err = clSetKernelArg (kernel, 5, sizeof (int), (void *)&A_width);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 6, sizeof (int), (void *)&B_height);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 7, sizeof (int), (void *)&B_width);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 8, sizeof (int), (void *)&C_height);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 9, sizeof (int), (void *)&C_width);
  CL_CHECK_ERROR (err);

  err = clSetKernelArg (kernel, 10, sizeof (int), (void *)&batch_size);
  CL_CHECK_ERROR (err);


		const size_t GWS1[3] = {A_height, B_width, batch_size};
    const size_t LWS1[3] = {8, 8, 1};


    const size_t GWS2[3] = {A_height, C_width, batch_size};
    const size_t LWS2[3] = {8, 8, 1};

		Event evKernel ("Kernel");

    clFinish (queue);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    if (targetDevice == TargetDevice::GPU) {
			err = clEnqueueNDRangeKernel (queue,
                                  	kernel,
                                  	3,
                                  	NULL,
                                  	GWS1,
                                  	LWS1,
                                  	0,
                                  	NULL,
                                  	&evKernel.CLEvent());
    } else {
      err = clEnqueueTask (queue, kernel, 0, NULL, &evKernel.CLEvent());
    }

		//clFinish (queue);
    //CL_CHECK_ERROR (err);
    //err = clFinish (queue);
    //CL_CHECK_ERROR (err);
    //err = clWaitForEvents (1, &evKernel1.CLEvent());
    //evKernel1.FillTimingInfo ();
    //clFinish (queue);
    //CL_CHECK_ERROR (err);
    //err =clFinish (queue);
    CL_CHECK_ERROR (err);
    err = clWaitForEvents (1, &evKernel.CLEvent());
    CL_CHECK_ERROR (err);
    evKernel.FillTimingInfo ();
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

		clFinish (queue);

		cout << "Reading back mem object" << endl;

    if (pas >= 2)
			//totalOffset += (duration_cast<nanoseconds>(t2-t1).count());
    	totalOffset += evKernel.StartEndRuntime ();

		readbackMemObject (queue, &mem_R, (int) sizeof (T), RSize, hostMem_R);

		cout << algorithm->getKernelLocation () << "-" << algorithm->getName() << "-" << batch_size << " " << (double)(duration_cast<nanoseconds>(t2-t1).count()) << endl;

		double gflops = (((A_height * B_width) * 2 * A_width) * batch_size + ((A_height * C_width) * 2 * B_width) * batch_size) / (double)(duration_cast<nanoseconds>(t2-t1).count());

    if (pas >= 2) {
    	resultDB.AddResult (string(algorithm->getKernelLocation ()) + string("-") + string(algorithm->getName()) + string("-") + string("b") + to_string(batch_size), string(), "GFLOPS", gflops );

    	//resultDB.AddResult (string(algorithm->getKernelLocation ()) + string("-") + string(algorithm->getName()) + string("-") + string("b") + to_string(batch_size) + string("-") + string("outRate"), string(), "Matrix/ms", batch_size / (((double)(duration_cast<nanoseconds>(t2-t1).count()) - this->offset) / 1000000.0));
    	resultDB.AddResult (string(algorithm->getKernelLocation ()) + string("-") + string(algorithm->getName()) + string("-") + string("b") + to_string(batch_size) + string("-") + string("outRate"), string(), "Matrix/ms", batch_size / ((evKernel.StartEndRuntime () - this->offset) / 1000000.0));
    	//resultDB.AddResult (string(algorithm->getKernelLocation ()) + string("-") + string(algorithm->getName()) + string("-") + string("b") + to_string(batch_size) + string("-") + string("outRate"), string(), "Matrix/ms", batch_size / (((double)(duration_cast<nanoseconds>(t2-t1).count())) / 1000000.0));
    }

    if (VERIFICATION) {
      algorithm->verifyMatrixPipeline (hostMem_A, hostMem_B, hostMem_C, verification_R,
                                       A_height, A_width,
                                       B_height, B_width,
                                       C_height, C_width,
                                       batch_size);

      if (verbose) cout << "Verification is done, now we get into comparison!" << endl;

      for (int element = 0; element < A_height*C_width*batch_size; element++) {
        float diff = verification_R[element] - hostMem_R[element];
        if (diff > 0.1 || diff < -0.1) {
          cout << "[ERROR] Verification Failed!" << endl;
          break;
        }
      }

     	cout << "Verification Passed!" << endl;
    }

    CL_CHECK_ERROR (err);
  }

   if (batch_size == 1)
   	this->offset = (totalOffset / (npasses-2)) * 0.88;

  cout << "offset is " << this->offset << endl;
  cout << endl;

  cout << "time to release" << endl;

  err = clReleaseKernel (kernel);
  CL_CHECK_ERROR (err);
  err = clReleaseMemObject (mem_A);
  CL_CHECK_ERROR (err);
  err = clReleaseMemObject (mem_B);
  CL_CHECK_ERROR (err);
  err = clReleaseMemObject (mem_C);
  CL_CHECK_ERROR (err);
  if (targetDevice == TargetDevice::GPU) {
		err = clReleaseMemObject (mem_I);
  	CL_CHECK_ERROR (err);
  }
  err = clReleaseMemObject (mem_R);
  CL_CHECK_ERROR (err);



  delete[] hostMem_A;
  delete[] hostMem_B;
  delete[] hostMem_C;
  delete[] hostMem_R;
  delete[] hostMem_I;
  delete[] verification_R;

}


template <class T>
string OpenCLEngine<T>::preparedVarDeclFormula (char *varDeclFormula, int depth) {

  string declFormula = string (varDeclFormula);
  char depthBuf[32];
  sprintf (depthBuf, "%d", depth);
  string depthSize = string (depthBuf);
	int pos = -1;
  while ((pos = declFormula.find ("$")) != -1)
    declFormula.replace (pos, 1, depthSize);

	if (VERBOSE)
    cout << "Var Decl Formula been prepared!" << endl;

  return declFormula;
}

template <class T>
string OpenCLEngine<T>::preparedVarDeclFormulaNonArray (char *varDeclFormula, int depth, bool lcdd) {

	string completeDeclFormula;

  if (lcdd == false) {

		// Check whether we have $ sign in the varDeclFormula
  	string declFormula = string (varDeclFormula);
  	int pos = -1;
  	if ((pos = declFormula.find ("$")) == -1)
    	depth = 1;

  	for (int i = 0; i < depth; i++) {
    	string declFormula = string (varDeclFormula);
    	char depthBuf[32];
    	sprintf (depthBuf, "%d", i);
    	string depthSize = string (depthBuf);
    	int pos = -1;
    	while ((pos = declFormula.find ("$")) != -1)
      	declFormula.replace (pos, 1, depthSize);

			if (i != (depth -1))
    		completeDeclFormula += (declFormula + ";\n");
    	else
				completeDeclFormula += (declFormula);

  	}
  } else {

    // Check whether we have $ sing in the varDeclFormula
    string declFormula = string (varDeclFormula);
    int pos = -1;
    //if ((pos = declFormula.find ("$")) == -1)
    //  exit (0);

    char depthBuf[32];
    sprintf (depthBuf, "%d", depth);
    string depthSize = string (depthBuf);
    pos = -1;
    while ((pos = declFormula.find("$")) != -1)
      declFormula.replace (pos, 1, depthSize);

   	completeDeclFormula = declFormula;

  }

  if (VERBOSE) cout << "[VERBOSE] Var Decl Formula been prepared!" << endl;

  return completeDeclFormula;
}

template <class T>
string OpenCLEngine<T>::prepareOriginalFormula (char *formula, int index, char *variable) {

	int pos = -1;

	string formulaStr = string (formula);
  char indexBuf[32];
  char indexBuf2[32];
  char indexBuf3[32];
  sprintf (indexBuf, "%d", index%PRIVATE_VECTOR_SIZE);
  sprintf (indexBuf2, "%d", (index-1)%PRIVATE_VECTOR_SIZE);
  if (index != 0)
  	sprintf (indexBuf3, "%d", (index-1));
  else
    sprintf (indexBuf3, "%c", 'i');

  string indexStr = string (indexBuf);
  string indexStr2 = string (indexBuf2);

  while ((pos = formulaStr.find ("$")) != (-1))
		formulaStr.replace (pos, 1, indexStr);

	pos = -1;
  formulaStr = string (formulaStr);
  while ((pos = formulaStr.find ("!")) != (-1))
    formulaStr.replace (pos, 1, indexBuf3);

  pos = -1;
  formulaStr = string (formulaStr);
  while ((pos = formulaStr.find ("#")) != (-1))
    formulaStr.replace (pos, 1, indexStr2);

  pos = -1;
  formulaStr = string (formulaStr);
  while ((pos = formulaStr.find("@")) != (-1))
    formulaStr.replace (pos, 1, string (variable));

	if (VERBOSE)
		cout << "[VERBOSE] Original Formula been prepared!";

  return formulaStr;

}

template <class T>
string OpenCLEngine<T>::preparedLocalMemoryInitialization (int depth, string dataType, char** origFormula) {

  // Preparing the intialization of local memory
	string return_statement;
	return_statement += (string("\t") + string("__local ")+ dataType + string(" localRands[") + to_string(depth) + string("];\n"));
  return_statement += (string("\t") + string("int depth = ") + to_string(depth) + string(";\n\n"));
  return_statement += (string("\t") + string("int gid = get_global_id(0);\n"));
  return_statement += (string("\t") + string("int lid = get_local_id(0);\n"));
  return_statement += (string("\t") + string("int localWorkSize = get_local_size(0);\n"));
  return_statement += (string("\t") + string("int workItemCopyPortion = depth / localWorkSize;\n\n"));
  return_statement += (string("\t") + string("event_t event = async_work_group_copy (localRands, &(rands[lid * workItemCopyPortion]), (depth - lid*workItemCopyPortion < workItemCopyPortion) ? (depth - lid*workItemCopyPortion) : workItemCopyPortion, 0);\n"));
	return_statement += (string("\t") + string("wait_group_events(1, &event);\n"));
	if (VERBOSE)
    cout << "[VERBOSE] return statement been prepared successfully!" << endl;
  // replace rands in original formula with localRands, if it does exists
	int pos = -1;
  string origFormulaStr = string(*origFormula);
  while ((pos = origFormulaStr.find ("rands")) != -1)
    origFormulaStr.replace (pos, 5, "localRands");

  int length = origFormulaStr.size();
  char* newOrigFormula = (char *) malloc (50 * sizeof(char));
  memcpy (newOrigFormula, origFormulaStr.c_str(), length);

  newOrigFormula[length] = '\0';
	*origFormula = newOrigFormula;
	if (VERBOSE)
    cout << "[VERBOSE] rands been replaced with localRands" << endl;


	return return_statement;

}

template <class T>
string OpenCLEngine<T>::prepareReturnOpCode (int streamSize, string returnOpCode) {

  string finalReturnOpCode = (returnOpCode + " = ");

  for (int i = 0; i < streamSize; i++){
    if (i == 10)
      finalReturnOpCode += (string("temp.s") + string("A"));
  	else if (i == 11)
      finalReturnOpCode += (string("temp.s") + string("B"));
    else if (i == 12)
			finalReturnOpCode += (string("temp.s") + string("C"));
    else if (i == 13)
      finalReturnOpCode += (string("temp.s") + string("D"));
    else if (i == 14)
      finalReturnOpCode += (string("temp.s") + string("E"));
    else if (i == 15)
      finalReturnOpCode += (string("temp.s") + string("F"));
		else
			finalReturnOpCode += (string("temp.s") + to_string(i));

    if (i != streamSize-1)
      finalReturnOpCode += string(" + ");
  }

  return finalReturnOpCode;
}

