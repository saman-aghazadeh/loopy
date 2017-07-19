#ifndef CUDAENGINE_H_
#define CUDAENGINE_H_

#include "AlgorithmType.h"
#include "CudaKernelExecution.h"
#include "NumberGenerator.h"

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
using namespace std;

#define PRIVATE_VECTOR_SIZE 5

struct _cuda_info {
	char name[100] = {'\0'};
  int num_workitems;
  int flops;
};


string replace_str(string& str, const string& from, const string& to)
{

	string newStr(str);

  if(from.empty())
    return string();
  string wsRet;
  wsRet.reserve(newStr.length());
  size_t start_pos = 0, pos;
  while((pos = newStr.find(from, start_pos)) != string::npos) {
    wsRet += newStr.substr(start_pos, pos - start_pos);
    wsRet += to;
    pos += from.length();
    start_pos = pos;
  }
  wsRet += newStr.substr(start_pos);
  newStr.swap(wsRet); // faster than str = wsRet;

  return newStr;
}

template<class T>
class CudaEngine {
public:

  CudaEngine (int executionMode, int targetDevice,
              struct _algorithm_type *tests) {
    this->executionMode = executionMode;
    this->targetDevice = targetDevice;
    this->tests = tests;
    init_kernel_map ();
  }
 	~CudaEngine () {}

  void executionCUDA (ResultDatabase &resultDB,
                      OptionParser &op,
                      char* precision);

	bool createCUDAObjects (T** memObject,
                          int singleElementSize,
                          const int memSize,
                          T* data);

	bool refillCUDAObject (T** memObject,
                        int singleElementSize,
                        const int memSize,
                        T* data);

  void generateCUDAMetas ();

  void generateSingleCUDAMeta (struct _algorithm_type &temp,
                               struct _cuda_info &info);

	void print_benchmark ();

	void generateCUDAs ();

  void generateSingleCUDACode (ostringstream &oss,
                               struct _algorithm_type &temp,
                               struct _cuda_info &info);

  string prepareVarDeclFormulaNonArray (char *varDeclFormula, int depth, bool lcdd);

	string prepareOriginalFormula (int streamSize, char *formula, int index, char *variable);

	string prepareReturnOpCode (int streamSize, string returnOpCode);

	string prepareVarInitFormula (int streamSize, string varInitFormula, string variable);

  void insertTab (ostringstream &oss, int numTabs);

private:

  int executionMode;
  int targetDevice;
  struct _algorithm_type *tests;
  vector<_cuda_info> cuda_metas;
	const bool VERBOSE = true;
  const bool VERIFICATION = true;

  int npasses = 5;
  // Path to folder where the generated CUDA kernels will resude. Change it effectively
  std::string cuda_built_kernels_folder = "/home/users/saman/shoc/src/cuda/level3/Algs";
};

template <class T>
void CudaEngine<T>::executionCUDA (ResultDatabase &resultDB,
                                   OptionParser &op,
                                   char *precision) {
  int err;
	char sizeStr[128];

	T* hostMem_data;
  T* hostMem2;
  T* hostMem_rands;
  T* verification_data;

	T* mem_data;
  T* mem_rands;

  if (VERBOSE) cout << "Start CUDA execution!" << endl;

  int aIdx = 0;
  while ((tests != 0) && (tests[aIdx].name != 0)) {
    if (strcmp (tests[aIdx].varType, precision)) {
      aIdx++;
      continue;
    }

    struct _algorithm_type alg = tests[aIdx];
    struct _cuda_info meta = cuda_metas[aIdx];

    if (VERBOSE) cout << "[Retrieved CUDA Meta] ";
    if (VERBOSE) cout << "name = " << meta.name << endl;

		int halfNumFloatsMax = alg.loopsLengths[0];
    int numFloatsMax = halfNumFloatsMax;

    if (VERBOSE) cout << "numFloatsMax=" << numFloatsMax << ", depth=" << alg.loopsDepth[0] << endl;

    hostMem_data = new T[numFloatsMax];
    hostMem2 = new T[numFloatsMax];
    hostMem_rands = new T[alg.loopsDepth[0]];
    verification_data = new T[numFloatsMax];

		if (VERBOSE) cout << "hostMem_data, hostMem2, and hostMem_rands are created successfully!" << endl;

    if (!VERIFICATION) {
			for (int length = 0; length < alg.loopsDepth[0]; length++) {
        hostMem_rands[length] = (float) rand() / ((float)RAND_MAX/2);
      }
    } else {
      for (int length = 0; length < alg.loopsDepth[0]; length++) {
        hostMem_rands[length] = (float) 1.1;
      }
    }

		// Time to allocate device global data
		if (!createMemObjects (&mem_data, (int) sizeof (T), numFloatsMax, hostMem_data)) {
      cerr << "Error allocating device memory!" << endl;
      exit(0);
    }

    if (!createMemObjects (&mem_rands, (int) sizeof (T), alg.loopsDepth[0], hostMem_rands)) {
      cerr << "Error allocating device memory!" << endl;
      exit(0);
    }

		int random = rand() % PRIVATE_VECTOR_SIZE;
		int rand_max = RAND_MAX;

    // Set up input memory for data, first half = second half
    int numFloat = numFloatsMax;
    if (!VERIFICATION) {
      for (int j = 0; j < numFloatsMax/2; j++) {
        hostMem_data[j] = hostMem_data[numFloat - j - 1] = (T)(drand48() * 5.0);
      }
    } else {
      for (int j = 0; j < numFloatsMax; j++) {
        hostMem_data[j] = 0.4;
      }
    }

		if (!refillCUDAObject (&mem_data, (int) sizeof (T), numFloatsMax, hostMem_data)) {
    	cerr << "Error refilling device memory!" << endl;
      exit(0);
    }

    if (!refillCUDAObject (&mem_rands, (int) sizeof (T), alg.loopsDepth[0], hostMem_rands)) {
      cerr << "Error refilling device memory!" << endl;
      exit(0);
    }

		int numBlocks;
		int threadsPerBlock;

		for (int wsBegin = alg.localWorkSizeMin; wsBegin <= alg.localWorkSizeMax; wsBegin *= alg.localWorkSizeStride) {

     	if (alg.loopCarriedDataDependency == false) {
				threadsPerBlock = wsBegin;
				numBlocks = numFloatsMax / wsBegin;
      } else if (alg.loopCarriedDataDependency == true) {
        threadsPerBlock = 1;
        numBlocks = 1;
      }
			char lwsString[10] = {'\0'};

      for (int pas = 0; pas < npasses; ++pas) {
				if (!refillCUDAObject (&mem_data, (int) sizeof (T), numFloatsMax, hostMem_data)) {
          cerr << "Error refilling device memory!" << endl;
          exit (0);
        }

        if (!refillCUDAObject (&mem_rands, (int) sizeof (T), alg.loopsDepth[0], hostMem_rands)) {
          cerr << "Error refilling device memory!" << endl;
          exit (0);
        }

        cudaEvent_t start, stop;
        float t = 0.0f;
        cudaEventCreate (&start);
        cudaEventCreate (&stop);
        cudaEventRecord (start, 0);
				kernelMaps[string(tests[aIdx].name)](mem_data, mem_rands, random, rand_max, numBlocks, threadsPerBlock);
        cudaEventRecord (stop, 0);
        cudaEventSynchronize (stop);
        CHECK_CUDA_ERROR ();
        cudaEventElapsedTime (&t, start, stop);
        t /= 1.e3;

				double flopCount = (double) numFloatsMax *
          													meta.flops *
          													alg.loopsDepth[0] *
          													alg.vectorSize;

        double gflops = flopCount / (double)(t * 1.e9);

        sprintf (sizeStr, "Size: %07d", numFloatsMax);
        sprintf (lwsString, "%d", wsBegin);
        resultDB.AddResult (string (alg.name) + string("-lws") + string (lwsString) + string ("-") + string(precision), sizeStr, "GFLOPS", gflops);

        cudaEventDestroy (start);
        cudaEventDestroy (stop);
      }
    }

		cudaFree ((void *) mem_data);
		cudaFree ((void *) mem_rands);

    delete hostMem_data;
    delete hostMem2;
    delete hostMem_rands;
    delete verification_data;

  }
}

template <class T>
void CudaEngine<T>::generateCUDAMetas () {

  int aIdx = 0;
  while ((tests != 0) && (tests[aIdx].name != 0)) {
    struct _cuda_info test_cuda_info;
    struct _algorithm_type temp = tests[aIdx];
    generateSingleCUDAMeta (temp, test_cuda_info);
    cuda_metas.push_back (test_cuda_info);
    aIdx++;
  }
}

template <class T>
void CudaEngine<T>::generateSingleCUDAMeta (_algorithm_type &test, _cuda_info &info) {

  memcpy (info.name, (char *)test.name, strlen ((char *)test.name));
  info.num_workitems = test.loopsLengths[0];
  info.flops = test.flopCount;
}

template <class T>
bool CudaEngine<T>::createCUDAObjects (T** memObject,
                        int singleElementSize,
                        const int memSize,
                        T* data) {
	cudaMalloc ((void**)memObject, singleElementSize * memSize);
  if (cudaGetLastError() != cudaSuccess) {
    cerr << "Error creating memory objects. " << endl;
    return false;
  }

  cudaEvent_t start, stop;
	cudaEventCreate (&start);
  cudaEventCreate (&stop);

  CHECK_CUDA_ERROR ();

	cudaEventRecord (start, 0);
	cudaMemcpy (*memObject, data, memSize * singleElementSize, cudaMemcpyHostToDevice);
  cudaEventRecord (stop, 0);
  cudaEventSynchronize (stop);

  float t = 0; cudaEventElapsedTime (&t, start, stop);
  if (VERBOSE)
    cout << "Size " << memSize*singleElementSize << "KB took " << t << "ms." << endl;

  return true;

}

template <class T>
bool CudaEngine<T>::refillCUDAObject (T** memObject,
                                   int singleElementSize,
                                   const int memSize,
                                   T* data) {

	cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  CHECK_CUDA_ERROR ();

  cudaEventRecord (start, 0);
  cudaMemcpy (*memObject, data, memSize * singleElementSize, cudaMemcpyHostToDevice);
  cudaEventRecord (stop, 0);
  cudaEventSynchronize (stop);

  float t = 0; cudaEventElapsedTime (&t, start, stop);
  if (VERBOSE)
    cout << "Size " << memSize*singleElementSize << "KB took " << t << "ms." << endl;

  return true;

}

template <class T>
void CudaEngine<T>::print_benchmark () {

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
void CudaEngine<T>::generateCUDAs () {

	int aIdx = 0;
  while ((tests != 0) && (tests[aIdx].name != 0)) {
    ostringstream oss;
    struct _cuda_info test_cuda_info;
    struct _algorithm_type temp = tests[aIdx];
    generateSingleCUDACode (oss, temp, test_cuda_info);
    aIdx++;
  }

	// generate cuda kernel execution .h file, which
  // hold reference to all cuda kernel files and
  // as a map from string to function pointer

	ostringstream oss;
  ofstream codeDump;
	string dumpFileName = cuda_built_kernels_folder + "/CudaKernelExecution.h";
  codeDump.open (dumpFileName.c_str());
  if (!codeDump.is_open()) {
    cout << "[ERROR] Dump File cannot be created or opened!" << endl;
  }

	oss << "#ifndef CUDAKERNELEXECUTION_H_" << endl;
  oss << "#define CUDAKERNELEXECUTION_H_" << endl;
	oss << endl;
	oss << "#include <map>" << endl;
  oss << "#include <string>" << endl;
  oss << "using namespace std;" << endl;
	oss << endl;
  aIdx = 0;
  while ((tests != 0) && (tests[aIdx].name != 0)) {
    oss << "#include \"" << tests[aIdx].name << ".h\"" << endl;
    aIdx++;
  }
	oss << endl;
	oss << "map<string, void(*)(float*, float*, int, int, int, int)> kernelMaps;" << endl;

  oss << "void init_kernel_map () {" << endl;
  aIdx = 0;
  while ((tests != 0) && (tests[aIdx].name != 0)) {
    insertTab (oss, 1); oss << "kernelMaps[string(\"" << tests[aIdx].name << "\")] = " << tests[aIdx].name << "_wrapper;" << endl;
    aIdx++;
  }
  oss << "}" << endl;
  oss << endl;
  oss << "#endif // CUDAKERNELEXECUTION_h_" << endl;

  codeDump << oss.str();
	codeDump.close();

}

template <class T>
void CudaEngine<T>::generateSingleCUDACode (ostringstream &oss, struct _algorithm_type &test, struct _cuda_info &info) {

	if (VERBOSE) cout << "Generate Single CUDA Code for " << test.name << endl;

	ofstream codeDump;
  string dumpFileName = cuda_built_kernels_folder + "/" + test.name + ".h";
  codeDump.open (dumpFileName.c_str());
  if (!codeDump.is_open()) {
    cout << "[ERROR] Dump File cannot be created or opened!" << endl;
  }

	if (test.loopCarriedDataDependency == false) {

		oss << "#ifndef " << test.name << "_H_" << endl;
    oss << "#define " << test.name << "_H_" << endl;
    oss << endl;
    oss << "__global__ void " << test.name << "( " << test.varType << " *data, " << test.varType << " *rands, int index, int rand_max){" << endl;
		string declFormula = prepareVarDeclFormulaNonArray ((char *) test.varDeclFormula, PRIVATE_VECTOR_SIZE, false);
    insertTab (oss, 1); oss << declFormula << ";" << endl;
		if (!test.doLocalMemory) {
      insertTab (oss, 1); oss << "int gid = blockIdx.x * blockDim.x + threadIdx.x;" << endl;
    } else {
      insertTab (oss, 1); oss << "int gid = blockIdx.x * blockDim.x + threadIdx.x;" << endl;
    }

    oss << endl;
    string preparedVarInitFormula = prepareVarInitFormula (test.vectorSize, test.varInitFormula, test.variable);
    insertTab (oss, 1); oss << preparedVarInitFormula << ";" << endl;
    if (VERBOSE) cout << "[VERBOSE] Init formula been inserted successfully!" << endl;

    if (test.doManualUnroll == true) {
      NumberGenerator numberGenerator (test.loopsDepth[0]);
      if (VERBOSE) cout << "[VERBOSE] Manually Unrolling is True!" << endl;
      for (int i = 1; i < test.loopsDepth[0]; i++) {
        string origFormula;
        if (test.randomAccessType == RandomAccessType::SEQUENTIAL)
          origFormula = prepareOriginalFormula (test.vectorSize, (char *) test.formula, numberGenerator.getNextSequential() + 1, (char *) test.variable);
        else if (test.randomAccessType == RandomAccessType::RANDOM)
          origFormula = prepareOriginalFormula (test.vectorSize, (char *) test.formula, numberGenerator.getNextRandom() + 1, (char *) test.variable);
        insertTab (oss, 1); oss << origFormula << ";" << endl;
      }
    }  else {
			if (VERBOSE) cout << "[VERBOSE] Manually Unrolling is False!" << endl;
			if (test.randomAccessType != RandomAccessType::SEQUENTIAL)
      	cout << "[WARNING] random access type cannot be generated for automatically unrolled kernels!" << endl
        	   << "\t Please re-design your \"test.h\" file!" << endl;
    	if (test.unrollFactor != 0) {
      	insertTab (oss, 1); oss << "#pragma unroll " << test.unrollFactor << endl;
    	}
    	insertTab (oss, 1); oss << "for (int i = 0; i < " << test.loopsDepth[0] << "; i++){" << endl;
    	string origFormula = prepareOriginalFormula (test.vectorSize, (char *)test.formula, 0, (char *) test.variable);
    	insertTab (oss, 2); oss << origFormula << ";" << endl;
    	insertTab (oss, 1); oss << "}" << endl;
  	}

  	int pos = -1;
  	char returnBuf[32];

  	string returnOpCode = prepareReturnOpCode (test.vectorSize, test.returnFormula);

  	if (VERBOSE) cout << "[VERBOSE] Return Op Code been prepared!" << endl;

  	insertTab (oss, 1); oss << returnOpCode << ";" << endl;

  	oss << endl;
  	oss << "}" << endl;

  	if (VERBOSE) cout << "[VERBOSE] CUDA Code for " << test.name << " been created Successfully!" << endl;
  } else if (test.loopCarriedDataDependency == true) {

  }

  // It's time to create the wrapper in the .h file,
  // which calls the kernel

	oss << endl << endl;

  oss << "void " << test.name << "_wrapper (" << test.varType << " *data, " << test.varType << " *rands, int index, int rand_max, int numBlocks, int threadPerBlock) {" << endl;

  insertTab (oss, 1); oss << test.name << "<<<numBlocks, threadPerBlock>>> (data, rands, index, rand_max);" << endl;

  oss << "}" << endl;
	oss << endl;
  oss << "#endif " << endl;

  codeDump << oss.str();
  codeDump.close();

}

template <class T>
void CudaEngine<T>::insertTab (ostringstream &oss, int numTabs) {
  for (int i = 0; i < numTabs; i++) {
    oss << "\t";
  }
}

template <class T>
string CudaEngine<T>::prepareVarDeclFormulaNonArray (char *varDeclFormula, int depth, bool lcdd) {

	string completeDeclFormula;

  if (lcdd == false) {

    // Check whether we have $ sing in the varDeclFormula
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

      if (i != (depth - 1))
        completeDeclFormula += (declFormula + ";\n");
      else
        completeDeclFormula += (declFormula);
    }

  } else {

    // Check whether we have $ sign in the varDeclFormula
    string declFormula = string (varDeclFormula);
    int pos = -1;

    char depthBuf[32];
    sprintf (depthBuf, "%d", depth);
    string depthSize = string (depthBuf);
    while ((pos = declFormula.find("$")) != -1)
      declFormula.replace (pos, 1, depthSize);

    completeDeclFormula = declFormula;
  }

  if (VERBOSE) cout << "[VERBOSE] var decl formula been prepared!" << endl;

  return completeDeclFormula;

}

template <class T>
string CudaEngine<T>::prepareOriginalFormula (int  streamSize, char *formula, int index, char* variable) {

	int pos = -1;

  string formulaStr = string (formula);
	char indexBuf[32];
  char indexBuf2[32];
  char indexBuf3[32];

  sprintf (indexBuf, "%d", index % PRIVATE_VECTOR_SIZE);
  sprintf (indexBuf2, "%d", (index-1) % PRIVATE_VECTOR_SIZE);
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
		cout << "[VERBOSE] Original Formula been prepared!" << endl;

	// Now it's time to replace every variable in the formula
  // with the equivalent with index included

	string variable_string (variable);
	string returnFormulaStr;

  for (int i = 0; i < streamSize; i++) {
    string index;
    if (i == 0) index = "x";
    else if (i == 1) index = "y";
    else if (i == 2) index = "z";
    else if (i == 3) index = "w";

    if (i != 0) returnFormulaStr += "\t\t";
    //int varIndex = -1;
    //int prevVarIndex = 0;
   	//while ((varIndex = formulaStr.find (variable_string, prevVarIndex)) != -1) {
    //returnFormulaStr += (formulaStr.substr(prevVarIndex, varIndex) + variable_string + "." + index);
    //prevVarIndex = varIndex + variable_string.length();
    //}

    returnFormulaStr += replace_str (formulaStr, variable_string, variable_string + "." + index);
    returnFormulaStr += ";";

    //returnFormulaStr += (formulaStr.substr (prevVarIndex) + ";");
		if (i != streamSize-1) returnFormulaStr += "\n";

  }

  return returnFormulaStr;

}

template <class T>
string CudaEngine<T>::prepareReturnOpCode (int streamSize, string returnOpCode){

  string finalReturnOpCode = (returnOpCode + " = ");

  for (int i = 0; i < streamSize; i++){
    if (i == 0)
      finalReturnOpCode += (string("temp.") + string("x"));
    else if (i == 1)
      finalReturnOpCode += (string("temp.") + string("y"));
    else if (i == 2)
      finalReturnOpCode += (string("temp.") + string("z"));
    else if (i == 3)
      finalReturnOpCode += (string("temp.") + string("w"));

    if (i != streamSize-1)
      finalReturnOpCode += string(" + ");
  }

  return finalReturnOpCode;

}

template <class T>
string CudaEngine<T>::prepareVarInitFormula (int streamSize, string varInitFormula, string variable) {

	string finalVarInitFormula;

  for (int i = 0; i < streamSize; i++) {
		int varIndex = varInitFormula.find (variable);
    if (varIndex == -1) {
      cerr << "Cannot find " << variable << " in " << varInitFormula << endl;
      return varInitFormula;
    }

   	string index;
    if (i == 0) index = "x";
    else if (i == 1) index = "y";
    else if (i == 2) index = "z";
    else if (i == 3) index = "w";

		if (i != 0) finalVarInitFormula += "\t";

		finalVarInitFormula += (variable + "." + index + varInitFormula.substr (varIndex + variable.length()) + ";");

    if (i != streamSize-1) finalVarInitFormula += "\n";
  }

	return finalVarInitFormula;

}

#endif // CUDAENGINE_H_

