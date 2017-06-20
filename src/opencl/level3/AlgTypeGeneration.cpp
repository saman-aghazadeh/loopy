#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>
#include <map>
#include <time.h>
#include <stdlib.h>

#include "ResultDatabase.h"
#include "OptionParser.h"
#include "ProgressBar.h"
#include "Event.h"


using namespace std;

struct _meta_algorithm_type {
	vector<int> streamSizes;
  int loopLevels;
  vector<int> loopIterSizes;
  vector<int> loopDepthSizes;
  bool loopCarriedDataDependency;
  vector<int> loopCarriedDDLengths;
  const char* variable;
  vector<string> varTypes;
  int halfBufSizeMin;
  int halfBufSizeMax;
  int halfBufSizeStride;

  int localWorkSizeMin;
  int localWorkSizeMax;
  int localWorkSizeStride;

  vector<map<string, string> > formulas;
  vector<bool> manualUnroll;
  vector<int> unrollFactor;
  vector<bool> localMemory;
};

struct _meta_algorithm_type meta_tests[] = {
  {vector<int>({16}), 1, vector<int>({1048576}), vector<int>({32,64,128, 256, 512}), false, vector<int>(), "temp", vector<string>({string("float")}), 1024, 1024, 2, 256, 256, 2, vector<map<string, string>>{map<string, string>{{"varDeclFormula", "@@$$ temp"}, {"varInitFormula", "temp = data[gid]"}, {"returnFormula", "data[gid] = temp.s0 + temp.s1 + temp.s2+ temp.s3 + temp.s4 + temp.s5 + temp.s6 + temp.s7 + temp.s8 + temp.s9 + temp.sA + temp.sB + temp.sC + temp.sD + temp.sE + temp.sF"}, {"formula", "@ = (@@) rands[!] * @"}}}, vector<bool>({false}), vector<int>({0, 8, 16, 32}), vector<bool>{false, true}},
  {vector<int>({16}), 1, vector<int>({1048576}), vector<int>({32,64,128, 256, 512}), false, vector<int>(), "temp", vector<string>({string("float")}), 1024, 1024, 2, 256, 256, 2, vector<map<string, string>>{map<string, string>{{"varDeclFormula", "@@$$ temp"}, {"varInitFormula", "temp = data[gid]"}, {"returnFormula", "data[gid] = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 + temp.s6 + temp.s7 + temp.s8 + temp.s9 + temp.sA + temp.sB + temp.sC + temp.sD + temp.sE + temp.sF"}, {"formula", "@ = (@@) rands[!] * @"}}}, vector<bool>({true}), vector<int>({0}), vector<bool>{false, true}},
  {vector<int>(), 0, vector<int>(), vector<int>(), 0, vector<int>(), 0, vector<string>(), 0, 0, 0, 0, 0, 0, vector<map<string, string>>()}
};

vector<map<string, string> > formulas;

string fileName = "../common/tests.h";

void addBenchmarkSpecOptions (OptionParser &op) {

}

void RunBenchmark (cl_device_id id,
                   cl_context ctx,
                   cl_command_queue queue,
                   ResultDatabase &resultDB,
                   OptionParser &op) {

	int aIdx = 0;
  ofstream headerFile;

  headerFile.open (fileName);
		headerFile << "#ifndef _TESTS_H_" << endl;
  	headerFile << "#define _TESTS_H_" << endl;
		headerFile << endl;
  	headerFile << "#include <vector>" << endl;
  	headerFile << "#include <map>" << endl;
  	headerFile << "#include <string>" << endl;
		headerFile << "using namespace std;" << endl;
		headerFile << endl << endl;

  	// Generating the structure _algorithm_type
		headerFile << "struct _algorithm_type {" << endl;
  	headerFile << "\t" << "const char* name;" << endl;
  	headerFile << "\t" << "int vectorSize;" << endl;
		headerFile << "\t" << "int numLoops;" << endl;
 		headerFile << "\t" << "vector<int> loopsLengths;" << endl;
  	headerFile << "\t" << "vector<int> loopsDepth;" << endl;
  	headerFile << "\t" << "bool loopCarriedDataDependency;" << endl;
  	headerFile << "\t" << "vector<int> loopCarriedDDLengths;" << endl;
  	headerFile << "\t" << "const char* variable;" << endl;
  	headerFile << "\t" << "const char* varDeclFormula;" << endl;
  	headerFile << "\t" << "const char* varInitFormula;" << endl;
	  headerFile << "\t" << "const char* returnFormula;" << endl;
	  headerFile << "\t" << "const char* formula;" << endl;
	  headerFile << "\t" << "int halfBufSizeMin;" << endl;
	  headerFile << "\t" << "int halfBufSizeMax;" << endl;
	  headerFile << "\t" << "int halfBufSizeStride;" << endl;
		headerFile << "\t" << "int localWorkSizeMin;" << endl;
 	  headerFile << "\t" << "int localWorkSizeMax;" << endl;
	  headerFile << "\t" << "int localWorkSizeStride;" << endl;
	  headerFile << "\t" << "int flopCount;" << endl;
  	headerFile << "\t" << "const char* varType;" << endl;
    headerFile << "\t" << "bool doManualUnroll;" << endl;
    headerFile << "\t" << "bool doLocalMemory;" << endl;
    headerFile << "\t" << "int unrollFactor;" << endl;
  	headerFile << "};" << endl << endl;

		headerFile << "struct _algorithm_type tests[] = {" << endl;


    while ((meta_tests != 0) && (meta_tests[aIdx].variable != 0)) {
		cout << "Meta test #" << aIdx << endl;

		vector<map<string, string>> formulas = meta_tests[aIdx].formulas;

    vector<int> streamSizes = meta_tests[aIdx].streamSizes;
    int loopLevels = meta_tests[aIdx].loopLevels;
    vector<int> loopIterSizes = meta_tests[aIdx].loopIterSizes;
    vector<int> loopDepthSizes = meta_tests[aIdx].loopDepthSizes;
    bool loopCarriedDataDependency = meta_tests[aIdx].loopCarriedDataDependency;
    vector<int> loopCarriedDDLengths = meta_tests[aIdx].loopCarriedDDLengths;
    const char* variable = meta_tests[aIdx].variable;
    vector<string> varTypes = meta_tests[aIdx].varTypes;
    int halfBufSizeMin = meta_tests[aIdx].halfBufSizeMin;
    int halfBufSizeMax = meta_tests[aIdx].halfBufSizeMax;
    int halfBufSizeStride = meta_tests[aIdx].halfBufSizeStride;
		int localWorkSizeMin = meta_tests[aIdx].localWorkSizeMin;
    int localWorkSizeMax = meta_tests[aIdx].localWorkSizeMax;
    int localWorkSizeStride = meta_tests[aIdx].localWorkSizeStride;
    vector<bool> manualUnroll = meta_tests[aIdx].manualUnroll;
		vector<int> unrollFactor = meta_tests[aIdx].unrollFactor;
		vector<bool> localMemory = meta_tests[aIdx].localMemory;

  	for (int formulasNum = 0; formulasNum < formulas.size(); formulasNum++){
			for (int varTypesNum = 0; varTypesNum < varTypes.size(); varTypesNum++) {
				for (int streamSizesNum = 0; streamSizesNum < streamSizes.size(); streamSizesNum++) {
      		for (int loopIterSizesNum = 0; loopIterSizesNum < loopIterSizes.size(); loopIterSizesNum++) {
        		for (int loopDepthSizesNum = 0; loopDepthSizesNum < loopDepthSizes.size(); loopDepthSizesNum++) {
              for (int manualUnrollMode = 0; manualUnrollMode < manualUnroll.size(); manualUnrollMode++) {
                for (int unrollFactorSize = 0; unrollFactorSize < unrollFactor.size(); unrollFactorSize++) {
                  for (int localMemoryMode = 0; localMemoryMode < localMemory.size(); localMemoryMode++) {
          					headerFile << "\t" << "{\"TestS" << streamSizes[streamSizesNum] << "" << "V" << varTypes[varTypesNum] << "" << "I" << loopIterSizes[loopIterSizesNum] << "" << "D" << loopDepthSizes[loopDepthSizesNum] << "Form" << (formulasNum+1) << "MUnrol" << manualUnroll[manualUnrollMode] << "U" << unrollFactor[unrollFactorSize] << "LM" << localMemory[localMemoryMode] << "\", ";
          					headerFile << streamSizes[streamSizesNum] << ", ";
          					headerFile << "1, ";
          					headerFile << "vector<int>({" << loopIterSizes[loopIterSizesNum] << "}), ";
          					headerFile << "vector<int>({" << loopDepthSizes[loopDepthSizesNum] << "}), ";
          					headerFile << "false, ";
          					headerFile << "vector<int>({0}), ";
          					headerFile << "\"" << variable << "\", ";

										string varDeclFormula = formulas[formulasNum]["varDeclFormula"];
            				int pos = -1;
            				while ((pos = varDeclFormula.find("@@")) != -1)
              				varDeclFormula.replace (pos, 2, string(varTypes[varTypesNum]));

            				pos = -1;
            				while ((pos = varDeclFormula.find("$$")) != -1)
              				varDeclFormula.replace (pos, 2, to_string(streamSizes.data()[streamSizesNum]));

            				headerFile << "\"" << varDeclFormula << "\", ";
            				headerFile << "\"" << formulas[formulasNum]["varInitFormula"] << "\", ";
            				headerFile << "\"" << formulas[formulasNum]["returnFormula"] << "\", ";
            				pos = -1;
										string formula = formulas[formulasNum]["formula"];
            				while ((pos = formula.find("@@")) != -1)
              				formula.replace (pos, 2, string(varTypes.data()[varTypesNum]));
            				headerFile << "\"" << formula << "\", ";
            				headerFile << halfBufSizeMin << ", ";
            				headerFile << halfBufSizeMax << ", ";
            				headerFile << halfBufSizeStride << ", ";

            				headerFile << localWorkSizeMin << ", ";
 										headerFile << localWorkSizeMax << ", ";
            				headerFile << localWorkSizeStride << ", ";
            				headerFile << "1, ";
            				headerFile <<"\"" << varTypes[varTypesNum] << "\", ";
              			headerFile << manualUnroll[manualUnrollMode] << ",";
              			headerFile << localMemory[localMemoryMode] << ",";
										headerFile << unrollFactor[unrollFactorSize] << "}," << endl;
                  }
                }
              }
						}
        	}
     		}
    	}
  	}

    aIdx++;
	}

	headerFile << "\t" << "{0, 0, 0, vector<int>(), vector<int>(), 0, vector<int>(), 0, 0, 0, 0, 0, 0, 0, 0, 0, false, false, 0}" << endl;
  headerFile << "};" << endl;
	headerFile << endl;
	headerFile << "#endif" << endl;
	headerFile.close();
}
