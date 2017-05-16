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

vector<int> streamSizes;
int loopLevels = 1;
vector<int> loopIterSizes = {1048576};
vector<int> loopDepthSizes = {64, 128, 256, 512, 1024};
bool loopCarriedDataDependency = false;
vector<int> loopCarriedDDLengths;
const char* variable = "temp";
vector<string> varTypes = {string("float"), string("double")};
int halfBufSizeMin = 1024;
int halfBufSizeMax = 1024;
int halfBufSizeStride = 2;

int localWorkSizeMin = 32;
int localWorkSizeMax = 128;
int localWorkSizeStride = 2;

vector<map<string, string> > formulas;

string fileName = "../common/tests.h";

void addBenchmarkSpecOptions (OptionParser &op) {

}

void RunBenchmark (cl_device_id id,
                   cl_context ctx,
                   cl_command_queue queue,
                   ResultDatabase &resultDB,
                   OptionParser &op) {



 	streamSizes.push_back(2); streamSizes.push_back(4);
  streamSizes.push_back(8); streamSizes.push_back(16);
	formulas.push_back (map<string, string>());
	formulas.push_back (map<string, string>());

  formulas[0]["varDeclFormula"] = "@@$$ temp";
  formulas[0]["varInitFormula"] = "temp = data[gid]";
  formulas[0]["returnFormula"] = "data[gid] = temp.s0";
  formulas[0]["formula"] = "@ = (@@) rands[!] * @";

	formulas[1]["varDeclFormula"] = "@@$$ temp$";
  formulas[1]["varInitFormula"] = "temp0 = data[gid]";
  formulas[1]["returnFormula"] = "data[gid] = temp0.s0";
  formulas[1]["formula"] = "@$ = (@@) rands[!] * @#";


	ofstream headerFile;
  headerFile.open (fileName);

	headerFile << "#ifndef _TESTS_H_" << endl;
  headerFile << "#define _TESTS_H_" << endl;
	headerFile << endl;
  headerFile << "#include <vector>" << endl;
  headerFile << "#include <map>" << endl;
  headerFile << "#include <string>" << endl;
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
  headerFile << "};" << endl << endl;;

	headerFile << "struct _algorithm_type tests[] = {" << endl;


  for (int formulasNum = 0; formulasNum < formulas.size(); formulasNum++){
		for (int varTypesNum = 0; varTypesNum < varTypes.size(); varTypesNum++) {
			for (int streamSizesNum = 0; streamSizesNum < streamSizes.size(); streamSizesNum++) {
      	for (int loopIterSizesNum = 0; loopIterSizesNum < loopIterSizes.size(); loopIterSizesNum++) {
        	for (int loopDepthSizesNum = 0; loopDepthSizesNum < loopDepthSizes.size(); loopDepthSizesNum++) {
          	headerFile << "\t" << "{\"Test_S" << streamSizes[streamSizesNum] << "_" << "V" << varTypes[varTypesNum] << "_" << "I" << loopIterSizes[loopIterSizesNum] << "_" << "D" << loopDepthSizes[loopDepthSizesNum] << "_Form" << (formulasNum+1) << "\", ";
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
            headerFile <<"\"" << varTypes[varTypesNum] << "\"}," << endl;
					}
        }
      }
    }
  }
	headerFile << "\t" << "{0, 0, 0, vector<int>(), vector<int>(), 0, vector<int>(), 0, 0, 0, 0, 0, 0, 0, 0, 0}" << endl;
	headerFile << "};" << endl;
	headerFile << endl;
  headerFile << "#endif" << endl;

	headerFile.close();

}
