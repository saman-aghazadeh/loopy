#include "Algorithm.h"

bool replace(string& str, const string& from, const string& to) {
  size_t start_pos = str.find(from);
  if(start_pos == string::npos)
    return false;
  str.replace(start_pos, from.length(), to);
  return true;
}


CircularNumberGenerator::CircularNumberGenerator (int topBound) {
  this->topBound = topBound;
  this->current = 0;
}

CircularNumberGenerator::~CircularNumberGenerator () {}

int CircularNumberGenerator::next () {
  if (current >= topBound) current = 0;
  int retCurrent = current;
  current++;
  return retCurrent;
}

WorkItemSet::WorkItemSet () {
  this->numOfHomogenousWorkItems = 0;
  this->dependency = false;
  this->numOfInstructions = 0;
}

WorkItemSet::~WorkItemSet () {

}

void WorkItemSet::setNumOfInstructions (int numOfInstructions) {
  this->numOfInstructions = numOfInstructions;
}

int WorkItemSet::getNumOfInstructions () {
  return this->numOfInstructions;
}

void WorkItemSet::setDependency (bool dependency) {
  this->dependency = dependency;
}

bool WorkItemSet::getDependency () {
  return this->dependency;
}

void WorkItemSet::setNumOfHomogenousWorkItems (int numOfHomogenousWorkItems) {
  this->numOfHomogenousWorkItems = numOfHomogenousWorkItems;
}

int WorkItemSet::getNumOfHomogenousWorkItems () {
  return this->numOfHomogenousWorkItems;
}

void WorkItemSet::setFormula (string formula) {
  this->formula = formula;
}

string WorkItemSet::getFormula () {
  return this->formula;
}

void WorkItemSet::setVectorSize (int vectorSize) {
  this->vectorSize = vectorSize;
}

int WorkItemSet::getVectorSize () {
  return this->vectorSize;
}

bool WorkItemSet::getUseLocalMem () {
  return this->useLocalMem;
}

void WorkItemSet::setUseLocalMem (bool useLocalMem) {
  this->useLocalMem = useLocalMem;
}

int WorkItemSet::getFops () {
  return this->fops;
}

void WorkItemSet::setFops (int fops) {
  this->fops = fops;
}

int WorkItemSet::getLoopCarriedDepDegree () {
	return this->loopCarriedDepDegree;
}

void WorkItemSet::setLoopCarriedDepDegree (int loopCarriedDepDegree) {
	this->loopCarriedDepDegree = loopCarriedDepDegree;
}

Algorithm::Algorithm() {
  this->currentIndentation = 0;
 	this->algorithmTargetDevice = -1;
  this->algorithmTargetLanguage = -1;
  this->numComputeUnits = -1;
  //this->currentKernelName = "test";
  this->numberOfNestedForLoops = 0;
  this->totalNumFlops = 0;
  // This item would be equal to 1, all the times.
  // Maybe in future we also support
  // multi-dimensional work groups
  this->workDim = 1;
  this->GInSize = 1;
  this->GOutSize = 1;
  this->currentLocalWorkSizeIndex = 0;
  this->onlyMeta = false;
  this->isV2 = false;
}

Algorithm::~Algorithm() {

}

void Algorithm::PERROR (string errorString) {
  cerr << "[ERROR] " << errorString << endl;
	exit (0);
}

string Algorithm::getIndent () {

  string indent = "";
  for (int i = 0; i < currentIndentation; i++)
    indent += "\t";

  return indent;

}

Algorithm& Algorithm::targetDeviceIs (int algorithmTargetDevice) {

  if (this->algorithmTargetDevice == -1) {
  	this->algorithmTargetDevice = algorithmTargetDevice;
  } else {
  	PERROR ("Target device has already been set!");
  }

  return *this;
}

Algorithm& Algorithm::targetLanguageIs (int algorithmTargetLanguage) {

  if (this->algorithmTargetLanguage == -1 && this->algorithmTargetDevice != -1) {
    if (this->algorithmTargetDevice == AlgorithmTargetDevice::FPGA && this->algorithmTargetLanguage == AlgorithmTargetLanguage::CUDA) {
      PERROR ("CUDA cannot be executed on FPGA platform!");
    } else {
      this->algorithmTargetLanguage = algorithmTargetLanguage;
    }
  } else if (this->algorithmTargetDevice == -1){
		PERROR ("Target Device should be set preliminary to target language!");
  } else if (this->algorithmTargetLanguage != -1) {
    PERROR ("Target Language has already been set!");
  }

  return *this;
}

Algorithm& Algorithm::numComputeUnitsIs (int numComputeUnits) {

  if (this->algorithmTargetDevice != AlgorithmTargetDevice::FPGA) {
    PERROR ("Number of compute units can only be applied on FPGA platform OpenCL codes!");
  }

	oss << getIndent() << "__attribute__((num_compute_units(" << numComputeUnits << ")))" << endl;

  return *this;

}

Algorithm& Algorithm::numSIMDWorkITemsIs (int numSIMDWorkItems) {

	if (this->algorithmTargetDevice != AlgorithmTargetDevice::FPGA) {
    PERROR ("Number of simd work items can only be applied on FPGA plarform OpenCL codes!");
  }

  oss << getIndent() << "__attribute__((num_simd_work_items(" << numSIMDWorkItems << ")))" << endl;

  return *this;

}

Algorithm& Algorithm::memAllocationPerWorkItemIs (int memAllocationPerWorkItem) {
  this->memAllocationPerWorkItem = memAllocationPerWorkItem;

  return *this;
}

Algorithm& Algorithm::workGroupSizeIs (int* workGroupSize) {
  this->localWorkSize = workGroupSize;

  return *this;
}

Algorithm& Algorithm::virtualWorkGroupSizeIs (int* virtualWorkGroupSize) {
  this->virtualLocalWorkSize = virtualWorkGroupSize;

  return *this;
}

Algorithm& Algorithm::memReuseFactorIs (int memoryReuseFactor) {
  this->memoryReuseFactor = memoryReuseFactor;

  return *this;
}

Algorithm& Algorithm::NameIs (string name) {
  this->currentName = name;

  return *this;
}

Algorithm& Algorithm::KernelNameIs (string kernelName) {
  this->currentKernelName = kernelName;

  return *this;
}

Algorithm& Algorithm::startKernelFunction () {

  if (algorithmTargetLanguage == AlgorithmTargetLanguage::OpenCL ) {
  	oss << "__kernel void " << this->currentKernelName << "("
        << "const __global float * restrict GIn, __global float * restrict GOut" << ") {"
        << endl;
		this->currentIndentation++;
  } else if (algorithmTargetLanguage == AlgorithmTargetLanguage::CUDA) {

  }

  return *this;

}

Algorithm& Algorithm::startKernelFunctionV2 () {

  if (algorithmTargetLanguage == AlgorithmTargetLanguage::OpenCL) {
    oss << "__kernel void " << this->currentKernelName << "("
        << "const __global float * restrict GIn, __global float * restrict GOut,"
        << " const int M, const int N, const int P) {" << endl;
    this->currentIndentation++;
  } else if (algorithmTargetLanguage == AlgorithmTargetLanguage::CUDA) {
    
  }

  return *this;
}

Algorithm& Algorithm::startKernelFunctionV3 () {

  if (algorithmTargetLanguage == AlgorithmTargetLanguage::OpenCL) {
    oss << "__kernel void " << this->currentKernelName << "("
        << "const __global float * restrict GIn, __global float * restrict GOut,"
        << " const int M, const int N, const int P) {" << endl;
    this->currentIndentation++;
  } else if (algorithmTargetLanguage == AlgorithmTargetLanguage::CUDA) {
    
  }
}

Algorithm& Algorithm::startKernelFunctionSimpleV1 () {

  if (algorithmTargetLanguage == AlgorithmTargetLanguage::OpenCL) {
    oss << "__kernel void " << this->currentKernelName << "("
        << "const __global float * restrict GIn, __global float * restrict GOut,"
        << " const float M, const float N, const float P) {" << endl;
    this->currentIndentation++;
  } else if (algorithmTargetLanguage == AlgorithmTargetLanguage::CUDA) {
    
  }

  return *this;

}

Algorithm& Algorithm::startKernelFunctionSimpleV2 () {

  if (algorithmTargetLanguage == AlgorithmTargetLanguage::OpenCL) {
    oss << "__kernel void " << this->currentKernelName << "("
        << "const __global float * restrict GIn, __global float * restrict GOut,"
        << " const float M, const float N, const float P) {" << endl;
    this->currentIndentation++;
  } else if (algorithmTargetLanguage == AlgorithmTargetLanguage::CUDA) {
    
  }

  return *this;

}

Algorithm& Algorithm::endKernelFunction () {

	if (algorithmTargetLanguage == AlgorithmTargetLanguage::OpenCL) {
    this->currentIndentation--;
    oss << "}" << endl;
  } else if (algorithmTargetLanguage == AlgorithmTargetLanguage::CUDA) {

  }

  return *this;
}

Algorithm& Algorithm::createFor (int numberOfInstructions, bool dependency,
                                 int loopLength, string formula, int vectorSize,
                                 bool useLocalMem, int fops, int loopCarriedDepDegree) {

	this->vectorSize = vectorSize;

  WorkItemSet workItemSet;

  workItemSet.setNumOfInstructions (numberOfInstructions);
  workItemSet.setDependency (dependency);
  workItemSet.setNumOfHomogenousWorkItems (loopLength);
  workItemSet.setFormula (formula);
  if (!(vectorSize == 1 || vectorSize == 2 || vectorSize == 4 || vectorSize == 8 || vectorSize == 16))
    PERROR ("vector size for the variable should be 1, 2, 4, 8, or 16");
	workItemSet.setVectorSize (vectorSize);
	workItemSet.setUseLocalMem (useLocalMem);
	workItemSet.setFops (fops);
  workItemSet.setLoopCarriedDepDegree (loopCarriedDepDegree);
	(this->forLoops).push_back (workItemSet);
	this->numberOfNestedForLoops++;

  return *this;
}

Algorithm& Algorithm::generateFors (bool onlyMeta) {

	// we need to modify the reuseFactor, so index values
  // access smaller set of data in a circular manner

  int oldMemoryReuseFactor = memoryReuseFactor;
  long long totalMemorySize = 1;
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    totalMemorySize *= forLoops.at(i).getNumOfHomogenousWorkItems();
  }
  totalMemorySize *= memAllocationPerWorkItem;
  memoryReuseFactor = totalMemorySize / memoryReuseFactor;


  this->onlyMeta = onlyMeta;

  if (!onlyMeta)
		generateSingleFor (0);

	memoryReuseFactor = oldMemoryReuseFactor;

  return *this;

}

Algorithm& Algorithm::generateForsV2 (bool onlyMeta) {

	// TODO: Reuse factor should be used here. Right now,
  // we don't consider any reuse factor. Please implement
  // reuse factor for the second version, which should be
  // pretty much easy.

  this->onlyMeta = onlyMeta;

  if (!onlyMeta) {
    int counter = 0;
    for (int i = numberOfNestedForLoops-1; i >= 0; i--) {
      WorkItemSet workItemSet = forLoops.at(i);
      if (workItemSet.getDependency () == false) {
        if (numberOfNestedForLoops == 3) {
      		if (i == 0) {
						oss << getIndent () << "int Z = get_global_id (" << counter << ");" << endl;
          	counter++;
        	} else if (i == 1) {
        		oss << getIndent () << "int Y = get_global_id (" << counter << ");" << endl;
          	counter++;
        	} else if (i == 2) {
        		oss << getIndent () << "int X = get_global_id (" << counter << ");" << endl;
          	counter++;
        	}
        } else if (numberOfNestedForLoops == 2) {
          if (i == 0) {
						oss << getIndent () << "int Y = get_global_id (" << counter << ");" << endl;
          	counter++;
        	} else if (i == 1) {
        		oss << getIndent () << "int X = get_global_id (" << counter << ");" << endl;
          	counter++;
        	}
        } else if (numberOfNestedForLoops == 1) {
          if (i == 0) {
            oss << getIndent () << "int X = get_global_id(" << counter << ");" << endl;
            counter++;
          }
        }
      }
    }

		// Claculating the index in each level makes things really
    // complicated inside the generateSingleForV2 function. Here,
    // we will create a 2-Dimensional array, where represent the
    // current depth in the function and the total depth. For each
    // item in the array we prepare the indexing calculation
    // function. Inside the generateSingleForV2, we just retrieve
    // the statement from the array, instead of performing highly
    // complex calculations and make the code hard to read.

    vector< vector<string> > indexingFormulas;
    vector< vector<string> > indexingFormulasPrev;
    for (int i = 0; i < 3; i++) {
      indexingFormulas.push_back (vector<string>());
      indexingFormulasPrev.push_back (vector<string>());
      for (int j = 0; j < 3; j++) {
        indexingFormulas.at(i).push_back(string());
        indexingFormulasPrev.at(i).push_back(string());
      }
    }

		indexingFormulas.at(0).at(0) = "X";
    indexingFormulas.at(0).at(1) = "";
    indexingFormulas.at(0).at(2) = "";
    indexingFormulas.at(1).at(0) = "Y*M";
    indexingFormulas.at(1).at(1) = "Y*M+X";
    indexingFormulas.at(1).at(2) = "";
    indexingFormulas.at(2).at(0) = "Z*M*N";
    indexingFormulas.at(2).at(1) = "Z*M*N+Y*M";
    indexingFormulas.at(2).at(2) = "Z*M*N+Y*M+X";

    indexingFormulasPrev.at(0).at(0) = "(X-1)";
		indexingFormulasPrev.at(0).at(1) = "";
    indexingFormulasPrev.at(0).at(2) = "";
    indexingFormulasPrev.at(1).at(0) = "(Y-1)*M";
    indexingFormulasPrev.at(1).at(1) = "Y*M+(X-1)";
    indexingFormulasPrev.at(1).at(2) = "";
    indexingFormulasPrev.at(2).at(0) = "(Z-1)*M*N";
    indexingFormulasPrev.at(2).at(1) = "Z*M*N+(Y-1)*M";
    indexingFormulasPrev.at(2).at(2) = "Z*M*N+Y*M+(X-1)";

    generateSingleForV2 (0, indexingFormulas, indexingFormulasPrev);
  }

  return *this;

}

Algorithm& Algorithm::generateForsV3 (bool onlyMeta) {

	// TODO: Reuse factor should be used here. Right now,
  // we don't consider any reuse factor. Please implement
  // reuse factor for the third version, Which should be
  // pretty much easy.

  this->onlyMeta = onlyMeta;

  if (!onlyMeta) {
    int counter = 0;
    for (int i = numberOfNestedForLoops-1; i >= 0; i--) {
      WorkItemSet workItemSet = forLoops.at(i);
      if (workItemSet.getDependency() == false) {
        if (numberOfNestedForLoops == 3){
					if (i == 0) {
          	oss << getIndent () << "int ZGL = get_global_id(" << counter << ");" << endl;
          	oss << getIndent () << "int ZGRid = get_group_id(" << counter << ");" << endl;
						oss << getIndent () << "int ZGRnum = get_num_groups(" << counter << ");" << endl;
            oss << getIndent () << "int ZLSize = get_local_size(" << counter << ");" << endl;
            oss << getIndent () << "int ZLid = get_local_id(" << counter << ");" << endl;
          	counter++;
        	} else if (i == 1) {
          	oss << getIndent () << "int YGL = get_global_id(" << counter << ");" << endl;
          	oss << getIndent () << "int YGRid = get_group_id(" << counter << ");" << endl;
          	oss << getIndent () << "int YGRnum = get_num_groups(" << counter << ");" << endl;
            oss << getIndent () << "int YLSize = get_local_size(" << counter << ");" << endl;
            oss << getIndent () << "int ZLid = get_local_id(" << counter << ");" << endl;
         		counter++;
        	} else if (i == 2) {
          	oss << getIndent () << "int XGL = get_global_id(" << counter << ");" << endl;
            oss << getIndent () << "int XGRid = get_group_id(" << counter << ");" << endl;
            oss << getIndent () << "int XGRnum = get_num_groups(" << counter << ");" << endl;
            oss << getIndent () << "int XLSize = get_local_size(" << counter << ");" << endl;
            oss << getIndent () << "int XLid = get_local_id(" << counter << ");" << endl;
            counter++;
        	}
        } else if (numberOfNestedForLoops == 2) {
          if (i == 0) {
            oss << getIndent () << "int YGL = get_global_id(" << counter << ");" << endl;
            oss << getIndent () << "int YGRid = get_group_id(" << counter << ");" << endl;
            oss << getIndent () << "int YGRnum = get_num_groups(" << counter << ");" << endl;
            oss << getIndent () << "int YLSize = get_local_size(" << counter << ");" << endl;
            oss << getIndent () << "int YLid = get_local_id(" << counter << ");" << endl;
            counter++;
          } else if (i == 1) {
            oss << getIndent () << "int XGL = get_global_id(" << counter << ");" << endl;
            oss << getIndent () << "int XGRid = get_group_id(" << counter << ");" << endl;
            oss << getIndent () << "int XGRnum = get_num_groups(" << counter << ");" << endl;
            oss << getIndent () << "int XLSize = get_local_size(" << counter << ");" << endl;
            oss << getIndent () << "int XLid = get_local_id(" << counter << ");" << endl;
            counter++;
          }
        } else if (numberOfNestedForLoops == 1) {
          if (i == 0) {
            oss << getIndent () << "int XGL = get_global_id(" << counter << ");" << endl;
            oss << getIndent () << "int XGRid = get_group_id(" << counter << ");" << endl;
            oss << getIndent () << "int XGRnum = get_num_groups(" << counter << ");" << endl;
            oss << getIndent () << "int XLSize = get_local_size(" << counter << ");" << endl;
            oss << getIndent () << "int XLid = get_local_id(" << counter << ");" << endl;
            counter++;
          }
        }
      }
    }

    // Claculating the index in each level makes things really
    // complicated inside the generateSingleForV2 function. Here,
    // we will create a 2-Dimensional array, where represent the
    // current depth in the function and the total depth. For each
    // item in the array we prepare the indexing calculation
    // function. Inside the generateSingleForV2, we just retrieve
    // the statement from the array, instead of performing highly
    // complex calculations and make the code hard to read.

    vector< vector<string> > indexingFormulas;
    vector< vector<string> > indexingFormulasPrev;

    for (int i = 0; i < 3; i++) {
      indexingFormulas.push_back (vector<string>());
      indexingFormulasPrev.push_back (vector<string>());
      for (int j = 0; j < 3; j++) {
        indexingFormulas.at(i).push_back (string());
        indexingFormulasPrev.at(i).push_back (string());
      }
    }

    stringstream memAllPerWI;
    memAllPerWI << memAllocationPerWorkItem;
    indexingFormulas.at(0).at(0) = "XGRid*XLSize*"+ memAllPerWI.str()
      + "+" + "XLid";
		indexingFormulas.at(0).at(1) = "";
    indexingFormulas.at(0).at(2) = "";

    indexingFormulas.at(1).at(0) = "YGRid*XGRnum*(XLSize*YLSize)*" + memAllPerWI.str()
      + "+" + "YLid*XLSize";
    indexingFormulas.at(1).at(1) = "(YGRid*XGRnum+XGRid)*(XLSize*YLSize)*" + memAllPerWI.str()
      + "+" + "YLid*XLSize + XLid";
		indexingFormulas.at(1).at(2) = "";

    indexingFormulas.at(2).at(0) = "ZGRid*YGRnum*XGRnum*(XLSize*YLSize*ZLSize)*" + memAllPerWI.str()
      + "+" + "ZLid*YLSize*XLSize";
    indexingFormulas.at(2).at(1) = "(ZGRid*YGRnum*XGRnum+YGRid*XGRnum)*(XLSize*YLSize*ZLSize)*" + memAllPerWI.str()
      + "+" + "ZLid*YLSize*XLSize+YLid*XLSize";
    indexingFormulas.at(2).at(2) = "(ZGRid*YGRnum*XGRnum+YGRid*XGRnum+XGRid)*(XLSize*YLSize*ZLSize)*" + memAllPerWI.str()
      + "+" + "ZLid*YLSize*XLSize+YLid*XLSize+XLid";

    stringstream memAllPerWIPrev;
    memAllPerWIPrev << memAllocationPerWorkItem;
    indexingFormulasPrev.at(0).at(0) = "(XRGid*XLSize*" + memAllPerWIPrev.str()
      + "+" + "XLid" + "-1)";
    indexingFormulasPrev.at(0).at(1) = "";
		indexingFormulasPrev.at(0).at(2) = "";

    indexingFormulasPrev.at(1).at(0) = "(YGRid*XGRnum*(XLSize*YLSize)*" + memAllPerWIPrev.str()
      + "+" + "YLid*XLSize" + "-1)";
    indexingFormulasPrev.at(1).at(1) = "((YGRid*XGRnum+XGRid)*(XLSize*YLSize)*" + memAllPerWIPrev.str()
      + "+" + "YLid*XLSize+XLid" + "-1)";
    indexingFormulasPrev.at(1).at(2) = "";

    indexingFormulasPrev.at(2).at(0) = "(ZGRid*YGRnum *XGRnum*(XLSize*YLSize*ZLSize)*" + memAllPerWIPrev.str()
      + "+" + "ZLid*YLSize*XLSize" + "-1)";
    indexingFormulasPrev.at(2).at(1) = "((ZGRid*YGRnum*XGRnum+YGRid*XGRnum)*(XLSize*YLSize*ZLSize)*"
      + memAllPerWIPrev.str() + "ZLid*YLSize*XLSize+YLid*XLSize" + "-1)";
    indexingFormulasPrev.at(2).at(2) = "((ZGRid*YGRnum*XGRnum+YGRid*XGRnum+XGRid)*(XLSize*YLSize*ZLSize)*"
     	+ memAllPerWIPrev.str() + "ZLid*YLSize*XLSize+YLid*XLSize+XLid" + "-1)";

    generateSingleForV3 (0, indexingFormulas, indexingFormulasPrev);
  }

  return *this;

}

Algorithm& Algorithm::generateForsSimpleV1 (bool onlyMeta) {

	// TODO: Reuse factor should be used here. Right now,
  // we don't consider any reuse factor. Please implement
  // reuse factor for the third version, Which should be
  // pretty much easy.

  this->onlyMeta = onlyMeta;

  if (!onlyMeta) {
    int counter = 0;
    for (int i = numberOfNestedForLoops-1; i >= 0; i--) {
      WorkItemSet workItemSet = forLoops.at(i);
      if (numberOfNestedForLoops == 3){
				if (i == 0) {
          if (workItemSet.getDependency() == false) {
          	oss << getIndent () << "const int ZGL = get_global_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int ZGRid = get_group_id(" << counter << ");" << endl;
						oss << getIndent () << "const int ZGRnum = get_num_groups(" << counter << ");" << endl;
          	oss << getIndent () << "const int ZLSize = get_local_size(" << counter << ");" << endl;
          	oss << getIndent () << "const int ZLid = get_local_id(" << counter << ");" << endl;
          } else if (workItemSet.getDependency() == true) {
            int numOfHomogenousWorkItems = workItemSet.getNumOfHomogenousWorkItems();
            oss << getIndent () << "const int ZGRnum = " << numOfHomogenousWorkItems << "/M;" << endl;
            oss << getIndent () << "const int ZLSize = M;" << endl; 
          }
          counter++;
        } else if (i == 1) {
          if (workItemSet.getDependency() == false) {
          	oss << getIndent () << "const int YGL = get_global_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int YGRid = get_group_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int YGRnum = get_num_groups(" << counter << ");" << endl;
          	oss << getIndent () << "const int YLSize = get_local_size(" << counter << ");" << endl;
          	oss << getIndent () << "const int YLid = get_local_id(" << counter << ");" << endl;
          } else if (workItemSet.getDependency() == true) {
            int numOfHomogenousWorkItems = workItemSet.getNumOfHomogenousWorkItems();
            oss << getIndent () << "const int YGRnum = " << numOfHomogenousWorkItems << "/N;" << endl;
            oss << getIndent () << "const int YLSize = N;" << endl;
          }
         	counter++;
        } else if (i == 2) {
          if (workItemSet.getDependency() == false) {
          	oss << getIndent () << "const int XGL = get_global_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int XGRid = get_group_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int XGRnum = get_num_groups(" << counter << ");" << endl;
          	oss << getIndent () << "const int XLSize = get_local_size(" << counter << ");" << endl;
          	oss << getIndent () << "const int XLid = get_local_id(" << counter << ");" << endl;
          } else if (workItemSet.getDependency() == true) {
            int numOfHomogenousWorkItems = workItemSet.getNumOfHomogenousWorkItems();
            oss << getIndent () << "const int XGRnum = " << numOfHomogenousWorkItems << "/P;" << endl;
            oss << getIndent () << "const int XLSize = N;" << endl;
          }
          counter++;
        }
      } else if (numberOfNestedForLoops == 2) {
        if (i == 0) {
          if (workItemSet.getDependency() == false) {
          	oss << getIndent () << "const int YGL = get_global_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int YGRid = get_group_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int YGRnum = get_num_groups(" << counter << ");" << endl;
          	oss << getIndent () << "const int YLSize = get_local_size(" << counter << ");" << endl;
          	oss << getIndent () << "const int YLid = get_local_id(" << counter << ");" << endl;
          } else if (workItemSet.getDependency() == true) {
            int numOfHomogenousWorkItems = workItemSet.getNumOfHomogenousWorkItems();
            oss << getIndent () << "const int YGRnum = " << numOfHomogenousWorkItems << "/N;" << endl;
            oss << getIndent () << "const int YLSize = N;" << endl;
          }
          counter++;
        } else if (i == 1) {
          if (workItemSet.getDependency() == false) {
          	oss << getIndent () << "const int XGL = get_global_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int XGRid = get_group_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int XGRnum = get_num_groups(" << counter << ");" << endl;
          	oss << getIndent () << "const int XLSize = get_local_size(" << counter << ");" << endl;
          	oss << getIndent () << "const int XLid = get_local_id(" << counter << ");" << endl;
          } else if (workItemSet.getDependency() == true) {
            int numOfHomogenousWorkItem = workItemSet.getNumOfHomogenousWorkItems();
						oss << getIndent () << "const int XGRnum = " << numOfHomogenousWorkItem << "/P;" << endl;
            oss << getIndent () << "const int XLSize = P;" << endl;
          }
          counter++;
        }
      } else if (numberOfNestedForLoops == 1) {
        if (i == 0) {
          if (workItemSet.getDependency() == false) {
          	oss << getIndent () << "const int XGL = get_global_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int XGRid = get_group_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int XGRnum = get_num_groups(" << counter << ");" << endl;
          	oss << getIndent () << "const int XLSize = get_local_size(" << counter << ");" << endl;
          	oss << getIndent () << "const int XLid = get_local_id(" << counter << ");" << endl;
          } else if (workItemSet.getDependency() == true) {
            int numOfHomogenousWorkItem = workItemSet.getNumOfHomogenousWorkItems();
            oss << getIndent () << "const int XGRnum = " << numOfHomogenousWorkItem << "/P;" << endl;
            oss << getIndent () << "const int XLSize = P;" << endl;
          }
          counter++;
        }
      }
    }

    // Claculating the index in each level makes things really
    // complicated inside the generateSingleForV2 function. Here,
    // we will create a 2-Dimensional array, where represent the
    // current depth in the function and the total depth. For each
    // item in the array we prepare the indexing calculation
    // function. Inside the generateSingleForV2, we just retrieve
    // the statement from the array, instead of performing highly
    // complex calculations and make the code hard to read.

    vector< vector<string> > indexingFormulas;
    vector< vector<string> > indexingFormulasPrev;
		vector< vector<string> > indexingLocalMem;
    for (int i = 0; i < 3; i++) {
      indexingFormulas.push_back (vector<string>());
      indexingFormulasPrev.push_back (vector<string>());
      indexingLocalMem.push_back (vector<string>());
      for (int j = 0; j < 3; j++) {
        indexingFormulas.at(i).push_back (string());
        indexingFormulasPrev.at(i).push_back (string());
        indexingLocalMem.at(i).push_back (string());
      }
    }

    stringstream memAllPerWI;
    memAllPerWI << memAllocationPerWorkItem;
    indexingFormulas.at(0).at(0) = "XGRid*XLSize*" + memAllPerWI.str() + "+XLid";
		indexingFormulas.at(0).at(1) = "";
    indexingFormulas.at(0).at(2) = "";

    indexingFormulas.at(1).at(0) = "YGRid*XGRnum*(XLSize*YLSize)*" + memAllPerWI.str() + "+YLid*XLSize";
    indexingFormulas.at(1).at(1) = "(YGRid*XGRnum+XGRid)*(XLSize*YLSize)*" + memAllPerWI.str()+ "+YLid*XLSize+XLid";
		indexingFormulas.at(1).at(2) = "";

    indexingFormulas.at(2).at(0) = "ZGRid*YGRnum*XGRnum*(XLSize*YLSize*ZLSize)*" + memAllPerWI.str() + "+ZLid*YLSize*XLSize";
    indexingFormulas.at(2).at(1) = "(ZGRid*YGRnum*XGRnum+YGRid*XGRnum)*(XLSize*YLSize*ZLSize)*" + memAllPerWI.str()+ "+ZLid*YLSize*XLSize+YLid*XLSize";
    indexingFormulas.at(2).at(2) = "(ZGRid*YGRnum*XGRnum+YGRid*XGRnum+XGRid)*(XLSize*YLSize*ZLSize)*" + memAllPerWI.str() + "+ZLid*YLSize*XLSize+YLid*XLSize+XLid";

    stringstream memAllPerWIPrev;
    memAllPerWIPrev << memAllocationPerWorkItem;
    indexingFormulasPrev.at(0).at(0) = "(XGRid*XLSize*" + memAllPerWIPrev.str() + "-1)";
    //indexingFormulasPrev.at(0).at(0) = "(XRGid*XLSize+XLid)*(XLSize*YLSize)*" + memAllPerWIPrev.str() + "-1)";
    indexingFormulasPrev.at(0).at(1) = "";
		indexingFormulasPrev.at(0).at(2) = "";

    indexingFormulasPrev.at(1).at(0) = "(YGRid*XGRnum*(XLSize*YLSize)*" + memAllPerWIPrev.str()
      + "+" + "YLid*XLSize" + "-1)";
    indexingFormulasPrev.at(1).at(1) = "((YGRid*XGRnum+XGRid)*(XLSize*YLSize)*" + memAllPerWIPrev.str()
      + "+" + "YLid*XLSize+XLid" + "-1)";
    indexingFormulasPrev.at(1).at(2) = "";

    indexingFormulasPrev.at(2).at(0) = "(ZGRid*YGRnum *XGRnum*(XLSize*YLSize*ZLSize)*" + memAllPerWIPrev.str()
      + "+" + "ZLid*YLSize*XLSize" + "-1)";
    indexingFormulasPrev.at(2).at(1) = "((ZGRid*YGRnum*XGRnum+YGRid*XGRnum)*(XLSize*YLSize*ZLSize)*"
      + memAllPerWIPrev.str() + "ZLid*YLSize*XLSize+YLid*XLSize" + "-1)";
    indexingFormulasPrev.at(2).at(2) = "((ZGRid*YGRnum*XGRnum+YGRid*XGRnum+XGRid)*(XLSize*YLSize*ZLSize)*"
     	+ memAllPerWIPrev.str() + "ZLid*YLSize*XLSize+YLid*XLSize+XLid" + "-1)";

    indexingLocalMem.at(0).at(0) = "XLid";
    indexingLocalMem.at(0).at(1) = "";
    indexingLocalMem.at(0).at(2) = "";

    indexingLocalMem.at(1).at(0) = "YLid*XLSize";
    indexingLocalMem.at(1).at(1) = "(YLid*XLSize+XLid)";
    indexingLocalMem.at(1).at(2) = "";

    indexingLocalMem.at(2).at(0) = "ZLid*YLSize*XLSize";
    indexingLocalMem.at(2).at(1) = "(ZLid*YLSize*XLSize+YLid*XLSize)";
    indexingLocalMem.at(2).at(2) = "(ZLid*YLSize*XLSize+YLid*XLSize+XLid)";

    generateSingleForSimpleV1 (0, indexingFormulas, indexingFormulasPrev, indexingLocalMem);
  }

  return *this;

}

Algorithm& Algorithm::generateForsSimpleV2 (bool onlyMeta) {

	// TODO: Reuse factor should be used here. Right now,
  // we don't consider any reuse factor. Please implement
  // reuse factor for the third version, Which should be
  // pretty much easy.

  this->onlyMeta = onlyMeta;

  if (!onlyMeta) {
    int counter = 0;
    for (int i = numberOfNestedForLoops-1; i >= 0; i--) {
      WorkItemSet workItemSet = forLoops.at(i);
      if (numberOfNestedForLoops == 3){
				if (i == 0) {
          if (workItemSet.getDependency() == false) {
          	oss << getIndent () << "const int ZGL = get_global_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int ZGRid = get_group_id(" << counter << ");" << endl;
						oss << getIndent () << "const int ZGRnum = get_num_groups(" << counter << ");" << endl;
          	oss << getIndent () << "const int ZLSize = get_local_size(" << counter << ");" << endl;
          	oss << getIndent () << "const int ZLid = get_local_id(" << counter << ");" << endl;
          } else if (workItemSet.getDependency() == true) {
            int numOfHomogenousWorkItems = workItemSet.getNumOfHomogenousWorkItems();
            oss << getIndent () << "const int ZGRnum = " << numOfHomogenousWorkItems << "/M;" << endl;
            oss << getIndent () << "const int ZLSize = M;" << endl; 
          }
          counter++;
        } else if (i == 1) {
          if (workItemSet.getDependency() == false) {
          	oss << getIndent () << "const int YGL = get_global_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int YGRid = get_group_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int YGRnum = get_num_groups(" << counter << ");" << endl;
          	oss << getIndent () << "const int YLSize = get_local_size(" << counter << ");" << endl;
          	oss << getIndent () << "const int YLid = get_local_id(" << counter << ");" << endl;
          } else if (workItemSet.getDependency() == true) {
            int numOfHomogenousWorkItems = workItemSet.getNumOfHomogenousWorkItems();
            oss << getIndent () << "const int YGRnum = " << numOfHomogenousWorkItems << "/N;" << endl;
            oss << getIndent () << "const int YLSize = N;" << endl;
          }
         	counter++;
        } else if (i == 2) {
          if (workItemSet.getDependency() == false) {
          	oss << getIndent () << "const int XGL = get_global_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int XGRid = get_group_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int XGRnum = get_num_groups(" << counter << ");" << endl;
          	oss << getIndent () << "const int XLSize = get_local_size(" << counter << ");" << endl;
          	oss << getIndent () << "const int XLid = get_local_id(" << counter << ");" << endl;
          } else if (workItemSet.getDependency() == true) {
            int numOfHomogenousWorkItems = workItemSet.getNumOfHomogenousWorkItems();
            oss << getIndent () << "const int XGRnum = " << numOfHomogenousWorkItems << "/P;" << endl;
            oss << getIndent () << "const int XLSize = N;" << endl;
          }
          counter++;
        }
      } else if (numberOfNestedForLoops == 2) {
        if (i == 0) {
          if (workItemSet.getDependency() == false) {
          	oss << getIndent () << "const int YGL = get_global_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int YGRid = get_group_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int YGRnum = get_num_groups(" << counter << ");" << endl;
          	oss << getIndent () << "const int YLSize = get_local_size(" << counter << ");" << endl;
          	oss << getIndent () << "const int YLid = get_local_id(" << counter << ");" << endl;
          } else if (workItemSet.getDependency() == true) {
            int numOfHomogenousWorkItems = workItemSet.getNumOfHomogenousWorkItems();
            oss << getIndent () << "const int YGRnum = " << numOfHomogenousWorkItems << "/N;" << endl;
            oss << getIndent () << "const int YLSize = N;" << endl;
          }
          counter++;
        } else if (i == 1) {
          if (workItemSet.getDependency() == false) {
          	oss << getIndent () << "const int XGL = get_global_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int XGRid = get_group_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int XGRnum = get_num_groups(" << counter << ");" << endl;
          	oss << getIndent () << "const int XLSize = get_local_size(" << counter << ");" << endl;
          	oss << getIndent () << "const int XLid = get_local_id(" << counter << ");" << endl;
          } else if (workItemSet.getDependency() == true) {
            int numOfHomogenousWorkItem = workItemSet.getNumOfHomogenousWorkItems();
						oss << getIndent () << "const int XGRnum = " << numOfHomogenousWorkItem << "/P;" << endl;
            oss << getIndent () << "const int XLSize = P;" << endl;
          }
          counter++;
        }
      } else if (numberOfNestedForLoops == 1) {
        if (i == 0) {
          if (workItemSet.getDependency() == false) {
          	oss << getIndent () << "const int XGL = get_global_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int XGRid = get_group_id(" << counter << ");" << endl;
          	oss << getIndent () << "const int XGRnum = get_num_groups(" << counter << ");" << endl;
          	oss << getIndent () << "const int XLSize = get_local_size(" << counter << ");" << endl;
          	oss << getIndent () << "const int XLid = get_local_id(" << counter << ");" << endl;
          } else if (workItemSet.getDependency() == true) {
            int numOfHomogenousWorkItem = workItemSet.getNumOfHomogenousWorkItems();
            oss << getIndent () << "const int XGRnum = " << numOfHomogenousWorkItem << "/P;" << endl;
            oss << getIndent () << "const int XLSize = P;" << endl;
          }
          counter++;
        }
      }
    }

    // Claculating the index in each level makes things really
    // complicated inside the generateSingleForV2 function. Here,
    // we will create a 2-Dimensional array, where represent the
    // current depth in the function and the total depth. For each
    // item in the array we prepare the indexing calculation
    // function. Inside the generateSingleForV2, we just retrieve
    // the statement from the array, instead of performing highly
    // complex calculations and make the code hard to read.

    vector< vector<string> > indexingFormulas;
    vector< vector<string> > indexingFormulasPrev;
		vector< vector<string> > indexingLocalMem;
    for (int i = 0; i < 3; i++) {
      indexingFormulas.push_back (vector<string>());
      indexingFormulasPrev.push_back (vector<string>());
      indexingLocalMem.push_back (vector<string>());
      for (int j = 0; j < 3; j++) {
        indexingFormulas.at(i).push_back (string());
        indexingFormulasPrev.at(i).push_back (string());
        indexingLocalMem.at(i).push_back (string());
      }
    }

    stringstream memAllPerWI;
    memAllPerWI << memAllocationPerWorkItem;
    indexingFormulas.at(0).at(0) = "XGRid*XLSize*" + memAllPerWI.str() + "+XLid";
		indexingFormulas.at(0).at(1) = "";
    indexingFormulas.at(0).at(2) = "";

    indexingFormulas.at(1).at(0) = "YGRid*XGRnum*(XLSize*YLSize)*" + memAllPerWI.str() + "+YLid*XLSize";
    indexingFormulas.at(1).at(1) = "(YGRid*XGRnum+XGRid)*(XLSize*YLSize)*" + memAllPerWI.str()+ "+YLid*XLSize+XLid";
		indexingFormulas.at(1).at(2) = "";

    indexingFormulas.at(2).at(0) = "ZGRid*YGRnum*XGRnum*(XLSize*YLSize*ZLSize)*" + memAllPerWI.str() + "+ZLid*YLSize*XLSize";
    indexingFormulas.at(2).at(1) = "(ZGRid*YGRnum*XGRnum+YGRid*XGRnum)*(XLSize*YLSize*ZLSize)*" + memAllPerWI.str()+ "+ZLid*YLSize*XLSize+YLid*XLSize";
    indexingFormulas.at(2).at(2) = "(ZGRid*YGRnum*XGRnum+YGRid*XGRnum+XGRid)*(XLSize*YLSize*ZLSize)*" + memAllPerWI.str() + "+ZLid*YLSize*XLSize+YLid*XLSize+XLid";

    stringstream memAllPerWIPrev;
    memAllPerWIPrev << memAllocationPerWorkItem;
    indexingFormulasPrev.at(0).at(0) = "(XGRid*XLSize*" + memAllPerWIPrev.str() + "-1)";
    //indexingFormulasPrev.at(0).at(0) = "(XRGid*XLSize+XLid)*(XLSize*YLSize)*" + memAllPerWIPrev.str() + "-1)";
    indexingFormulasPrev.at(0).at(1) = "";
		indexingFormulasPrev.at(0).at(2) = "";

    indexingFormulasPrev.at(1).at(0) = "(YGRid*XGRnum*(XLSize*YLSize)*" + memAllPerWIPrev.str()
      + "+" + "YLid*XLSize" + "-1)";
    indexingFormulasPrev.at(1).at(1) = "((YGRid*XGRnum+XGRid)*(XLSize*YLSize)*" + memAllPerWIPrev.str()
      + "+" + "YLid*XLSize+XLid" + "-1)";
    indexingFormulasPrev.at(1).at(2) = "";

    indexingFormulasPrev.at(2).at(0) = "(ZGRid*YGRnum *XGRnum*(XLSize*YLSize*ZLSize)*" + memAllPerWIPrev.str()
      + "+" + "ZLid*YLSize*XLSize" + "-1)";
    indexingFormulasPrev.at(2).at(1) = "((ZGRid*YGRnum*XGRnum+YGRid*XGRnum)*(XLSize*YLSize*ZLSize)*"
      + memAllPerWIPrev.str() + "ZLid*YLSize*XLSize+YLid*XLSize" + "-1)";
    indexingFormulasPrev.at(2).at(2) = "((ZGRid*YGRnum*XGRnum+YGRid*XGRnum+XGRid)*(XLSize*YLSize*ZLSize)*"
     	+ memAllPerWIPrev.str() + "ZLid*YLSize*XLSize+YLid*XLSize+XLid" + "-1)";

    indexingLocalMem.at(0).at(0) = "XLid";
    indexingLocalMem.at(0).at(1) = "";
    indexingLocalMem.at(0).at(2) = "";

    indexingLocalMem.at(1).at(0) = "YLid*XLSize";
    indexingLocalMem.at(1).at(1) = "(YLid*XLSize+XLid)";
    indexingLocalMem.at(1).at(2) = "";

    indexingLocalMem.at(2).at(0) = "ZLid*YLSize*XLSize";
    indexingLocalMem.at(2).at(1) = "(ZLid*YLSize*XLSize+YLid*XLSize)";
    indexingLocalMem.at(2).at(2) = "(ZLid*YLSize*XLSize+YLid*XLSize+XLid)";

    generateSingleForSimpleV2 (0, indexingFormulas, indexingFormulasPrev, indexingLocalMem);
  }

  return *this;

}


Algorithm& Algorithm::generateSingleFor (int loopIndex) {

  bool leaf = false;

	if (loopIndex == numberOfNestedForLoops)
    return *this;

  if (loopIndex == numberOfNestedForLoops - 1)
  	leaf = true;

  WorkItemSet currentWorkItemSet = forLoops.at (loopIndex);
	bool dependency = currentWorkItemSet.getDependency ();
  int numOfInstructions = currentWorkItemSet.getNumOfInstructions ();
  int numOfHomogenousWorkItems = currentWorkItemSet.getNumOfHomogenousWorkItems ();
	string formula = currentWorkItemSet.getFormula ();
	int vectorSize = currentWorkItemSet.getVectorSize ();
	int useLocalMem = currentWorkItemSet.getUseLocalMem ();

  if (loopIndex == 0) {
  	oss << getIndent() << "// Just a private variable" << endl;
    if (vectorSize == 1)
			oss << getIndent() << "float temp = 1.0;" << endl;
    else
      oss << getIndent() << "float" << vectorSize << " temp = 1.0;" << endl;
  	oss << endl;
  }

  if (loopIndex == 0)
  	oss << getIndent () << "int TID = get_global_id(0);" << endl;

  // Here we will check if we are at the last nested for section,
  // and whether we are going to read from local memory. In this
  // case we generate a piece of code, which copies the relative
  // part of global memory into local, and then forces the ops
  // to read the values from the local memory, instead of global.
	if (useLocalMem == true)
		oss << getIndent() << "__local float GLIn[" << localWorkSize[0]*memAllocationPerWorkItem << "];" << endl;

	// First we prepare dividend value, which be used the find the index of the block
  // Which current level of granularity need to access.
  int dividendValue = 1;
  for (int i = loopIndex+1; i < forLoops.size(); i++ )
		dividendValue *= forLoops.at(i).getNumOfHomogenousWorkItems();

  stringstream dividend;
  dividend << "int dividend" << loopIndex+1 << " = " << dividendValue << ";";


  // Now let's derive the begining location of the data the work item is going to
  // access. It is not the final correct index, but just points to the beginning
  // of the block in current level granularity, where the real index reside;
	stringstream i_d;
  i_d << "int i" << loopIndex+1 << "d = dividend" << loopIndex+1 << " * " << "i"
      << loopIndex+1 << ";";

  // Now let's calculate the final index, which points to the beginning of the block,
  // where the work item will access for processing.
  stringstream index;
  index << "int index" << loopIndex+1 << " = ";
  for (int i = loopIndex; i >=0 ; i-- ) {
    if (i == 0)
      index << "i" << i+1 << "d;";
    else
			index << "i" << i+1 << "d + ";
  }

  // If we have memory reuse factor being set, we need to replace index,
  // with it's residue by memoryResuseFactor value
  stringstream indexReused;
  if (memoryReuseFactor != 0) {
    indexReused << "index" << loopIndex+1 << " = index" << loopIndex+1 << " % " << memoryReuseFactor << ";";
  }

  // Now let's prepare the TID, which is going to be divided by the dividend,
  // and then give the index.
	stringstream TID;
  TID << "(TID";
  for (int i = loopIndex+1; i < numberOfNestedForLoops; i++) {
    if (forLoops.at(i).getDependency() == true)
    	TID << " * " << forLoops.at(i).getNumOfHomogenousWorkItems();
  }
  TID << ")";


  // Now it's time to flush the code into the stream,
  // Existance or non-existance of dependency will affect,
  // the final outcome.
	if (dependency == false) {
    oss << endl;
    oss << getIndent() << "// Start of a new level of for loop" << endl;
		oss << getIndent() << dividend.str() << endl;
    // This is the index for current level. In case of dependency = true,
    // it will come from the loop index.
		oss << getIndent() << "int i" << loopIndex+1 << " = (" << TID.str() << " / " << "dividend" << loopIndex+1 << ") % "
        << numOfHomogenousWorkItems << ";" << endl;
    oss << getIndent() << i_d.str() << endl;
    oss << getIndent() << index.str() << endl;
    oss << getIndent() << indexReused.str() << endl;

    if (useLocalMem)
    	oss << getIndent() << "async_work_group_copy (&(GLIn["
          << "get_local_id(0) * " << memAllocationPerWorkItem
          << "]), &(GIn[index" << loopIndex+1 << "]), "
          << memAllocationPerWorkItem << ", 0);"
          << endl;

		// Here we insert the operation. Basically, we do this operation
    // number of instruction's times.
		CircularNumberGenerator CNG (memAllocationPerWorkItem);
    if (useLocalMem) {
      // Preparing the index, which points into the local memory
      oss << getIndent() << "int li" << loopIndex+1 << " = get_local_id(0) * "
          << memAllocationPerWorkItem << ";" << endl;
    }
		for (int i = 0; i < numOfInstructions; i++) {
      if (!useLocalMem) {
      	string formulaRepl (formula);
      	stringstream replacement;
      	// This part will replace the index in GIn with the index that
      	// should be really accessed in this work item
      	replacement << "index" << loopIndex+1 << " + " << CNG.next();
				replace (formulaRepl, "@", replacement.str());
				oss << getIndent() << formulaRepl << ";" << endl;
      } else {
        string formulaRepl (formula);
        stringstream replacement;

        // Since we are reading from the local memory, we will replace
        // index#i offset in the GLIn, with i#i;
        replacement << "li" << loopIndex+1 << " + " << CNG.next();
        replace (formulaRepl, "@", replacement.str());
				replace (formulaRepl, "GIn", "GLIn");
        oss << getIndent() << formulaRepl << ";" << endl;
      }
    }

    // if this is the last nested for loops, we will go on and write the
    // calculated value into the memory.
		if (leaf == true) {
      if (vectorSize == 1)
				oss << getIndent() << "GOut[index" << loopIndex+1 << "] = temp;" << endl;
      else {
        oss << getIndent() << "GOut[index" << loopIndex+1 << "] = temp.s0";
				for (int i = 1; i < vectorSize; i++) {
          if (i < 10)
          	oss << " + " << "temp.s" << i;
          else if (i == 10)
						oss << " + " << "temp.sA";
          else if (i == 11)
            oss << " + " << "temp.sB";
          else if (i == 12)
            oss << " + " << "temp.sC";
          else if (i == 13)
            oss << " + " << "temp.sD";
          else if (i == 14)
            oss << " + " << "temp.sE";
          else if (i == 15)
            oss << " + " << "temp.sF";
        }
        oss << ";" << endl;
      }

    }

    // Now it's time to recursively call the single for algorithm,
    // in order to generate the next block of work items.
    generateSingleFor (loopIndex+1);

  } else if (dependency == true) {
    oss << endl;
    if (useLocalMem)
    	oss << getIndent() << "async_work_group_copy (GLIn, &(GLIn[index" << loopIndex+1 << "]), " << localWorkSize[0] * memAllocationPerWorkItem << ", 0);" << endl;

    oss << getIndent() << "// Start of a new level of for loop" << endl;
    oss << getIndent() << "for (int i" << loopIndex+1 << " = 0; i" << loopIndex+1 << " < "
        << numOfHomogenousWorkItems << "; i" << loopIndex+1 << " = " << "i" << loopIndex+1 << " + 1){" << endl;
    // increase indentation, since we got into the for loop
    currentIndentation++;

    oss << getIndent() << dividend.str() << endl;
		oss << getIndent() << i_d.str() << endl;
    oss << getIndent() << index.str() << endl;
    oss << getIndent() << indexReused.str() << endl;

    // This refers as an index for accessing the data, which the operation will depend onB
		stringstream i_dd;
    i_dd << "int i" << loopIndex+1 << "dd = dividend" << loopIndex+1 << " * (i" << loopIndex+1 << "d - 1);";

    // This is the final index for the dependend data
    stringstream indexd;
    indexd << "int index" << loopIndex+1 << "d = i" << loopIndex+1 << "dd";
    for (int i = loopIndex; i >= 1; i--) {
      indexd << " + i" << i << "d";
    }
    indexd << ";";

    oss << getIndent() << i_dd.str() << endl;
    oss << getIndent() << indexd.str() << endl;

		if (useLocalMem) {
      oss << getIndent() << "int li" << loopIndex+1 << " = get_local_id(0) * "
          << memAllocationPerWorkItem << ";" << endl;
    }

		// Here we insert the operation. Basically, we do this operation
    // number of instruction's times.
		CircularNumberGenerator CNG (memAllocationPerWorkItem);
		for (int i = 0; i < numOfInstructions; i++) {
      if (!useLocalMem) {
      	string formulaRepl (formula);
      	stringstream replacementIdx1;
      	stringstream replacementIdx2;

      	// This part will replace the current index and the dependent index
      	// in GIn with the index that should be really accessed i nthis work
      	// item.
      	int cngNext = CNG.next();
      	replacementIdx1 << "index" << loopIndex+1 << " + " << cngNext;
     	 	replacementIdx2 << "index" << loopIndex+1 << "d + " << cngNext;
      	replace (formulaRepl, "@", replacementIdx1.str());
      	replace (formulaRepl, "!", replacementIdx2.str());
      	oss << getIndent() << formulaRepl << ";" << endl;
      } else {
        string formulaRepl (formula);
        stringstream replacementIdx1;
				stringstream replacementIdx2;

        // Here we do two things different from the case where we will
        // utilize the global memory. First, we will avoid inserting
        // the index#i offset on first appearance of GIn and we will
        // still read the from global memory for the second appearance
        // of GIn in the fromula.
				//
        // TODO: we should modify the formula, to read the depndee
        // part from local memory too. This may require some minor
        // changes in the local memory allocation too.
        int cngNext = CNG.next();
        replacementIdx1 << "li" << loopIndex+1 << " + " << cngNext;
        replacementIdx2 << "index" << loopIndex+1 << "d + " << cngNext;
				replace (formulaRepl, "@", replacementIdx1.str());
        replace (formulaRepl, "!", replacementIdx2.str());
        replace (formulaRepl, "GIn", "GLIn");
        oss << getIndent() << formulaRepl << ";" << endl;
      }
    }

    // if this is the last nested for loops, we will go on and write the
    // calculated value into the memory.
		if (leaf == true) {
      if (vectorSize == 1)
				oss << getIndent() << "GOut[index" << loopIndex+1 << "] = temp;" << endl;
      else {
        oss << getIndent() << "GOut[index" << loopIndex+1 << "] = temp.s0";
				for (int i = 1; i < vectorSize; i++) {
          if (i < 10)
          	oss << " + " << "temp.s" << i;
          else if (i == 10)
						oss << " + " << "temp.sA";
          else if (i == 11)
            oss << " + " << "temp.sB";
          else if (i == 12)
            oss << " + " << "temp.sC";
          else if (i == 13)
            oss << " + " << "temp.sD";
          else if (i == 14)
            oss << " + " << "temp.sE";
          else if (i == 15)
            oss << " + " << "temp.sF";
        }
        oss << ";" << endl;
      }
    }


    // Now it's time to recursively cal the single for algorithm,
    // in order to generate the next block of work items.
		generateSingleFor (loopIndex+1);

		currentIndentation--;
    oss << getIndent() << "}" << endl;

  }

  return *this;

}

Algorithm& Algorithm::generateSingleForV2 (
                       				int loopIndex,
                             	vector<vector<string> >& indexingFormulas,
                              vector<vector<string> >& indexingFormulasPrev) {

  bool leaf = false;

  // Return back if we have generated all levels for loops
  if (loopIndex == numberOfNestedForLoops)
    return* this;

  if (loopIndex == numberOfNestedForLoops - 1)
    leaf = true;

  WorkItemSet currentWorkItemSet = forLoops.at(loopIndex);
  bool dependency = currentWorkItemSet.getDependency ();
  int numOfInstructions = currentWorkItemSet.getNumOfInstructions ();
  int numOfHomogenousWorkItems = currentWorkItemSet.getNumOfHomogenousWorkItems ();
	string formula = currentWorkItemSet.getFormula ();
  int vectorSize = currentWorkItemSet.getVectorSize ();
  int useLocalMem = currentWorkItemSet.getUseLocalMem ();

  if (loopIndex == 0) {
    oss << getIndent () << "// Just a private variable" << endl;
    if (vectorSize == 1)
      oss << getIndent () << "float temp = 1.0;" << endl;
    else
      oss << getIndent () << "float" << vectorSize << " temp = 1.0;" << endl;
    oss << endl;
  }

	if (dependency == false) {
    oss << endl;
    oss << getIndent () << "// Start of a new level of for loop" << endl;

		// We will first create the base index. This index is basically points
    // into the location of the beginning of the memory where the processing
    // should be performed. We will keep it in an string variable. This can
    // either be specified first and be used in formula, or specifically be
    // placed in each formula.
    // The definition depends on the depth of the for loop.
    // we only support loopIndex = 0, 1, 2 for now.
    //stringstream baseIndex;
    //if (loopIndex == 0) {
    //  baseIndex << "X";
    //} else if (loopIndex == 1) {
    //  if (forLoops.at(0).getDependency() == false)
    //  	baseIndex << "X*P + Y";
    //  else if (forLoops.at(0).getDependency() == true)
    //    baseIndex << "(X-1)*P + Y";
    //} else if (loopIndex == 2) {
    //  if (forLoops.at(0).getDependency() == false && forLoops.at(1).getDependency() == false)
    //  	baseIndex << "X*N*P + Y*P + Z";
    //  else if (forLoops.at(0).getDependency() == true && forLoops.at(1).getDependency() == false)
    //    baseIndex << "(X-1)*N*P + Y*P + Z";
    //  else if (forLoops.at(0).getDependency() == false && forLoops.at(1).getDependency() == true)
    //    baseIndex << "X*N*P + (Y-1)*P + Z";
    //  else if (forLoops.at(0).getDependency() == true && forLoops.at(1).getDependency() == true)
    //    baseIndex << "(X-1)*N*P + (Y-1)*P + Z";
    //}
    stringstream baseIndex2;
    //baseIndex2 << "(";
    baseIndex2 << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex);
    //baseIndex2 << ") * " << memAllocationPerWorkItem;
    //baseIndex2 << "(" << baseIndex.str() << ")" << " * " << memAllocationPerWorkItem;
		if (numOfInstructions != 0)
    	oss << getIndent() << "int baseIndex" << loopIndex+1 << " = " << baseIndex2.str() << ";" << endl;
    CircularNumberGenerator CNG (memAllocationPerWorkItem);
		for (int i = 0; i < numOfInstructions; i++) {
      string formulaRepl (formula);
      stringstream replacement;
      // This part will replace the index in GIn with the index that
      // should be really accessed in this work item.

      replacement << "baseIndex" << loopIndex+1 << " + " << CNG.next() << "*" << memAllocationPerWorkItem;
			replace (formulaRepl, "@", replacement.str());
      oss << getIndent () << formulaRepl << ";" << endl;
    }

    // if this is the last nested for loop, we will go on and write the
    // calculated value into the memory.
    if (leaf == true) {
      if (vectorSize == 1)
        oss << getIndent () << "GOut["
            << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
            << "] = temp;" << endl;
      else {
        oss << getIndent () << "GOut["
            << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
            << "] = temp.s0";
        for (int i = 1; i < vectorSize; i++) {
          if (i < 10)
            oss << " + " << "temp.s" << i;
          else if (i == 10)
            oss << " + " << "temp.sA";
          else if (i == 11)
            oss << " + " << "temp.sB";
          else if (i == 12)
            oss << " + " << "temp.sC";
          else if (i == 13)
            oss << " + " << "temp.sD";
          else if (i == 14)
            oss << " + " << "temp.sE";
          else if (i == 15)
            oss << " + " << "temp.sF";
        }
        oss << ";" << endl;
      }
    }

    // Now it's time to recursively call the single for algorithm,
    // in order to generate the next block of work items.
    generateSingleForV2 (loopIndex+1, indexingFormulas, indexingFormulasPrev);

  } else if (dependency == true){
    oss << endl;

    oss << getIndent () << "// Start of a new level of for loop" << endl;
		if (loopIndex == 0) {
			oss << getIndent () << "for (int Z = 0; Z < " << numOfHomogenousWorkItems << "; Z++){" << endl;
      currentIndentation++;
    }
    else if (loopIndex == 1) {
      oss << getIndent () << "for (int Y = 0; Y < " << numOfHomogenousWorkItems << "; Y++){" << endl;
      currentIndentation++;
    }
    else if (loopIndex == 2) {
      oss << getIndent () << "for (int X = 0; X < " << numOfHomogenousWorkItems << "; X++){" << endl;
			currentIndentation++;
    }


 		// We will first create the base index. This index is basically points
    // into the location of the beginning of the memory where the processing
    // should be performed. We will keep it in an string variable. This can
    // either be specified first and be used in formula, or specifically be
    // placed in each formula.
    // The definition depends on the depth of the for loop.
    // we only support loopIndex = 0, 1, 2 for now.
    //sringstream baseIndex;
    //sringstream baseIndexPrev;
    //if (loopIndex == 0) {
    //	baseIndex << "X";
    //  baseIndexPrev << "X-1";
    //} else if (loopIndex == 1) {
    //  if (forLoops.at(0).getDependency() == false) {
    //  	baseIndex << "X*P + Y";
    //    baseIndexPrev << "X*P + (Y-1)";
    //  } else if (forLoops.at(0).getDependency() == true) {
    //    baseIndex << "(X-1)*P + Y";
    //    baseIndexPrev << "(X-1)*P + (Y-1)";
    //  }
    //} else if (loopIndex == 2) {
    //  if (forLoops.at(0).getDependency() == false && forLoops.at(0).getDependency() == false) {
    //  	baseIndex << "X*N*P + Y*P + Z";
    //    baseIndexPrev << "X*N*P + Y*P + (Z-1)";
    //  } else if (forLoops.at(0).getDependency() == true && forLoops.at(0).getDependency() == false) {
    //    baseIndex << "(X-1)*N*P + Y*P + Z";
    //    baseIndexPrev << "(X-1)*N*P + Y*P + (Z-1)";
    //  } else if (forLoops.at(0).getDependency() == false && forLoops.at(0).getDependency() == true) {
    //    baseIndex << "X*N*P + (Y-1)*P + Z";
    //    baseIndexPrev << "X*N*P + (Y-1)*P + (Z-1)";
    //  } else if (forLoops.at(0).getDependency() == true && forLoops.at(0).getDependency() == true) {
    //    baseIndex << "(X-1)*N*P + (Y-1)*P + Z";
    //    baseIndexPrev << "(X-1)*N*P + (Y-1)*P + (Z-1)";
    //  }
    //}
    stringstream baseIndex2;
    stringstream baseIndexPrev2;
    //baseIndex2 << "(";
    baseIndex2 << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex);
    //baseIndex2 << ") * " << memAllocationPerWorkItem;
    //baseIndex2 << "(" << baseIndex.str() << ")" << " * " << memAllocationPerWorkItem;
    //baseIndexPrev2 << "(";
    baseIndexPrev2 << indexingFormulasPrev.at(numberOfNestedForLoops-1).at(loopIndex);
    //baseIndexPrev2 << ") * " << memAllocationPerWorkItem;
		//seIndexPrev2 << "(" << baseIndexPrev.str() << ")" << " * " << memAllocationPerWorkItem;

    oss << getIndent() << "int baseIndex" << loopIndex+1 << " = " << baseIndex2.str() << ";" << endl;
    oss << getIndent() << "int baseIndexPrev" << loopIndex+1 << " = " << baseIndexPrev2.str() << ";" << endl;


    CircularNumberGenerator CNG (memAllocationPerWorkItem);
    for (int i = 0; i < numOfInstructions; i++) {
      string formulaRepl (formula);
			stringstream replacementIdx1;
      stringstream replacementIdx2;

      // This part will replace the current index and dependent index
      // in GIn with the index thart should be really accessed in this
      // work item.

      int cngNext = CNG.next ();
			replacementIdx1 << "baseIndex" << loopIndex+1 << " + " << cngNext << "*" << memAllocationPerWorkItem;
      replacementIdx2 << "baseIndexPrev" << loopIndex+1 << " + " << cngNext << "*" << memAllocationPerWorkItem;
      replace (formulaRepl, "@", replacementIdx1.str());
      replace (formulaRepl, "!", replacementIdx2.str());
      oss << getIndent() << formulaRepl << ";" << endl;
    }
    // if this is the last nested for loop, we will go on and write the
    // calculated value into the memory.
    if (leaf == true) {
      if (vectorSize == 1)
        oss << getIndent () << "GOut["
            << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
            << "] = temp;" << endl;
      else {
        oss << getIndent () << "GOut["
            << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
            << "] = temp.s0";
        for (int i = 1; i < vectorSize; i++) {
          if (i < 10)
            oss << " + " << "temp.s" << i;
          else if (i == 10)
            oss << " + " << "temp.sA";
          else if (i == 11)
            oss << " + " << "temp.sB";
          else if (i == 12)
            oss << " + " << "temp.sC";
          else if (i == 13)
            oss << " + " << "temp.sD";
          else if (i == 14)
            oss << " + " << "temp.sE";
          else if (i == 15)
            oss << " + " << "temp.sF";
        }
        oss << ";" << endl;
      }
    }

    // Now it's time to recursively call the single for algorithm,
    // in order to generate the next block of work items.
    generateSingleForV2 (loopIndex+1, indexingFormulas, indexingFormulasPrev);

    currentIndentation--;
    oss << getIndent() << "}" << endl;

  }

	return *this;

}

Algorithm& Algorithm::generateSingleForV3 (int loopIndex,
                                           vector<vector<string> >& indexingFormulas,
                                           vector<vector<string> >& indexingFormulasPrev) {

	bool leaf = false;

	int totalGroupSize = 1;
  int tempNONFR = numberOfNestedForLoops;
  for (int i = 0; i < numberOfNestedForLoops; i++) {
		if (forLoops.at(i).getDependency() == true) tempNONFR--;
  }
  for (int i = 0; i < tempNONFR; i++)
    totalGroupSize *= localWorkSize[i];

	// Return back if we have generated all levels for loops
	if (loopIndex == numberOfNestedForLoops)
    return *this;

  if (loopIndex == numberOfNestedForLoops - 1)
    leaf = true;

	WorkItemSet currentWorkItemSet = forLoops.at(loopIndex);
  bool dependency = currentWorkItemSet.getDependency ();
  int numOfInstructions = currentWorkItemSet.getNumOfInstructions ();
  int numOfHomogenousWorkItems = currentWorkItemSet.getNumOfHomogenousWorkItems ();
  string formula = currentWorkItemSet.getFormula ();
  int vectorSize = currentWorkItemSet.getVectorSize ();
  int useLocalMem = currentWorkItemSet.getUseLocalMem ();

  if (loopIndex == 0) {
    oss << getIndent () << "// Just a private variable" << endl;
    if (vectorSize == 1)
      oss << getIndent () << "float temp = 1.0;" << endl;
    else
      oss << getIndent () << "float" << vectorSize << " temp = 1.0;" << endl;
    oss << endl;
  }

  if (dependency == false) {
    oss << endl;
    oss << getIndent () << "// Start of a new level of for loop" << endl;

    stringstream baseIndex2;
		baseIndex2 << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex);

    if (numOfInstructions != 0)
      oss << getIndent () << "int baseIndex" << loopIndex+1 << " = " << baseIndex2.str() << ";" << endl;
    CircularNumberGenerator CNG (memAllocationPerWorkItem);
    for (int i = 0; i < numOfInstructions; i++) {
      string formulaRepl (formula);
      stringstream replacement;
			// This part will replace the index in GIn with the index that
      // should be really accessed in this work item.

      replacement << "baseIndex" << loopIndex+1 << " + " << CNG.next() << "*" << totalGroupSize;
      replace (formulaRepl, "@", replacement.str());
      oss << getIndent () << formulaRepl << ";" << endl;
    }


    // If this is the last nested for loop, we will go on and write the
    // calculated value into the memory.
    if (leaf == true) {
      if (vectorSize == 1) {
        oss << getIndent () << "GOut["
            << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
            << "] = temp;" << endl;
      } else {
        oss << getIndent () << "GOut["
            << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
            << "] = temp.s0";
        for (int i = 1; i < vectorSize; i++) {
          if (i < 10)
						oss << " + " << "temp.s" << i;
          else if (i == 10)
            oss << " + " << "temp.sA";
          else if (i == 11)
            oss << " + " << "temp.sB";
          else if (i == 12)
            oss << " + " << "temp.sC";
          else if (i == 13)
            oss << " + " << "temp.sD";
          else if (i == 14)
            oss << " + " << "temp.sE";
          else if (i == 15)
            oss << " + " << "temp.sF";
      	}
        oss << ";" << endl;
      }
  	}

    // Now it's time to recursively call the single for algorithm,
    // in order to generate the next block of work items.
    generateSingleForV3 (loopIndex+1, indexingFormulas, indexingFormulasPrev);

  } else if (dependency == true) {
    oss << endl;

    oss << getIndent () << "// Start of a new level of for loop" << endl;
    if (loopIndex == 0) {
      oss << getIndent () << "for (int Z = 0; Z < " << numOfHomogenousWorkItems << "; Z++){" << endl;
      currentIndentation++;
    } else if (loopIndex == 1) {
      oss << getIndent () << "for (int Y = 0; Y < " << numOfHomogenousWorkItems << "; Y++){" << endl;
      currentIndentation++;
    } else if (loopIndex == 2) {
      oss << getIndent () << "for (int X = 0; X < " << numOfHomogenousWorkItems << "; X++){" << endl;
      currentIndentation++;
    }

    stringstream baseIndex2;
    stringstream baseIndexPrev2;
    baseIndex2 << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex);
    baseIndexPrev2 << indexingFormulasPrev.at(numberOfNestedForLoops-1).at(loopIndex);

    oss << getIndent () << "int baseIndex" << loopIndex+1 << " = " << baseIndex2.str() << ";" << endl;
    oss << getIndent () << "int baseIndexPrev" << loopIndex+1 << " = " << baseIndexPrev2.str() << ";" << endl;

		CircularNumberGenerator CNG (memAllocationPerWorkItem);
    for (int i = 0; i < numOfInstructions; i++) {
			string formulaRepl (formula);
      stringstream replacementIdx1;
      stringstream replacementIdx2;

      // This part will replace the current index and dependent index
      // in GIn with the index that should be really accessed in this
      // work item.

      int cngNext = CNG.next();
      replacementIdx1 << "baseIndex" << loopIndex+1 << " + " << cngNext << "*" << totalGroupSize;
      replacementIdx2 << "baseIndexPrev" << loopIndex+1 << " + " << cngNext << "*" << totalGroupSize;
			replace (formulaRepl, "@", replacementIdx1.str());
      replace (formulaRepl, "!", replacementIdx2.str());
      oss << getIndent () << formulaRepl << ";" << endl;
    }

    // If this is the last nested for loop, we will go on and write the
    // calculated value into the memory.
    if (leaf == true) {
      if (vectorSize == 1)
        oss << getIndent () << "GOut["
            << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
            << "] = temp;" << endl;
      else {
        oss << getIndent () << "GOut["
            << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
            << "] = temp.s0";
        for (int i = 0; i < vectorSize; i++) {
          if (i < 10)
            oss << " + " << "temp.s" << i;
          else if (i == 10)
            oss << " + " << "temp.sA";
          else if (i == 11)
            oss << " + " << "temp.sB";
          else if (i == 12)
            oss << " + " << "temp.sC";
          else if (i == 13)
            oss << " + " << "temp.sD";
          else if (i == 14)
            oss << " + " << "temp.sE";
          else if (i == 15)
            oss << " + " << "temp.sF";
        }
        oss << ";" << endl;
      }
    }

    // Now it's time to recursively call the single for algorithm,
    // in order to generate the next block of work items.
		generateSingleForV3 (loopIndex+1, indexingFormulas, indexingFormulasPrev);

    currentIndentation--;
    oss << getIndent () << "}" << endl;

  }

  return *this;

}

Algorithm& Algorithm::generateSingleForSimpleV1 (int loopIndex,
                                           vector<vector<string> >& indexingFormulas,
                                           vector<vector<string> >& indexingFormulasPrev,
                                           vector<vector<string> >& indexingLocalMem) {

	bool leaf = false;

	int totalGroupSize = 1;
  int tempNONFR = numberOfNestedForLoops;
  for (int i = 0; i < numberOfNestedForLoops; i++) {
		if (forLoops.at(i).getDependency() == true) tempNONFR--;
  }
  for (int i = 0; i < tempNONFR; i++)
    totalGroupSize *= localWorkSize[i];

	// Return back if we have generated all levels for loops
	if (loopIndex == numberOfNestedForLoops)
    return *this;

  if (loopIndex == numberOfNestedForLoops - 1)
    leaf = true;

	WorkItemSet currentWorkItemSet = forLoops.at(loopIndex);
  bool dependency = currentWorkItemSet.getDependency ();
  int numOfInstructions = currentWorkItemSet.getNumOfInstructions ();
  int numOfHomogenousWorkItems = currentWorkItemSet.getNumOfHomogenousWorkItems ();
  string formula = currentWorkItemSet.getFormula ();
  int vectorSize = currentWorkItemSet.getVectorSize ();
  int useLocalMem = currentWorkItemSet.getUseLocalMem ();

  if (loopIndex == 0) {
    oss << getIndent () << "// Just a private variable" << endl;
    if (vectorSize == 1) {
      oss << getIndent () << "float MF = (float) XGL;" << endl;
      oss << getIndent () << "float NF = (float) N;" << endl;
			oss << getIndent () << "float PF = (float) P;" << endl;
    }
    else {
      oss << getIndent () << "float MF = (float) XGL;" << endl;
      oss << getIndent () << "float NF = (float) N;" << endl;
      oss << getIndent () << "float PF = (float) P;" << endl;
    }
    oss << endl;
  }

  if (dependency == false) {
    oss << endl;
    oss << getIndent () << "// Start of a new level of for loop" << endl;

    stringstream baseIndex2;
		baseIndex2 << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex);

    if (numOfInstructions != 0) {
      oss << getIndent () << "long baseIndex" << loopIndex+1 << " = " << baseIndex2.str() << ";" << endl;
    }

    // This part will participate in copying data from global memory into the local
    // memory. We try to consider the memory coalescing access to reduce the global
		// memory access as much as possible.
		// First we define the local memory here
    int workGroupSize = 1;
    for (int i = 0; i < numberOfNestedForLoops; i++) {
      workGroupSize *= localWorkSize[i];
    }

    if (numOfInstructions != 0) {
      if (useLocalMem && algorithmTargetDevice != AlgorithmTargetDevice::FPGA) {
				oss << getIndent () << "__local float GInL[" << workGroupSize * memAllocationPerWorkItem << "];" << endl;

				oss << getIndent () << "for (int i = 0; i < " << memAllocationPerWorkItem << "; i++) {" << endl;
				this->currentIndentation++;
				oss << getIndent () << "GInL["
        		<< indexingLocalMem.at(numberOfNestedForLoops-1).at(loopIndex)
        		<< "+" << "i*" << workGroupSize << "]"
        		<< " = " << "GIn["
      			<< "baseIndex" << loopIndex+1 << "+i*" << workGroupSize
        		<< "];" << endl;
   			this->currentIndentation--;
    		oss << getIndent () << "}" << endl << endl;

				oss << getIndent () << "baseIndex" << loopIndex+1 << " = "
        		<< indexingLocalMem.at(numberOfNestedForLoops-1).at(loopIndex) << ";" << endl;
      }
    }

    if (vectorSize == 1) {
      oss << getIndent () << "float temp1 = 1.0;" << endl;
      oss << getIndent () << "float temp2 = 1.0;" << endl;
      oss << getIndent () << "float temp3 = 1.0;" << endl;
      oss << getIndent () << "float temp4 = 1.0;" << endl;
      oss << getIndent () << "float tempOut;" << endl;
    } else {
      oss << getIndent () << "float" << vectorSize << " temp1 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " temp2 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " temp3 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " temp4 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " tempOut;" << endl;
    }

    CircularNumberGenerator CNG (memAllocationPerWorkItem);
    for (int i = 0; i < numOfInstructions; i++) {
      string formulaRepl (formula);
      stringstream replacement;
			// This part will replace the index in GIn with the index that
      // should be really accessed in this work item.

      replacement << "baseIndex" << loopIndex+1 << " + " << CNG.next() << "*" << workGroupSize;
      replace (formulaRepl, "@", replacement.str());
      replace (formulaRepl, "@", replacement.str());
      if (useLocalMem && algorithmTargetDevice != AlgorithmTargetDevice::FPGA) {
      	replace (formulaRepl, "GIn", "GInL");
      }
      oss << getIndent () << formulaRepl << ";" << endl;
    }


    // If this is the last nested for loop, we will go on and write the
    // calculated value into the memory.
    if (leaf == true) {
      if (vectorSize == 1) {
				oss << getIndent () << "tempOut = temp1 + temp2 + temp3 + temp4;" << endl;
        oss << getIndent () << "GOut["
            << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
            << "] = tempOut;" << endl;
      } else {
        oss << getIndent () << "tempOut = temp1 + temp2 + temp3 + temp4;" << endl;
        oss << getIndent () << "GOut["
            << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
            << "] = tempOut.s0";
        for (int i = 1; i < vectorSize; i++) {
          if (i < 10)
						oss << " + " << "tempOut.s" << i;
          else if (i == 10)
            oss << " + " << "tempOut.sA";
          else if (i == 11)
            oss << " + " << "tempOut.sB";
          else if (i == 12)
            oss << " + " << "tempOut.sC";
          else if (i == 13)
            oss << " + " << "tempOut.sD";
          else if (i == 14)
            oss << " + " << "tempOut.sE";
          else if (i == 15)
            oss << " + " << "tempOut.sF";
      	}
        oss << ";" << endl;
      }
  	}

    // Now it's time to recursively call the single for algorithm,
    // in order to generate the next block of work items.
    generateSingleForSimpleV1 (loopIndex+1, indexingFormulas, indexingFormulasPrev, indexingLocalMem);

  } else if (dependency == true) {
    oss << endl;

    oss << getIndent () << "// Start of a new level of for loop" << endl;

    // This part will participate in copying data from global memory into the local
    // memory. We try to consider the memory coalescing access to reduce the global
		// memory access as much as possible.
		// First we define the local memory here
    int workGroupSize = 1;
    for (int i = 0; i < numberOfNestedForLoops; i++) {
			workGroupSize *= virtualLocalWorkSize[i];
      //workGroupSize *= localWorkSize[i];
    }
    if (numOfInstructions != 0) {
      if (useLocalMem && algorithmTargetDevice != AlgorithmTargetDevice::FPGA) {
				oss << getIndent () << "__local float GInL[" << workGroupSize * memAllocationPerWorkItem << "];" << endl;
      }
    }

    if ((numberOfNestedForLoops-loopIndex-1) == 2) {
      oss << getIndent () << "for (int Z = 0; Z < " << numOfHomogenousWorkItems << "; Z++){" << endl;
      currentIndentation++;
      oss << getIndent () << "const int ZGRid = Z/M;" << endl;
      oss << getIndent () << "const int ZLid = Z%M;" << endl;
    } else if ((numberOfNestedForLoops-loopIndex-1) == 1) {
      oss << getIndent () << "for (int Y = 0; Y < " << numOfHomogenousWorkItems << "; Y++){" << endl;
      currentIndentation++;
      oss << getIndent () << "const int YGRid = Y/N;" << endl;
      oss << getIndent () << "const int YLid = Y%N;" << endl;
    } else if ((numberOfNestedForLoops-loopIndex-1) == 0) {
      oss << getIndent () << "for (int X = 0; X < " << numOfHomogenousWorkItems << "; X++){" << endl;
      currentIndentation++;
      oss << getIndent () << "const int XGRid = X/P;" << endl;
      oss << getIndent () << "const int XLid = X%P;" << endl;
    }

    stringstream baseIndex2;
    stringstream baseIndexPrev2;
    baseIndex2 << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex);
    baseIndexPrev2 << indexingFormulasPrev.at(numberOfNestedForLoops-1).at(loopIndex);

    oss << getIndent () << "long baseIndex" << loopIndex+1 << " = " << baseIndex2.str() << ";" << endl;
    oss << getIndent () << "long baseIndexPrev" << loopIndex+1 << " = " << baseIndexPrev2.str() << ";" << endl;

    if (numOfInstructions != 0) {
      if (useLocalMem && algorithmTargetDevice != AlgorithmTargetDevice::FPGA) {
        oss << getIndent () << "for (int i = 0; i < " << memAllocationPerWorkItem << "; i++) {" << endl;
        this->currentIndentation++;
        oss << getIndent () << "GInL["
            << indexingLocalMem.at(numberOfNestedForLoops-1).at(loopIndex)
            << "+" << "i*" << workGroupSize << "]"
            << " = " << "GIn["
            << "baseIndex" << loopIndex+1 << "+i*" << workGroupSize
            << "];" << endl;
        this->currentIndentation--;
        oss << getIndent () << "}" << endl << endl;

      }
    }

    if (numOfInstructions != 0) {
      if (useLocalMem && algorithmTargetDevice != AlgorithmTargetDevice::FPGA) {
				oss << getIndent () << "baseIndex" << loopIndex+1 << " = "
        		<< indexingLocalMem.at(numberOfNestedForLoops-1).at(loopIndex) << ";" << endl;
      }
    }

    if (vectorSize == 1) {
      oss << getIndent () << "float temp1 = 1.0;" << endl;
      oss << getIndent () << "float temp2 = 1.0;" << endl;
      oss << getIndent () << "float temp3 = 1.0;" << endl;
      oss << getIndent () << "float temp4 = 1.0;" << endl;
      oss << getIndent () << "float tempOut;" << endl;
    } else {
      oss << getIndent () << "float" << vectorSize << " temp1 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " temp2 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " temp3 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " temp4 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " tempOut;" << endl;
    }

		CircularNumberGenerator CNG (memAllocationPerWorkItem);
    for (int i = 0; i < numOfInstructions; i++) {
			string formulaRepl (formula);
      stringstream replacementIdx1;
      stringstream replacementIdx2;

      // This part will replace the current index and dependent index
      // in GIn with the index that should be really accessed in this
      // work item.

      int cngNext = CNG.next();
      replacementIdx1 << "baseIndex" << loopIndex+1 << " + " << cngNext << "*" << totalGroupSize;
      replacementIdx2 << "baseIndexPrev" << loopIndex+1 << " + " << cngNext << "*" << totalGroupSize;
			replace (formulaRepl, "@", replacementIdx1.str());
      replace (formulaRepl, "@", replacementIdx1.str());
      replace (formulaRepl, "!", replacementIdx2.str());
      replace (formulaRepl, "!", replacementIdx2.str());
      if (useLocalMem && algorithmTargetDevice != AlgorithmTargetDevice::FPGA) {
      	replace (formulaRepl, "GIn", "GInL");
      }
      oss << getIndent () << formulaRepl << ";" << endl;
    }

    // If this is the last nested for loop, we will go on and write the
    // calculated value into the memory.
    if (leaf == true) {
      if (vectorSize == 1) {
				oss << getIndent () << "tempOut = temp1 + temp2 + temp3 + temp4;" << endl;
        oss << getIndent () << "GOut["
            << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
            << "] = tempOut;" << endl;
      } else {
        oss << getIndent () << "tempOut = temp1 + temp2 + temp3 + temp4;" << endl;
        oss << getIndent () << "GOut["
            << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
            << "] = tempOut.s0";
        for (int i = 0; i < vectorSize; i++) {
          if (i < 10)
            oss << " + " << "tempOut.s" << i;
          else if (i == 10)
            oss << " + " << "tempOut.sA";
          else if (i == 11)
            oss << " + " << "tempOut.sB";
          else if (i == 12)
            oss << " + " << "tempOut.sC";
          else if (i == 13)
            oss << " + " << "tempOut.sD";
          else if (i == 14)
            oss << " + " << "tempOut.sE";
          else if (i == 15)
            oss << " + " << "tempOut.sF";
        }
        oss << ";" << endl;
      }
    }

    // Now it's time to recursively call the single for algorithm,
    // in order to generate the next block of work items.
		generateSingleForSimpleV1 (loopIndex+1, indexingFormulas, indexingFormulasPrev, indexingLocalMem);

    currentIndentation--;
    oss << getIndent () << "}" << endl;

  }

  return *this;

}

Algorithm& Algorithm::generateSingleForSimpleV2 (int loopIndex,
                                           vector<vector<string> >& indexingFormulas,
                                           vector<vector<string> >& indexingFormulasPrev,
                                           vector<vector<string> >& indexingLocalMem) {

	bool leaf = false;

	int totalGroupSize = 1;
  int tempNONFR = numberOfNestedForLoops;
  for (int i = 0; i < numberOfNestedForLoops; i++) {
		if (forLoops.at(i).getDependency() == true) tempNONFR--;
  }
  for (int i = 0; i < tempNONFR; i++)
    totalGroupSize *= localWorkSize[i];

	// Return back if we have generated all levels for loops
	if (loopIndex == numberOfNestedForLoops)
    return *this;

  if (loopIndex == numberOfNestedForLoops - 1)
    leaf = true;

	WorkItemSet currentWorkItemSet = forLoops.at(loopIndex);
  bool dependency = currentWorkItemSet.getDependency ();
  int numOfInstructions = currentWorkItemSet.getNumOfInstructions ();
  int numOfHomogenousWorkItems = currentWorkItemSet.getNumOfHomogenousWorkItems ();
  string formula = currentWorkItemSet.getFormula ();
  int vectorSize = currentWorkItemSet.getVectorSize ();
  int useLocalMem = currentWorkItemSet.getUseLocalMem ();
  int loopCarriedDepDegree = currentWorkItemSet.getLoopCarriedDepDegree ();

  if (loopIndex == 0) {
    oss << getIndent () << "// Just a private variable" << endl;
    if (vectorSize == 1) {
      //oss << getIndent () << "float MF = (float) XGL;" << endl;
      oss << getIndent () << "float NF = (float) N;" << endl;
			oss << getIndent () << "float PF = (float) P;" << endl;
    }
    else {
      //oss << getIndent () << "float MF = (float) XGL;" << endl;
      oss << getIndent () << "float NF = (float) N;" << endl;
      oss << getIndent () << "float PF = (float) P;" << endl;
    }
    oss << endl;
  }

  if (dependency == false) {
    oss << endl;
    oss << getIndent () << "// Start of a new level of for loop" << endl;

    stringstream baseIndex2;
		baseIndex2 << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex);

    //if (numOfInstructions != 0) {
    //  oss << getIndent () << "long baseIndex" << loopIndex+1 << " = " << baseIndex2.str() << ";" << endl;
    //}

    // This part will participate in copying data from global memory into the local
    // memory. We try to consider the memory coalescing access to reduce the global
		// memory access as much as possible.
		// First we define the local memory here
    int workGroupSize = 1;
    for (int i = 0; i < numberOfNestedForLoops; i++) {
      workGroupSize *= localWorkSize[i];
    }

    if (numOfInstructions != 0) {
      if (useLocalMem && algorithmTargetDevice != AlgorithmTargetDevice::FPGA) {
				oss << getIndent () << "__local float GInL[" << workGroupSize * memAllocationPerWorkItem << "];" << endl;

				oss << getIndent () << "for (int i = 0; i < " << memAllocationPerWorkItem << "; i++) {" << endl;
				this->currentIndentation++;
				oss << getIndent () << "GInL["
        		<< indexingLocalMem.at(numberOfNestedForLoops-1).at(loopIndex)
        		<< "+" << "i*" << workGroupSize << "]"
        		<< " = " << "GIn["
      			<< "baseIndex" << loopIndex+1 << "+i*" << workGroupSize
        		<< "];" << endl;
   			this->currentIndentation--;
    		oss << getIndent () << "}" << endl << endl;

				//oss << getIndent () << "baseIndex" << loopIndex+1 << " = "
        //		<< indexingLocalMem.at(numberOfNestedForLoops-1).at(loopIndex) << ";" << endl;
      }
    }

		if (loopCarriedDepDegree != 1) {
      oss << getIndent () << "for (int lcdd = 0; lcdd < " << loopCarriedDepDegree << "; lcdd++) {" << endl;
      this->currentIndentation++;
    }

    if (vectorSize == 1) {
      oss << getIndent () << "float temp1 = 1.0;" << endl;
      oss << getIndent () << "float temp2 = 1.0;" << endl;
      oss << getIndent () << "float temp3 = 1.0;" << endl;
      oss << getIndent () << "float temp4 = 1.0;" << endl;
      oss << getIndent () << "float MF = (float) lcdd;" << endl;
      oss << getIndent () << "float tempOut;" << endl;
    } else {
      oss << getIndent () << "float" << vectorSize << " temp1 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " temp2 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " temp3 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " temp4 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " MF = (float) lcdd;" << endl;
      oss << getIndent () << "float" << vectorSize << " tempOut;" << endl;
    }

    CircularNumberGenerator CNG (memAllocationPerWorkItem);
    for (int i = 0; i < numOfInstructions; i++) {
      string formulaRepl (formula);
      stringstream replacement;
			// This part will replace the index in GIn with the index that
      // should be really accessed in this work item.

      replacement << "baseIndex" << loopIndex+1 << " + " << CNG.next() << "*" << workGroupSize;
      replace (formulaRepl, "@", replacement.str());
      replace (formulaRepl, "@", replacement.str());
      if (useLocalMem && algorithmTargetDevice != AlgorithmTargetDevice::FPGA) {
      	replace (formulaRepl, "GIn", "GInL");
      }
      oss << getIndent () << formulaRepl << ";" << endl;
    }


    // If this is the last nested for loop, we will go on and write the
    // calculated value into the memory.
    if (leaf == true) {
      if (vectorSize == 1) {
				oss << getIndent () << "tempOut = temp1 + temp2 + temp3 + temp4;" << endl;
        oss << getIndent () << "GOut["
          	//<< indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
          	<< "2*XGL+lcdd"
            << "] = tempOut;" << endl;
      } else {
        oss << getIndent () << "tempOut = temp1 + temp2 + temp3 + temp4;" << endl;
        oss << getIndent () << "GOut["
          	//<< indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
          	<< "2*XGL+lcdd"
            << "] = tempOut.s0";
        for (int i = 1; i < vectorSize; i++) {
          if (i < 10)
						oss << " + " << "tempOut.s" << i;
          else if (i == 10)
            oss << " + " << "tempOut.sA";
          else if (i == 11)
            oss << " + " << "tempOut.sB";
          else if (i == 12)
            oss << " + " << "tempOut.sC";
          else if (i == 13)
            oss << " + " << "tempOut.sD";
          else if (i == 14)
            oss << " + " << "tempOut.sE";
          else if (i == 15)
            oss << " + " << "tempOut.sF";
      	}
        oss << ";" << endl;
      }
  	}

    if (loopCarriedDepDegree != 1) {
      this->currentIndentation--;
      oss << getIndent () << "}" << endl;
      oss << endl;
    }

    // Now it's time to recursively call the single for algorithm,
    // in order to generate the next block of work items.
    generateSingleForSimpleV2 (loopIndex+1, indexingFormulas, indexingFormulasPrev, indexingLocalMem);

  } else if (dependency == true) {
    oss << endl;

    oss << getIndent () << "// Start of a new level of for loop" << endl;

    // This part will participate in copying data from global memory into the local
    // memory. We try to consider the memory coalescing access to reduce the global
		// memory access as much as possible.
		// First we define the local memory here
    int workGroupSize = 1;
    for (int i = 0; i < numberOfNestedForLoops; i++) {
			workGroupSize *= virtualLocalWorkSize[i];
      //workGroupSize *= localWorkSize[i];
    }
    if (numOfInstructions != 0) {
      if (useLocalMem && algorithmTargetDevice != AlgorithmTargetDevice::FPGA) {
				oss << getIndent () << "__local float GInL[" << workGroupSize * memAllocationPerWorkItem << "];" << endl;
      }
    }

    if ((numberOfNestedForLoops-loopIndex-1) == 2) {
      oss << getIndent () << "for (int Z = 0; Z < " << numOfHomogenousWorkItems << "; Z++){" << endl;
      currentIndentation++;
      oss << getIndent () << "const int ZGRid = Z/M;" << endl;
      oss << getIndent () << "const int ZLid = Z%M;" << endl;
    } else if ((numberOfNestedForLoops-loopIndex-1) == 1) {
      oss << getIndent () << "for (int Y = 0; Y < " << numOfHomogenousWorkItems << "; Y++){" << endl;
      currentIndentation++;
      oss << getIndent () << "const int YGRid = Y/N;" << endl;
      oss << getIndent () << "const int YLid = Y%N;" << endl;
    } else if ((numberOfNestedForLoops-loopIndex-1) == 0) {
      oss << getIndent () << "for (int X = 0; X < " << numOfHomogenousWorkItems << "; X++){" << endl;
      currentIndentation++;
      oss << getIndent () << "const int XGRid = X/P;" << endl;
      oss << getIndent () << "const int XLid = X%P;" << endl;
    }

    stringstream baseIndex2;
    stringstream baseIndexPrev2;
    baseIndex2 << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex);
    baseIndexPrev2 << indexingFormulasPrev.at(numberOfNestedForLoops-1).at(loopIndex);

    oss << getIndent () << "long baseIndex" << loopIndex+1 << " = " << baseIndex2.str() << ";" << endl;
    oss << getIndent () << "long baseIndexPrev" << loopIndex+1 << " = " << baseIndexPrev2.str() << ";" << endl;

    if (numOfInstructions != 0) {
      if (useLocalMem && algorithmTargetDevice != AlgorithmTargetDevice::FPGA) {
        oss << getIndent () << "for (int i = 0; i < " << memAllocationPerWorkItem << "; i++) {" << endl;
        this->currentIndentation++;
        oss << getIndent () << "GInL["
            << indexingLocalMem.at(numberOfNestedForLoops-1).at(loopIndex)
            << "+" << "i*" << workGroupSize << "]"
            << " = " << "GIn["
            << "baseIndex" << loopIndex+1 << "+i*" << workGroupSize
            << "];" << endl;
        this->currentIndentation--;
        oss << getIndent () << "}" << endl << endl;

      }
    }

    if (numOfInstructions != 0) {
      if (useLocalMem && algorithmTargetDevice != AlgorithmTargetDevice::FPGA) {
				oss << getIndent () << "baseIndex" << loopIndex+1 << " = "
        		<< indexingLocalMem.at(numberOfNestedForLoops-1).at(loopIndex) << ";" << endl;
      }
    }

    if (vectorSize == 1) {
      oss << getIndent () << "float temp1 = 1.0;" << endl;
      oss << getIndent () << "float temp2 = 1.0;" << endl;
      oss << getIndent () << "float temp3 = 1.0;" << endl;
      oss << getIndent () << "float temp4 = 1.0;" << endl;
      oss << getIndent () << "float tempOut;" << endl;
    } else {
      oss << getIndent () << "float" << vectorSize << " temp1 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " temp2 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " temp3 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " temp4 = 1.0;" << endl;
      oss << getIndent () << "float" << vectorSize << " tempOut;" << endl;
    }

		CircularNumberGenerator CNG (memAllocationPerWorkItem);
    for (int i = 0; i < numOfInstructions; i++) {
			string formulaRepl (formula);
      stringstream replacementIdx1;
      stringstream replacementIdx2;

      // This part will replace the current index and dependent index
      // in GIn with the index that should be really accessed in this
      // work item.

      int cngNext = CNG.next();
      replacementIdx1 << "baseIndex" << loopIndex+1 << " + " << cngNext << "*" << totalGroupSize;
      replacementIdx2 << "baseIndexPrev" << loopIndex+1 << " + " << cngNext << "*" << totalGroupSize;
			replace (formulaRepl, "@", replacementIdx1.str());
      replace (formulaRepl, "@", replacementIdx1.str());
      replace (formulaRepl, "!", replacementIdx2.str());
      replace (formulaRepl, "!", replacementIdx2.str());
      if (useLocalMem && algorithmTargetDevice != AlgorithmTargetDevice::FPGA) {
      	replace (formulaRepl, "GIn", "GInL");
      }
      oss << getIndent () << formulaRepl << ";" << endl;
    }

    // If this is the last nested for loop, we will go on and write the
    // calculated value into the memory.
    if (leaf == true) {
      if (vectorSize == 1) {
				oss << getIndent () << "tempOut = temp1 + temp2 + temp3 + temp4;" << endl;
        oss << getIndent () << "GOut["
            << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
            << "] = tempOut;" << endl;
      } else {
        oss << getIndent () << "tempOut = temp1 + temp2 + temp3 + temp4;" << endl;
        oss << getIndent () << "GOut["
            << indexingFormulas.at(numberOfNestedForLoops-1).at(loopIndex)
            << "] = tempOut.s0";
        for (int i = 0; i < vectorSize; i++) {
          if (i < 10)
            oss << " + " << "tempOut.s" << i;
          else if (i == 10)
            oss << " + " << "tempOut.sA";
          else if (i == 11)
            oss << " + " << "tempOut.sB";
          else if (i == 12)
            oss << " + " << "tempOut.sC";
          else if (i == 13)
            oss << " + " << "tempOut.sD";
          else if (i == 14)
            oss << " + " << "tempOut.sE";
          else if (i == 15)
            oss << " + " << "tempOut.sF";
        }
        oss << ";" << endl;
      }
    }

    // Now it's time to recursively call the single for algorithm,
    // in order to generate the next block of work items.
		generateSingleForSimpleV2 (loopIndex+1, indexingFormulas, indexingFormulasPrev, indexingLocalMem);

    currentIndentation--;
    oss << getIndent () << "}" << endl;

  }

  return *this;

}


Algorithm& Algorithm::writeToFile (string fileName) {

  if (!onlyMeta) {
		ofstream ofs;
  	ofs.open (fileName.c_str());

  	ofs << oss.str() << endl;
  	ofs.close();

  }
  kernelLocation = fileName;

  return *this;

}

Algorithm& Algorithm::verboseKernel () {

	cout << oss.str() << endl;

  return *this;

}

Algorithm& Algorithm::popMetas () {

  // First: Calculate the total number of floating point
  // operations are being done in this phase.
	for (int i = 0; i < numberOfNestedForLoops; i++) {
    long long newFlops = 1;

		//cout << "--Loop #" << i <<": Numinstr = " << forLoops.at(i).getNumOfInstructions() << endl;
    //cout << "--Loop #" << i <<": NumHom = " << forLoops.at(i).getNumOfHomogenousWorkItems() << endl;

    newFlops *= ((long long)(forLoops.at(i).getNumOfInstructions())
                 * (long long)(forLoops.at(i).getNumOfHomogenousWorkItems()));
    for (int j = i-1; j >= 0; j--)
      newFlops *= ((long long)(forLoops.at(j).getNumOfHomogenousWorkItems()));

    if (forLoops.at(i).getFops() == -1) {
    	if (forLoops.at(i).getDependency() == true) {
      	newFlops *= (long long)2;
      }
    } else {
			newFlops *= forLoops.at(i).getFops();
    }

    totalNumFlops += newFlops;
  }

  //cout << "--vector size is " << vectorSize << endl;

  totalNumFlops *= (long long)vectorSize;
	//cout << "--Total num flops is: " << totalNumFlops << endl;
	// Second: Now let's calculate the global Work Size
  // Remember something: if the total number of global
  // work size cannot be divided by the local work size,
  // then it's gonna fail.
  globalWorkSize = new int[1];
  globalWorkSize[0] = 1;
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    if (forLoops.at(i).getDependency() == false)
      globalWorkSize[0] *= forLoops.at(i).getNumOfHomogenousWorkItems();
  }

	if (globalWorkSize[0] % 64 != 0 || globalWorkSize[0] % 128 != 0 || globalWorkSize[0] % 256 != 0)
    PERROR ("Cannot executed the kernel, since globalWorkSize is not dividable by localWorkSize!");

  // Calculating GIn Size here
	for (int i = 0; i < numberOfNestedForLoops; i++) {
    GInSize *= forLoops.at(i).getNumOfHomogenousWorkItems();
  }
  GInSize *= memAllocationPerWorkItem;

  if (memoryReuseFactor != 0) {
  	if (GInSize % memoryReuseFactor != 0)
      PERROR ("Total memory allocation of GIn should be divisible by the memory reuse factor size.");
		GInSize /= memoryReuseFactor;
  }

	// Calculating GOut Size here
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    GOutSize *= forLoops.at(i).getNumOfHomogenousWorkItems();
	}
  // TODO: This is not OK at all. should be fixed in near future,
  // by reducing the size of the output buffer
  GOutSize *= memAllocationPerWorkItem;

  if (memoryReuseFactor != 0) {
    if (GOutSize % memoryReuseFactor != 0)
      PERROR ("Total memory allocation of GOut should be divisible by the memory reuse factor size.");
    GOutSize /= memoryReuseFactor;
  }

  return *this;

}

Algorithm& Algorithm::verificationFunctionIs (void (*verify)
                                              (float* GOut,
                                               int GOutSize,
                                               float M,
                                               float N,
                                               float P,
                                               long LL,
                                               int localSize)) {

  // Do verification
  this->verify = verify;

  return *this;

}

Algorithm& Algorithm::popMetasV2 () {

	// First: Calculate the total number of floating point
  // operation are being done in this phase.
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    long long newFlops = 1;

    newFlops *= ((long long)(forLoops.at(i).getNumOfInstructions())
                 * (long long)(forLoops.at(i).getNumOfHomogenousWorkItems()));
    for (int j = i-1; j >= 0; j--)
      newFlops *= ((long long)(forLoops.at(j).getNumOfHomogenousWorkItems()));

    if (forLoops.at(i).getFops() == -1) {
    	if (forLoops.at(i).getDependency() == true) {
      	newFlops *= (long long)2;
      }
    } else {
      newFlops *= forLoops.at(i).getFops();
    }


    totalNumFlops += newFlops;
  }

  totalNumFlops *= (long long) vectorSize;

  // Let's initiate the global work size vector and also number of
  // dimensions

  workDim = 0;
	for (int i = 0; i < numberOfNestedForLoops; i++) {
    if (forLoops.at(i).getDependency() == false)
      workDim++;
  }

  int numNonDependentIters = 0;
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    if (forLoops.at(i).getDependency() == false)
      numNonDependentIters++;
  }
	globalWorkSize = new int[numNonDependentIters];
  int counter = 0;
	for (int i = 0; i < numberOfNestedForLoops; i++) {
    if (forLoops.at(i).getDependency() == false) {
      globalWorkSize[numNonDependentIters-counter-1] = forLoops.at(i).getNumOfHomogenousWorkItems();
      counter++;
    }
  }


  // Calculating values of M, N, and P
  if (numberOfNestedForLoops > 0) {
    M = forLoops.at(0).getNumOfHomogenousWorkItems();
    if (numberOfNestedForLoops > 1) {
      N = forLoops.at(1).getNumOfHomogenousWorkItems();
      if (numberOfNestedForLoops > 2) {
        P = forLoops.at(2).getNumOfHomogenousWorkItems();
      } else {
        P = 1;
      }
    } else {
      N = 1;
      P = 1;
    }
  } else {
    M = 1;
    N = 1;
    P = 1;
  }

	// Calculating GIn Size here
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    GInSize *= forLoops.at(i).getNumOfHomogenousWorkItems();
  }
  GInSize *= memAllocationPerWorkItem;

	// TODO: Memory Reuse Factor should be included here

  // Calculating GOut Size here
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    GOutSize *= forLoops.at(i).getNumOfHomogenousWorkItems();
  }

  // TODO: This is not OK at all. Should be fixed in near future,
  // by reducing the size of the ouput buffer
  GOutSize *= memAllocationPerWorkItem;

  // TODO: Memory Reuse Factor for GOut memory should be
  // included here

	isV2 = true;

  return *this;
}

Algorithm& Algorithm::popMetasV3 () {


	// First: Calculate the total number of floating point
  // operation are being done in this phase.
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    long long newFlops = 1;

    newFlops *= ((long long)(forLoops.at(i).getNumOfInstructions())
                 * (long long)(forLoops.at(i).getNumOfHomogenousWorkItems()));
    for (int j = i-1; j >= 0; j--)
      newFlops *= ((long long)(forLoops.at(j).getNumOfHomogenousWorkItems()));

    if (forLoops.at(i).getFops() == -1) {
    	if (forLoops.at(i).getDependency() == true) {
      	newFlops *= (long long)2;
      }
    } else {
      newFlops *= forLoops.at(i).getFops();
    }

    totalNumFlops += newFlops;
  }

  totalNumFlops *= (long long) vectorSize;

  // Let's initiate the global work size vector and also number of
  // dimensions

  workDim = 0;
	for (int i = 0; i < numberOfNestedForLoops; i++) {
    if (forLoops.at(i).getDependency() == false)
      workDim++;
	}

  int numNonDependentIters = 0;
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    if (forLoops.at(i).getDependency() == false)
      numNonDependentIters++;
  }
	globalWorkSize = new int[numNonDependentIters];
  int counter = 0;
	for (int i = 0; i < numberOfNestedForLoops; i++) {
    if (forLoops.at(i).getDependency() == false) {
      globalWorkSize[numNonDependentIters-counter-1] = forLoops.at(i).getNumOfHomogenousWorkItems();
      counter++;
    }
  }

  // Calculating values of M, N, and P
  if (numberOfNestedForLoops > 0) {
    M = forLoops.at(0).getNumOfHomogenousWorkItems();
    if (numberOfNestedForLoops > 1) {
      N = forLoops.at(1).getNumOfHomogenousWorkItems();
      if (numberOfNestedForLoops > 2) {
        P = forLoops.at(2).getNumOfHomogenousWorkItems();
      } else {
        P = 1;
      }
    } else {
      N = 1;
      P = 1;
    }
  } else {
    M = 1;
    N = 1;
    P = 1;
  }

	// Calculating GIn Size here
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    GInSize *= forLoops.at(i).getNumOfHomogenousWorkItems();
  }
  GInSize *= memAllocationPerWorkItem;

	// TODO: Memory Reuse Factor should be included here

  // Calculating GOut Size here
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    GOutSize *= forLoops.at(i).getNumOfHomogenousWorkItems();
  }

  // TODO: This is not OK at all. Should be fixed in near future,
  // by reducing the size of the ouput buffer
  GOutSize *= memAllocationPerWorkItem;

  // TODO: Memory Reuse Factor for GOut memory should be
  // included here

	isV2 = true;

  return *this;
}

Algorithm& Algorithm::popMetasSimpleV1 () {


	// First: Calculate the total number of floating point
  // operation are being done in this phase.
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    long long newFlops = 1;

    newFlops *= ((long long)(forLoops.at(i).getNumOfInstructions())
                 * (long long)(forLoops.at(i).getNumOfHomogenousWorkItems()));
    for (int j = i-1; j >= 0; j--)
      newFlops *= ((long long)(forLoops.at(j).getNumOfHomogenousWorkItems()));

    if (forLoops.at(i).getFops() == -1) {
    	if (forLoops.at(i).getDependency() == true) {
      	newFlops *= (long long)2;
      }
    } else {
      newFlops *= forLoops.at(i).getFops();
    }

    totalNumFlops += newFlops;
  }

  totalNumFlops *= (long long) vectorSize;

  // Let's initiate the global work size vector and also number of
  // dimensions

  workDim = 0;
	for (int i = 0; i < numberOfNestedForLoops; i++) {
    if (forLoops.at(i).getDependency() == false)
      workDim++;
	}

  if (workDim == 0) workDim = 1;

  int numNonDependentIters = 0;
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    if (forLoops.at(i).getDependency() == false)
      numNonDependentIters++;
  }

  if (numNonDependentIters == 0)
    numNonDependentIters = 1;

	globalWorkSize = new int[numNonDependentIters];
  int counter = 0;
	for (int i = 0; i < numberOfNestedForLoops; i++) {
    if (forLoops.at(i).getDependency() == false) {
      globalWorkSize[numNonDependentIters-counter-1] = forLoops.at(i).getNumOfHomogenousWorkItems();
      counter++;
    }
  }

	if (counter == 0) {
    for (int i = 0; i < numNonDependentIters; i++) {
      globalWorkSize[i] = 1;
    }
  }

  // Calculating values of M, N, and P
  // if (numberOfNestedForLoops > 0) {
  //  M = forLoops.at(0).getNumOfHomogenousWorkItems();
  //  if (numberOfNestedForLoops > 1) {
  //    N = forLoops.at(1).getNumOfHomogenousWorkItems();
  //    if (numberOfNestedForLoops > 2) {
  //      P = forLoops.at(2).getNumOfHomogenousWorkItems();
  //    } else {
  //      P = 1;
  //    }
  //  } else {
  //    N = 1;
  //    P = 1;
  //  }
  //} else {
  //  M = 1;
  //  N = 1;
  //  P = 1;
  // }

  M = 1;
  N = 1;
  P = 1;
	for (int i = 0; i < numberOfNestedForLoops; i++) {
    if (i == 0) {
      P = getVirtualLocalWorkSize()[0];
    } else if (i == 1) {
			N = getVirtualLocalWorkSize()[1];
    } else if (i == 2) {
			M = getVirtualLocalWorkSize()[2];
    }
  }

	// Calculating GIn Size here
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    GInSize *= forLoops.at(i).getNumOfHomogenousWorkItems();
  }
  GInSize *= memAllocationPerWorkItem;

	// TODO: Memory Reuse Factor should be included here

  // Calculating GOut Size here
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    GOutSize *= forLoops.at(i).getNumOfHomogenousWorkItems();
  }

  // TODO: This is not OK at all. Should be fixed in near future,
  // by reducing the size of the ouput buffer
  GOutSize *= memAllocationPerWorkItem;

  // TODO: Memory Reuse Factor for GOut memory should be
  // included here

	isV2 = true;

  return *this;
}

Algorithm& Algorithm::popMetasSimpleV2 () {


	// First: Calculate the total number of floating point
  // operation are being done in this phase.
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    long long newFlops = 1;

    newFlops *= ((long long)(forLoops.at(i).getNumOfInstructions())
                 * (long long)(forLoops.at(i).getNumOfHomogenousWorkItems()));
    // This is a fast and simple hack for a basic checking of something. Should be changed immediately
    //newFlops *= ((long long)16) * (long long)(forLoops.at(i).getNumOfHomogenousWorkItems());
    for (int j = i-1; j >= 0; j--)
      newFlops *= ((long long)(forLoops.at(j).getNumOfHomogenousWorkItems()));

    if (forLoops.at(i).getFops() == -1) {
    	if (forLoops.at(i).getDependency() == true) {
      	newFlops *= (long long)2;
      }
    } else {
      newFlops *= forLoops.at(i).getFops();
    }

    totalNumFlops += newFlops;
  }

  totalNumFlops *= (long long) vectorSize;

  // Let's initiate the global work size vector and also number of
  // dimensions

  workDim = 0;
	for (int i = 0; i < numberOfNestedForLoops; i++) {
    if (forLoops.at(i).getDependency() == false)
      workDim++;
	}

  if (workDim == 0) workDim = 1;

  int numNonDependentIters = 0;
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    if (forLoops.at(i).getDependency() == false)
      numNonDependentIters++;
  }

  if (numNonDependentIters == 0)
    numNonDependentIters = 1;

	globalWorkSize = new int[numNonDependentIters];
  int counter = 0;
	for (int i = 0; i < numberOfNestedForLoops; i++) {
    if (forLoops.at(i).getDependency() == false) {
      globalWorkSize[numNonDependentIters-counter-1] = forLoops.at(i).getNumOfHomogenousWorkItems();
      if (forLoops.at(i).getLoopCarriedDepDegree() != 1)
        globalWorkSize[numNonDependentIters-counter-1] =
          globalWorkSize[numNonDependentIters-counter-1] / forLoops.at(i).getLoopCarriedDepDegree();
      counter++;
    }
  }

	if (counter == 0) {
    for (int i = 0; i < numNonDependentIters; i++) {
      globalWorkSize[i] = 1;
    }
  }

  // Calculating values of M, N, and P
  // if (numberOfNestedForLoops > 0) {
  //  M = forLoops.at(0).getNumOfHomogenousWorkItems();
  //  if (numberOfNestedForLoops > 1) {
  //    N = forLoops.at(1).getNumOfHomogenousWorkItems();
  //    if (numberOfNestedForLoops > 2) {
  //      P = forLoops.at(2).getNumOfHomogenousWorkItems();
  //    } else {
  //      P = 1;
  //    }
  //  } else {
  //    N = 1;
  //    P = 1;
  //  }
  //} else {
  //  M = 1;
  //  N = 1;
  //  P = 1;
  // }

  M = 1;
  N = 1;
  P = 1;
	for (int i = 0; i < numberOfNestedForLoops; i++) {
    if (i == 0) {
      P = getVirtualLocalWorkSize()[0];
    } else if (i == 1) {
			N = getVirtualLocalWorkSize()[1];
    } else if (i == 2) {
			M = getVirtualLocalWorkSize()[2];
    }
  }

	// Calculating GIn Size here
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    GInSize *= forLoops.at(i).getNumOfHomogenousWorkItems();
  }
  GInSize *= memAllocationPerWorkItem;

	// TODO: Memory Reuse Factor should be included here

  // Calculating GOut Size here
  for (int i = 0; i < numberOfNestedForLoops; i++) {
    GOutSize *= forLoops.at(i).getNumOfHomogenousWorkItems();
  }

  // TODO: This is not OK at all. Should be fixed in near future,
  // by reducing the size of the ouput buffer
  GOutSize *= memAllocationPerWorkItem;

  // TODO: Memory Reuse Factor for GOut memory should be
  // included here

	isV2 = true;

  return *this;
}

long long Algorithm::getGInSize () {
  return GInSize;
}

long long Algorithm::getGOutSize () {
  return GOutSize;
}

string Algorithm::getKernelLocation () {
  return kernelLocation;
}

int* Algorithm::nextLocalWorkSize () {
  if (currentLocalWorkSizeIndex == 0) {
    currentLocalWorkSizeIndex++;
    return localWorkSize;
  }
  else
    return NULL;
  /*
  if (currentLocalWorkSizeIndex != localWorkSize.size()) {
    currentLocalWorkSizeIndex++;
    return localWorkSize.at(currentLocalWorkSizeIndex-1);
  } else {
    return -1;
  }
  */
}

int* Algorithm::getGlobalWorkSize () {
  return globalWorkSize;
}

int* Algorithm::getVirtualLocalWorkSize () {
  return virtualLocalWorkSize;
}

long long Algorithm::getTotalNumFlops () {
  return totalNumFlops;
}

string Algorithm::getKernelName () {
  return currentKernelName;
}

string Algorithm::getName  () {
  return currentName;
}

int Algorithm::getWorkDim () {
  return workDim;
}

int Algorithm::getM () {
  return M;
}

int Algorithm::getN () {
  return N;
}

int Algorithm::getP () {
  return P;
}

bool Algorithm::getIsV2 () {
  return isV2;
}

int Algorithm::getAlgorithmTargetDevice () {
  return this->algorithmTargetDevice;
}

int Algorithm::getAlgorithmTargetLanguage () {
  return this->algorithmTargetLanguage;
}

Algorithm& Algorithm::verbose () {

  cout << "------- Algorithm Specification -------" << endl;

  cout << "$$ Target Device: ";
  if (algorithmTargetDevice == AlgorithmTargetDevice::FPGA)
    cout << "FPGA" << endl;
  else if (algorithmTargetDevice == AlgorithmTargetDevice::GPU)
    cout << "GPU" << endl;

  cout << "$$ Target Language: ";
  if (algorithmTargetLanguage == AlgorithmTargetLanguage::OpenCL)
    cout << "OpenCL" << endl;
  else if (algorithmTargetLanguage == AlgorithmTargetLanguage::CUDA)
    cout << "CUDA" << endl;

  cout << "$$ Memory Allocation Per Item: " << memAllocationPerWorkItem << endl;
  cout << "$$ Memory Reuse Factor: " << memoryReuseFactor << endl;
	cout << "$$ Number of Nested For Loops: " << numberOfNestedForLoops << endl;
  cout << "$$ Total Number of Floating Points Operations: "
       << totalNumFlops << endl;
	cout << "$$ Only Meta: " << onlyMeta << endl;

  for (int i = 0; i < numberOfNestedForLoops; i++) {
    cout << "$$ For Loop #" << i << ", Number of iterations: "
         << forLoops.at(i).getNumOfHomogenousWorkItems () << endl;
    cout << "$$ For Loop #" << i << ", Dependency: "
         << forLoops.at(i).getDependency () << endl;
    cout << "$$ For Loop #" << i << ", Number of instruction: "
         << forLoops.at(i).getNumOfInstructions () << endl;
    cout << "$$ For Loop #" << i << ", Formula: "
         << forLoops.at(i).getFormula () << endl;
    cout << "$$ For Loop #" << i << ", Vector Size: "
         << forLoops.at(i).getVectorSize () << endl;
    cout << "$$ For Loop #" << i << ", Use Local Memory: "
         << forLoops.at(i).getUseLocalMem () << endl;
  }

	cout << "$$ Global Work Size: [";
	for (int i = 0; i < workDim; i++) {
    cout << globalWorkSize[i] << ",";
  }
  cout << "]" << endl;

  cout << "$$ Local Work Size: [";
  for (int i = 0; i < workDim; i++) {
    cout << localWorkSize[i] << ",";
  }
	cout << "]" << endl;

	cout << "$$ M, N and P: " << M << "," << N << "," << P << endl;

	cout << "$$ Input Data Allocation Size: " << GInSize << endl;
	cout << "$$ Output Data Allocation Size: " << GOutSize << endl;

	return *this;

}
