#ifndef ALGORITHMTYPE_H_
#define ALGORITHMTYPE_H_

#include <vector>
#include <map>
#include <string>
using namespace std;

struct _algorithm_type {
	const char* name;
	int vectorSize;
	int numLoops;
	vector<int> loopsLengths;
	vector<int> loopsDepth;
	bool loopCarriedDataDependency;
	vector<int> loopCarriedDDLengths;
	const char* variable;
	const char* varDeclFormula;
	const char* varInitFormula;
	const char* returnFormula;
	const char* formula;
	int halfBufSizeMin;
	int halfBufSizeMax;
	int halfBufSizeStride;
	int localWorkSizeMin;
	int localWorkSizeMax;
	int localWorkSizeStride;
	int flopCount;
	const char* varType;
	bool doManualUnroll;
	bool doLocalMemory;
	int unrollFactor;
	int randomAccessType;
};

class RandomAccessType {
public:
	enum randomAccessType {SEQUENTIAL, SEMIRANDOM, RANDOM};
};


#endif // ALGORITHMTYPE_H_
