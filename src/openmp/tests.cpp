#include "omp.h"
#include "stdio.h"
#include "funcs.h"
#include "OptionParser.h"
#include "ResultDatabase.h"

#include <chrono>
#include <iostream>
#include <sstream>
#include "stdlib.h"
#include "time.h"

using namespace std;
using namespace std::chrono;

#define GETTIME std::chrono::high_resolution_clock::now()

#define DIFFTIME(t2,t1) std::chrono::duration_cast<std::chrono::milliseconds>\
	(t2-t1).count()

#include "ConditionalDependent.h"
#include "AntiDependent.h"
#include "IntraDimensionDependent.h"

void runner(OptionParser &op, ResultDatabase &resultDB); 
int getAdjustedDataSize (unsigned long long dataSize);
void initializeBuffers (void *AA, void *BB, void *CC, 
			int sizeX, int sizeY);
int getAdjustedDataSize (unsigned long long dataSize);
void addBenchmarkSpecOptions (OptionParser &op); 

template <class T> inline std::string toString (const T& t) {
	
	std::stringstream ss;
	ss << t;
	return ss.str();

}
 
int main(int argc, char* argv[]) {

	srand(time(0));

	ResultDatabase resultDB;

	OptionParser op;
	addBenchmarkSpecOptions(op);

	if (!op.parse(argc, argv)) {
		op.usage();
		return (op.HelpRequested() ? 0 : 1);
	}

	runner(op, resultDB);
	resultDB.DumpDetailed(cout);

	return 0;

}

void addBenchmarkSpecOptions (OptionParser &op) {
        op.addOption ("min_data_size", OPT_INT, "0", "minimum data size (in Kilobytes)");
        op.addOption ("max_data_size", OPT_INT, "0", "maximum data size (in Kilobytes)");
	op.addOption ("pass", OPT_INT, "", "Number of passes");
}


void runner(OptionParser &op, ResultDatabase &resultDB) {
	
	int minDataSize = op.getOptionInt("min_data_size") * 1024 * 1024;
	int maxDataSize = op.getOptionInt("max_data_size") * 1024 * 1024;
	int passes = op.getOptionInt("pass");

	cout << "[INFO] Minimum Data Size is " << minDataSize << endl;
	cout << "[INFO] Maximum Data Size is " << maxDataSize << endl;

	if (minDataSize == 0 || maxDataSize == 0) return;

	for (unsigned long long dataSize = minDataSize; dataSize <= maxDataSize; dataSize *= 2) {

		int adjustedDataSize = getAdjustedDataSize (dataSize);
		for (int llly = 256; llly < adjustedDataSize/4; llly *= 2) {

			void *AA, *BB, *CC;

			int lllx = adjustedDataSize / llly;

			AA = (void *) malloc (dataSize + 4 * sizeof(DTYPE));
			BB = (void *) malloc (dataSize + 4 * sizeof(DTYPE));
			CC = (void *) malloc (dataSize + 4 * sizeof(DTYPE));

			initializeBuffers (AA, BB, CC, 
				llly, lllx);

			for (int iter = 0; iter < passes; iter++) {
	
#if IntraDimensionDependent
				ulong runtime = intraDimensionDependent(
					AA, BB, CC,
					llly, lllx);
				resultDB.AddResult("DTYPE", "lllX(Dep)" + toString(lllx) + "-lllY(Par)" + toString(llly), "Bytes", runtime);
#elif DiagonalDependent

#elif ConditionalDependent
				
#if Branch2
				ulong runtime = conditionalDependent2(
					AA, BB, CC,
					adjustedDataSize);
				resultDB.AddResult("DTYPE", "Size(Par)" + toString(adjustedDataSize), "Bytes", runtime);
#elif Branch4
				ulong runtime = conditionalDependent4(
					AA, BB, CC,
					adjustedDataSize);
				resultDB.AddResult("DTYPE", "Size(Par)" + toString(adjustedDataSize), "Bytes", runtime);
#elif Branch6
				ulong runtime = conditionalDependent6(
					AA, BB, CC,
					adjustedDataSize);
				resultDB.AddResult("DTYPE", "Size(Par)" + toString(adjustedDataSize), "Bytes", runtime);
#elif Branch8
				ulong runtime = conditionalDependent8(
					AA, BB, CC,
					adjustedDataSize);
				resultDB.AddResult("DTYPE", "Size(Par)" + toString(adjustedDataSize), "Bytes", runtime);
#endif

#elif AntiDependent

#if TWOSTAGE
				ulong runtime = AntiDependent2(
					AA, BB, CC,
					adjustedDataSize);
				resultDB.AddResult("DTYPE", "Size(Par)" + toString(adjustedDataSize), "Bytes", runtime);
#elif FOURSTAGE
				ulong runtime = AntiDependent4(
					AA, BB, CC,
					adjustedDataSize);
				resultDB.AddResult("DTYPE", "Size(Par)" + toString(adjustedDataSize), "Bytes", runtime);
#elif SIXSTAGE
				ulong runtime = AntiDependent6(
					AA, BB, CC,
					adjustedDataSize);
				resultDB.AddResult("DTYPE", "Size(Par)" + toString(adjustedDataSize), "Bytes", runtime);
#endif

#elif HalfParallelismHalfDependent
				ulong runtime1 = HalfParallelHalfDependentSec1(
					AA, BB, CC,
					adjustedDataSize);
				ulong runtime2 = HalfParallelHalfDependentSec2(
					AA, BB, CC,
					adjustedDataSize);
				resultDB.AddResult("DTYPE", "Size(Par)" + toString(adjustedDataSize), "Bytes", runtime);
#endif
			}
		}					
	}
}

int getAdjustedDataSize (unsigned long long dataSize) {
	
	int adjustedDataSize = 0;

	adjustedDataSize = dataSize / sizeof (DTYPE);
	cout << "[INFO] Adjusted Data Size is: " << adjustedDataSize << endl;

	return adjustedDataSize;

}

void initializeBuffers (void *AA, void *BB, void *CC, 
			int sizeX, int sizeY) {

	for (int i = 0; i < sizeX; i++) {
		for (int j = 0; j < sizeY; j++) {
			((DTYPE *) AA)[i*sizeY+j] = rand() % 100;
			((DTYPE *) BB)[i*sizeY+j] = rand() % 100;
			((DTYPE *) CC)[i*sizeY+j] = rand() % 100;
		}
	}
}
