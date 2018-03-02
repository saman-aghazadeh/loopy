#ifndef ALGORITHMFACTORY_H_
#define ALGORITHMFACTORY_H_

#include "Algorithm.h"

#include <vector>
using namespace std;

class AlgorithmFactory {
public:
	AlgorithmFactory ();
  ~AlgorithmFactory ();

	Algorithm& createNewAlgorithm ();
 	Algorithm* nextAlgorithm();
  void resetIndex ();

private:
	vector<Algorithm*> algorithms;
	int index;

};

#endif // Algorithm Factory
