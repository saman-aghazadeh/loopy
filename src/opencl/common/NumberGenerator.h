#ifndef NUMBERGENERATOR_H_
#define NUMBERGENERATOR_H_

#include <time.h>
#include <stdlib.h>

#include <vector>
using namespace std;

class NumberGenerator {
public:

  NumberGenerator (int size);
  ~NumberGenerator ();

  int getNextSequential ();

  int getNextRandom ();

private:
  int size;
  vector<bool> numbersIndices;
  int seqIndex = 0;
};

#endif // NUMBERGENERATOR_H_
