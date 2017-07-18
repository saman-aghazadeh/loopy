#include <NumberGenerator.h>

NumberGenerator::NumberGenerator (int size) {
  this->size = size;
  for (int i = 0; i < size; i++)
    numbersIndices.push_back (true);

  srand ((unsigned) time(0));
}

NumberGenerator::~NumberGenerator () {}

int NumberGenerator::getNextSequential () {
  if (seqIndex < size) {
    numbersIndices[seqIndex] = false;
    seqIndex++;
    return seqIndex-1;
  }
}

int NumberGenerator::getNextRandom () {
  while (true) {
    int random = rand() % size;
    if (numbersIndices[random] == false) continue;
    else {
      numbersIndices[random] = false;
      return random;
    }
  }
}

