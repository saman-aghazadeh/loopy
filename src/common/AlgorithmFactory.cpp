#include "AlgorithmFactory.h"

AlgorithmFactory::AlgorithmFactory () {
  this->index = 0;
}

AlgorithmFactory::~AlgorithmFactory () {

}

Algorithm& AlgorithmFactory::createNewAlgorithm () {

  Algorithm *algorithm = new Algorithm;
  algorithms.push_back (algorithm);

  return *algorithm;

}

Algorithm* AlgorithmFactory::nextAlgorithm () {
  if (index != algorithms.size()) {
    index++;
    return algorithms.at(index-1);
  }
  else
    return NULL;
}

void AlgorithmFactory::resetIndex () {
  this->index = 0;
}
