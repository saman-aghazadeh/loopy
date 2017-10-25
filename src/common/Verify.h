#ifndef VERIFY_H_
#define VERIFY_H_

#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

string FILE_PREFIX = "/home/user/sbiookag/Algs/1For/nodep/GAP0-FloatParam/";

void verifyWGSXMAPIXLLXOPS1024GAP0 (float* GOut, int GOutSize, float M, float N, float P, long LL, int localSize) {

	ofstream output;
  string outName = FILE_PREFIX + "verifyWGSXMAPIXLLXOPS1024GAP0.txt";
  output.open (outName);

	cout << "Executing verification function verifyWGSXMAPIXLLXOPS1024GAP0" << endl;

	float* currentGOut = new float[GOutSize];

	for (int i = 0; i < LL; i++) {

		int XGL = i;
		int XGRid = i / localSize;
    int XGRnum = LL / localSize;
    int XLSize = localSize;
    int XLid = i % localSize;

		float temp1 = 1.0;
    float temp2 = 1.0;
    float temp3 = 1.0;
    float temp4 = 1.0;
    float tempOut;
    for (int j = 0; j < 1024; j++) {
			temp1 += temp1 * M;
    }

    tempOut = temp1 + temp2 + temp3 + temp4;
		currentGOut[XGRid*XLSize*2+XLid] = tempOut;
  }

  // For now just print out some initial values of both Gout and currentGOut

	cout << "Print out verification for case: WGS" << localSize << "MAPIXLL" << LL << "OPS1024" << endl;
  for (int i = 0; i < GOutSize; i++) {
    output << GOut[i] << " " << currentGOut[i] << endl;
    if (GOut[i] != currentGOut[i] ) {
 			cout << "inconsistence results on index " << i << " => " << GOut[i] << " " << currentGOut[i] << endl;
    }
  }
  output.close ();
  cout << endl;

}

void verifyWGSXMAPIXLLXOPS64GAP0 (float* GOut, int GOutSize, float M, float N, float P, long LL, int localSize) {

	ofstream output;
  string outName = FILE_PREFIX + "verifyWGSXMAPIXLLXOPS64GAP0.txt";
  output.open (outName);

	cout << "Executing verification function verifyWGSXMAPIXLLXOPS64GAP0" << endl;

	float* currentGOut = new float[GOutSize];

	for (int i = 0; i < LL; i++) {

		int XGL = i;
		int XGRid = i / localSize;
    int XGRnum = LL / localSize;
    int XLSize = localSize;
    int XLid = i % localSize;

		float temp1 = 1.0;
    float temp2 = 1.0;
    float temp3 = 1.0;
    float temp4 = 1.0;
    float tempOut;
    for (int j = 0; j < 64; j++) {
			temp1 += temp1 * M;
    }

    tempOut = temp1 + temp2 + temp3 + temp4;
		currentGOut[XGRid*XLSize*2+XLid] = tempOut;
  }

  // For now just print out some initial values of both Gout and currentGOut

	cout << "Print out verification for case: WGS" << localSize << "MAPIXLL" << LL << "OPS64" << endl;
  for (int i = 0; i < GOutSize; i++) {
    output << GOut[i] << " " << currentGOut[i] << endl;
    if (GOut[i] != currentGOut[i] ) {
 			cout << "inconsistence results on index " << i << " => " << GOut[i] << " " << currentGOut[i] << endl;
    }
  }
	output.close ();
  cout << endl;

}

void verifyWGSXMAPIXLLXOPS128GAP0 (float* GOut, int GOutSize, float M, float N, float P, long LL, int localSize) {

	ofstream output;
  string outName = FILE_PREFIX + "verifyWGSXMAPIXLLXOPS128GAP0.txt";
  output.open (outName);

	cout << "Executing verification function verifyWGSXMAPIXLLXOPS128GAP0" << endl;

	float* currentGOut = new float[GOutSize];

	for (int i = 0; i < LL; i++) {

		int XGL = i;
		int XGRid = i / localSize;
    int XGRnum = LL / localSize;
    int XLSize = localSize;
    int XLid = i % localSize;

		float temp1 = 1.0;
    float temp2 = 1.0;
    float temp3 = 1.0;
    float temp4 = 1.0;
    float tempOut;
    for (int j = 0; j < 128; j++) {
			temp1 += temp1 * M;
    }

    tempOut = temp1 + temp2 + temp3 + temp4;
		currentGOut[XGRid*XLSize*2+XLid] = tempOut;
  }

  // For now just print out some initial values of both Gout and currentGOut

	cout << "Print out verification for case: WGS" << localSize << "MAPIXLL" << LL << "OPS128" << endl;
  for (int i = 0; i < GOutSize; i++) {
    output << GOut[i] << " " << currentGOut[i] << endl;
    if (GOut[i] != currentGOut[i] ) {
 			cout << "inconsistence results on index " << i << " => " << GOut[i] << " " << currentGOut[i] << endl;
    }
  }
  output.close();
  cout << endl;

}

void verifyWGSXMAPIXLLXOPS512GAP0 (float* GOut, int GOutSize, float M, float N, float P, long LL, int localSize) {

	ofstream output;
  string outName = FILE_PREFIX + "verifyWGSXMAPIXLLXOPS512GAP0.txt";
  output.open (outName);

	cout << "Executing verification function verifyWGSXMAPIXLLXOPS512GAP0" << endl;

	float* currentGOut = new float[GOutSize];

	for (int i = 0; i < LL; i++) {

		int XGL = i;
		int XGRid = i / localSize;
    int XGRnum = LL / localSize;
    int XLSize = localSize;
    int XLid = i % localSize;

		float temp1 = 1.0;
    float temp2 = 1.0;
    float temp3 = 1.0;
    float temp4 = 1.0;
    float tempOut;
    for (int j = 0; j < 512; j++) {
			temp1 += temp1 * M;
    }

    tempOut = temp1 + temp2 + temp3 + temp4;
		currentGOut[XGRid*XLSize*2+XLid] = tempOut;
  }

  // For now just print out some initial values of both Gout and currentGOut

	cout << "Print out verification for case: WGS" << localSize << "MAPIXLL" << LL << "OPS512" << endl;
  for (int i = 0; i < GOutSize; i++) {
    output << GOut[i] << " " << currentGOut[i] << endl;
    if (GOut[i] != currentGOut[i] ) {
 			cout << "inconsistence results on index " << i << " => " << GOut[i] << " " << currentGOut[i] << endl;
    }
  }
	output.close();
  cout << endl;

}

void verifyWGSXMAPIXLLXOPS1024GAP1 (float* GOut, int GOutSize, float M, float N, float P, long LL, int localSize) {

	float* currentGOut = new float[GOutSize];

	for (int i = 0; i < LL; i++) {

		int XGL = i;
		int XGRid = i / localSize;
    int XGRnum = LL / localSize;
    int XLSize = localSize;
    int XLid = i % localSize;

		float temp1 = 1.0;
    float temp2 = 1.0;
    float temp3 = 1.0;
    float temp4 = 1.0;
    float tempOut;
    for (int j = 0; j < 512; j++) {
			temp1 += temp1 * M; temp2 += temp2 * N;
    }

    tempOut = temp1 + temp2 + temp3 + temp4;
		currentGOut[XGRid*XLSize*2+XLid] = tempOut;
  }

  // For now just print out some initial values of both Gout and currentGOut

	cout << "Print out verification for case: WGS" << localSize << "MAPIXLL" << LL << "OPS1024" << endl;
  for (int i = 0; i < GOutSize; i++) {
    if (GOut[i] != currentGOut[i] ) {
 			cout << "inconsistence results on index " << i << " => " << GOut[i] << " " << currentGOut[i] << endl;
    }
  }
  cout << endl;

}

void verifyWGSXMAPIXLLXOPS64GAP1 (float* GOut, int GOutSize, float M, float N, float P, long LL, int localSize) {

	float* currentGOut = new float[GOutSize];

	for (int i = 0; i < LL; i++) {

		int XGL = i;
		int XGRid = i / localSize;
    int XGRnum = LL / localSize;
    int XLSize = localSize;
    int XLid = i % localSize;

		float temp1 = 1.0;
    float temp2 = 1.0;
    float temp3 = 1.0;
    float temp4 = 1.0;
    float tempOut;
    for (int j = 0; j < 32; j++) {
			temp1 += temp1 * M; temp2 += temp2 * N;
    }

    tempOut = temp1 + temp2 + temp3 + temp4;
		currentGOut[XGRid*XLSize*2+XLid] = tempOut;
  }

  // For now just print out some initial values of both Gout and currentGOut

	cout << "Print out verification for case: WGS" << localSize << "MAPIXLL" << LL << "OPS64" << endl;
  for (int i = 0; i < GOutSize; i++) {
    if (GOut[i] != currentGOut[i] ) {
 			cout << "inconsistence results on index " << i << " => " << GOut[i] << " " << currentGOut[i] << endl;
    }
  }

  cout << endl;

}

void verifyWGSXMAPIXLLXOPS128GAP1 (float* GOut, int GOutSize, float M, float N, float P, long LL, int localSize) {

	float* currentGOut = new float[GOutSize];

	for (int i = 0; i < LL; i++) {

		int XGL = i;
		int XGRid = i / localSize;
    int XGRnum = LL / localSize;
    int XLSize = localSize;
    int XLid = i % localSize;

		float temp1 = 1.0;
    float temp2 = 1.0;
    float temp3 = 1.0;
    float temp4 = 1.0;
    float tempOut;
    for (int j = 0; j < 64; j++) {
			temp1 += temp1 * M; temp2 += temp2 * N;
    }

    tempOut = temp1 + temp2 + temp3 + temp4;
		currentGOut[XGRid*XLSize*2+XLid] = tempOut;
  }

  // For now just print out some initial values of both Gout and currentGOut

	cout << "Print out verification for case: WGS" << localSize << "MAPIXLL" << LL << "OPS128" << endl;
  for (int i = 0; i < GOutSize; i++) {
    if (GOut[i] != currentGOut[i] ) {
 			cout << "inconsistence results on index " << i << " => " << GOut[i] << " " << currentGOut[i] << endl;
    }
  }

  cout << endl;

}

void verifyWGSXMAPIXLLXOPS512GAP1 (float* GOut, int GOutSize, float M, float N, float P, long LL, int localSize) {

	float* currentGOut = new float[GOutSize];

	for (int i = 0; i < LL; i++) {

		int XGL = i;
		int XGRid = i / localSize;
    int XGRnum = LL / localSize;
    int XLSize = localSize;
    int XLid = i % localSize;

		float temp1 = 1.0;
    float temp2 = 1.0;
    float temp3 = 1.0;
    float temp4 = 1.0;
    float tempOut;
    for (int j = 0; j < 256; j++) {
			temp1 += temp1 * M; temp2 += temp2 * N;
    }

    tempOut = temp1 + temp2 + temp3 + temp4;
		currentGOut[XGRid*XLSize*2+XLid] = tempOut;
  }

  // For now just print out some initial values of both Gout and currentGOut

	cout << "Print out verification for case: WGS" << localSize << "MAPIXLL" << LL << "OPS512" << endl;
  for (int i = 0; i < GOutSize; i++) {
    if (GOut[i] != currentGOut[i] ) {
 			cout << "inconsistence results on index " << i << " => " << GOut[i] << " " << currentGOut[i] << endl;
    }
  }

  cout << endl;

}

void verifyWGSXMAPIXLLXOPS1024GAP3 (float* GOut, int GOutSize, float M, float N, float P, long LL, int localSize) {

	float* currentGOut = new float[GOutSize];

	for (int i = 0; i < LL; i++) {

		int XGL = i;
		int XGRid = i / localSize;
    int XGRnum = LL / localSize;
    int XLSize = localSize;
    int XLid = i % localSize;

		float temp1 = 1.0;
    float temp2 = 1.0;
    float temp3 = 1.0;
    float temp4 = 1.0;
    float tempOut;
    for (int j = 0; j < 256; j++) {
			temp1 += temp1 * M; temp2 += temp2 * N; temp3 += temp3 * P; temp4 += temp4 * M;
    }

    tempOut = temp1 + temp2 + temp3 + temp4;
		currentGOut[XGRid*XLSize*2+XLid] = tempOut;
  }

  // For now just print out some initial values of both Gout and currentGOut

	cout << "Print out verification for case: WGS" << localSize << "MAPIXLL" << LL << "OPS1024" << endl;
  for (int i = 0; i < GOutSize; i++) {
    if (GOut[i] != currentGOut[i] ) {
 			cout << "inconsistence results on index " << i << " => " << GOut[i] << " " << currentGOut[i] << endl;
    }
  }

  cout << endl;

}

void verifyWGSXMAPIXLLXOPS64GAP3 (float* GOut, int GOutSize, float M, float N, float P, long LL, int localSize) {

	float* currentGOut = new float[GOutSize];

	for (int i = 0; i < LL; i++) {

		int XGL = i;
		int XGRid = i / localSize;
    int XGRnum = LL / localSize;
    int XLSize = localSize;
    int XLid = i % localSize;

		float temp1 = 1.0;
    float temp2 = 1.0;
    float temp3 = 1.0;
    float temp4 = 1.0;
    float tempOut;
    for (int j = 0; j < 16; j++) {
			temp1 += temp1 * M; temp2 += temp2 * N; temp3 += temp3 * P; temp4 += temp4 * M;
    }

    tempOut = temp1 + temp2 + temp3 + temp4;
		currentGOut[XGRid*XLSize*2+XLid] = tempOut;
  }

  // For now just print out some initial values of both Gout and currentGOut

	cout << "Print out verification for case: WGS" << localSize << "MAPIXLL" << LL << "OPS64" << endl;
  for (int i = 0; i < GOutSize; i++) {
    if (GOut[i] != currentGOut[i] ) {
 			cout << "inconsistence results on index " << i << " => " << GOut[i] << " " << currentGOut[i] << endl;
    }
  }

  cout << endl;

}

void verifyWGSXMAPIXLLXOPS128GAP3 (float* GOut, int GOutSize, float M, float N, float P, long LL, int localSize) {

	float* currentGOut = new float[GOutSize];

	for (int i = 0; i < LL; i++) {

		int XGL = i;
		int XGRid = i / localSize;
    int XGRnum = LL / localSize;
    int XLSize = localSize;
    int XLid = i % localSize;

		float temp1 = 1.0;
    float temp2 = 1.0;
    float temp3 = 1.0;
    float temp4 = 1.0;
    float tempOut;
    for (int j = 0; j < 32; j++) {
			temp1 += temp1 * M; temp2 += temp2 * N; temp3 += temp3 * P; temp4 += temp4 * M;
    }

    tempOut = temp1 + temp2 + temp3 + temp4;
		currentGOut[XGRid*XLSize*2+XLid] = tempOut;
  }

  // For now just print out some initial values of both Gout and currentGOut

	cout << "Print out verification for case: WGS" << localSize << "MAPIXLL" << LL << "OPS128" << endl;
  for (int i = 0; i < GOutSize; i++) {
    if (GOut[i] != currentGOut[i] ) {
 			cout << "inconsistence results on index " << i << " => " << GOut[i] << " " << currentGOut[i] << endl;
    }
  }

  cout << endl;

}

void verifyWGSXMAPIXLLXOPS512GAP3 (float* GOut, int GOutSize, float M, float N, float P, long LL, int localSize) {

	float* currentGOut = new float[GOutSize];

	for (int i = 0; i < LL; i++) {

		int XGL = i;
		int XGRid = i / localSize;
    int XGRnum = LL / localSize;
    int XLSize = localSize;
    int XLid = i % localSize;

		float temp1 = 1.0;
    float temp2 = 1.0;
    float temp3 = 1.0;
    float temp4 = 1.0;
    float tempOut;
    for (int j = 0; j < 128; j++) {
			temp1 += temp1 * M; temp2 += temp2 * N; temp3 += temp3 * P; temp4 += temp4 * M;
    }

    tempOut = temp1 + temp2 + temp3 + temp4;
		currentGOut[XGRid*XLSize*2+XLid] = tempOut;
  }

  // For now just print out some initial values of both Gout and currentGOut

	cout << "Print out verification for case: WGS" << localSize << "MAPIXLL" << LL << "OPS512" << endl;
  for (int i = 0; i < GOutSize; i++) {
    if (GOut[i] != currentGOut[i] ) {
 			cout << "inconsistence results on index " << i << " => " << GOut[i] << " " << currentGOut[i] << endl;
    }
  }

  cout << endl;

}

#endif
