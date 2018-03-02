#ifndef VERIFY_H_
#define VERIFY_H_

#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

string FILE_PREFIX = "/home/user/sbiookag/Algs/1For/nodep/GAP0-FloatParam/";

void verifyMatrixMultiplication (float *A, float *B, float *C, float *R,
                                 int A_height, int A_width,
                                 int B_height, int B_width,
                                 int C_height, int C_width,
                                 int batch_size) {

  	int verbose = 1;

  	float *I = new float[A_height*B_width*batch_size];
    for (int z = 0; z < batch_size; z++) {
      int A_offset = A_height * A_width * z;
      int B_offset = B_height * B_width * z;
      int I_offset = A_height * B_width * z;
    	for (int i = 0; i < A_height; i++) {
      	for (int j = 0; j < B_width; j++) {
        	float sum = 0;
        	for (int k = 0; k < A_width; k++) {

            //						if (verbose) cout << "Accessing index (" << i
            //                  << ", " << j
            //                  << ", " << k
            //                  << ") = " << "("
            //                  << i*A_width+k+A_offset
            //                  << ", "
            //                  << j+k*B_width+B_offset
            //                  << ")" << endl;

          	sum += (A[i*A_width+k+A_offset]*B[j+k*B_width+B_offset]);
        	}
        	I[i*A_width+j+I_offset] = sum;
      	}
    	}
    }

    int I_height = A_height;
    int I_width = B_width;

		if (verbose) cout << "Second Stage" << endl;

    for (int z = 0; z < batch_size; z++) {
      int I_offset = A_height * B_width * z;
      int C_offset = C_height * C_width * z;
      int R_offset = A_height * C_width * z;
    	for (int i = 0; i < I_height; i++) {
      	for (int j = 0; j < C_width; j++) {
        	float sum = 0;
        	for (int k = 0; k < I_width; k++) {
          	sum += (I[i*I_width+k+I_offset]*C[j+k*C_width+C_offset]);
        	}
        	R[i*I_width+j+R_offset] = sum;
      	}
    	}
    }

}

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
