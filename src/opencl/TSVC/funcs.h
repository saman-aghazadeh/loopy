//
// (c) August 1, 2018 Saman Biookaghazadeh @ Arizona State University
//

// Mainly for S1119 test case and other thing like that


#define OP(input1,input2,input3) input1 = input1 + input2 * input3
#define OP2(input1,input2,input3) OP(input1,input2,input3); OP(input1,input2,input3)
#define OP3(input1,input2,input3) OP2(input1,input2,input3); OP(input1,input2,input3)
#define OP4(input1,input2,input3) OP3(input1,input2,input3); OP(input1,input2,input3)
#define OP5(input1,input2,input3) OP4(input1,input2,input3); OP(input1,input2,input3)
#define OP6(input1,input2,input3) OP5(input1,input2,input3); OP(input1,input2,input3)
#define OP7(input1,input2,input3) OP6(input1,input2,input3); OP(input1,input2,input3)
#define OP8(input1,input2,input3) OP7(input1,input2,input3); OP(input1,input2,input3)

#if NUMFMAS==1
#define OPX(input1,input2,input3) OP(input1,input2,input3)
#elif NUMFMAS==2
#define OPX(input1,input2,input3) OP2(input1,input2,input3)
#elif NUMFMAS==3
#define OPX(input1,input2,input3) OP3(input1,input2,input3)
#elif NUMFMAS==4
#define OPX(input1,input2,input3) OP4(input1,input2,input3)
#elif NUMFMAS==5
#define OPX(input1,input2,input3) OP5(input1,input2,input3)
#elif NUMFMAS==6
#define OPX(input1,input2,input3) OP6(input1,input2,input3)
#elif NUMFMAS==7
#define OPX(input1,input2,input3) OP7(input1,input2,input3)
#elif NUMFMAS==8
#define OPX(input1,input2,input3) OP8(input1,input2,input3)
#endif

#define OPNAME OP

#define PPCAT_NX(A,B) A ## B
#define PPCAT(A,B) PPCAT_NX(A,B)
#define STRINGIZE_NX(A) #A
#define STRINGIZE(A) STRINGIZE_NX(A)
//#define OPX(input1,input2,input3) STRINGIZE(PPCAT(OP, NUMFMAS)) ## (input1,input2,input3)

/*
inline DTYPE megafunc(DTYPE input1, DTYPE input2) {
	DTYPE a = 1.385;
  DTYPE b = 1.8;
  #pragma unroll
  for (int megaCounter = 0; megaCounter < megaIter; megaCounter++) {
    input1 = input1 * a + input2;
  }

  return input1;
}
*/
//#define megafunc(input1, input2) input1 = input1 * a + input2


#define AInit(A,B,C) DTYPE tempA = 0; DTYPE tempB = B; DTYPE tempC = C
#define AFinal(A) A = tempA

#define megaAInit(A,B,C) DTYPE tempA = A; DTYPE tempB = B; DTYPE tempC = C
#define megaAFinal(A) A = tempA

#define CInit(A,B,C,D) DTYPE tempA = 0; DTYPE tempB = B; DTYPE tempC = C; DTYPE tempD = D
#define CFinal(A) A = tempA

#define megaCInit(A,B,C,D) DTYPE tempA = A; DTYPE tempB = B; DTYPE tempC = C; DTYPE tempD = D
#define megaCFinal(A) A = tempA

#define BBInit(A,B,C) VTYPE tempA = 0; VTYPE tempB = B; VTYPE tempC = C
#define BBfinal(A) A = tempA

#define DInit(A,B,C,D,E) DTYPE tempA = 0; DTYPE tempB = B; DTYPE tempC = C; DTYPE tempD = D; DTYPE tempE = E;
#define DFinal(A) A = tempA

#define Afunction(A,B,C) AInit (A, B, C); tempA = pow(tempB,tempC); AFinal (A)
#define Afunction2(A,B,C) AInit (A, B, C); tempA = pow(tempB,tempC); tempA = pow(tempB, tempA); AFinal(A)
#define Afunction3(A,B,C) AInit (A, B, C); tempA = pow(tempB, tempC); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); AFinal(A)
#define Afunction4(A,B,C) AInit(A, B, C); tempA = pow(tempB, tempC); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); AFinal(A)
#define Afunction5(A,B,C) AInit(A, B, C); tempA = pow(tempB, tempC); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); AFinal(A)
#define Afunction6(A,B,C) AInit(A, B, C); tempA = pow(tempB, tempC); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); AFinal(A)
#define Afunction7(A,B,C) AInit(A, B, C); tempA = pow(tempB, tempC); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); AFinal(A)
#define Afunction8(A,B,C) AInit(A, B, C); tempA = pow(tempB, tempC); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); tempA = pow(tempB, tempA); AFinal(A)

#define BFMAI	tempA = tempB * tempC + tempA
#define BFMA1	tempA = tempA * tempC + tempA
#define BFMA2 	BFMA1; BFMA1
#define BFMA3	BFMA2; BFMA1
#define BFMA4	BFMA3; BFMA1
#define BFMA5	BFMA4; BFMA1
#define	BFMA6	BFMA5; BFMA1
#define	BFMA7	BFMA6; BFMA1
#define BFMA8	BFMA7; BFMA1
#define BFMA9	BFMA8; BFMA1
#define BFMA10	BFMA9; BFMA1
#define BFMA11	BFMA10; BFMA1
#define BFMA12	BFMA11; BFMA1
#define BFMA13	BFMA12; BFMA1
#define BFMA14	BFMA13; BFMA1
#define BFMA15	BFMA14; BFMA1

#define Bfunction(A, B, C) AInit(A, B, C); BFMAI; BFMA1; AFinal (A)
#define Bfunction2(A, B, C) AInit(A, B, C); BFMAI; BFMA3; AFinal (A)
#define Bfunction3(A, B, C) AInit(A, B, C); BFMAI; BFMA5; AFinal (A)
#define Bfunction4(A, B, C) AInit(A, B, C); BFMAI; BFMA7; AFinal (A)
#define Bfunction5(A, B, C) AInit(A, B, C); BFMAI; BFMA9; AFinal (A)


// #define Bfunction(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; AFinal (A)
// #define Bfunction2(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; AFinal(A)
// #define Bfunction3(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; AFinal(A)
// #define Bfunction4(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; AFinal(A)
// #define Bfunction5(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; AFinal(A)
// #define Bfunction6(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; AFinal(A)
// #define Bfunction7(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; AFinal(A)
// #define Bfunction8(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; AFinal(A)

#define megaBfunction(A,B,C) megaAInit(A,B,C); OPX (tempA,tempB,tempC); megaAFinal (A)
#define megaBfunction2(A,B,C) megaAInit(A,B,C); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); megaAFinal (A)
#define megaBfunction3(A,B,C) megaAInit(A,B,C); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); megaAFinal (A)
#define megaBfunction4(A,B,C) megaAInit(A,B,C); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); megaAFinal(A)
#define megaBfunction5(A,B,C) megaAInit(A,B,C); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); megaAFinal(A)
#define megaBfunction6(A,B,C) megaAInit(A,B,C); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); megaAFinal(A)
#define megaBfunction7(A,B,C) megaAInit(A,B,C); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); megaAFinal(A)
#define megaBfunction8(A,B,C) megaAInit(A,B,C); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); megaAFinal(A)

#define megaBfunctionNoAcc(A,B,C) AInit(A,B,C); OPX (tempA,tempB,tempC); megaAFinal (A)
#define megaBfunctionNoAcc2(A,B,C) AInit(A,B,C); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); megaAFinal (A)
#define megaBfunctionNoAcc3(A,B,C) AInit(A,B,C); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); megaAFinal (A)
#define megaBfunctionNoAcc4(A,B,C) AInit(A,B,C); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); megaAFinal(A)
#define megaBfunctionNoAcc5(A,B,C) AInit(A,B,C); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); OPX(tempA,tempB,tempC); megaAFinal(A)

#define BBfunction(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; BBfinal (A)
#define BBfunction2(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; BBfinal(A)
#define BBfunction3(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; BBfinal(A)
#define BBfunction4(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; BBfinal(A)
#define BBfunction5(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; BBfinal(A)
#define BBfunction6(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; BBfinal(A)
#define BBfunction7(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; BBfinal(A)
#define BBfunction8(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; BBfinal(A)

#define megaBBfunction(A,B,C) BBInit(A, B, C); OP8(tempA,tempB,tempC); BBfinal (A)
#define megaBBfunction2(A,B,C) BBInit(A, B, C); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); BBfinal(A)
#define megaBBfunction3(A,B,C) BBInit(A, B, C); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); BBfinal(A)
#define megaBBfunction4(A,B,C) BBInit(A, B, C); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); BBfinal(A)
#define megaBBfunction5(A,B,C) BBInit(A, B, C); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); BBfinal(A)


#define Cfunction(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC * tempD; CFinal(A)
#define Cfunction2(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC * tempD; tempA = cos(tempA) * tempC; CFinal(A)
#define Cfunction3(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC * tempD; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; CFinal(A)
#define Cfunction4(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC * tempD; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; CFinal(A)
#define Cfunction5(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC * tempD; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; CFinal(A)
#define Cfunction6(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC * tempD; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; CFinal(A)
#define Cfunction7(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC * tempD; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; CFinal(A)
#define Cfunction8(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC * tempD; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; CFinal(A)

#define CCfunction(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC + tempD; CFinal(A)
#define CCfunction2(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC + tempD; tempA = cos(tempA) * tempC; CFinal(A)
#define CCfunction3(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC + tempD; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; CFinal(A)
#define CCfunction4(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC + tempD; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; CFinal(A)
#define CCfunction5(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC + tempD; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; CFinal(A)
#define CCfunction6(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC + tempD; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; CFinal(A)
#define CCfunction7(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC + tempD; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; CFinal(A)
#define CCfunction8(A,B,C,D) CInit(A, B, C, D); tempA = cos(tempB) * tempC + tempD; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; CFinal(A)

#define megaCfunction(A,B,C,D) megaCInit(A,B,C,D); OPX(tempA,tempB,(tempC+tempD)); megaCFinal (A)
#define megaCfunction2(A,B,C,D) megaCInit(A,B,C,D); OPX(tempA,tempB,(tempC+tempD)); OPX(tempA, tempB, tempC); megaCFinal (A)
#define megaCfunction3(A,B,C,D) megaCInit(A,B,C,D); OPX(tempA,tempB,(tempC+tempD)); OPX(tempA, tempB, tempC); OPX(tempA, tempB, tempC); megaCFinal (A)
#define megaCfunction4(A,B,C,D) megaCInit(A,B,C,D); OPX(tempA,tempB,(tempC+tempD)); OPX(tempA, tempB, tempC); OPX(tempA, tempB, tempC); OPX(tempA, tempB, tempC); megaCFinal (A)
#define megaCfunction5(A,B,C,D) megaCInit(A,B,C,D); OPX(tempA,tempB,(tempC+tempD)); OPX(tempA, tempB, tempC); OPX(tempA, tempB, tempC); OPX(tempA, tempB, tempC); OPX(tempA, tempB, tempC); megaCFinal (A)

#define Dfunction(A,B,C,D,E) DInit(A, B, C, D, E); tempA = cos(tempB) * tempC * tempD * tempE; DFinal(A)
#define Dfunction2(A,B,C,D,E) DInit(A, B, C, D, E); tempA = cos(tempB) * tempC * tempD * tempE; tempA = cos(tempA) * tempE; DFinal(A)
#define Dfunction3(A,B,C,D,E) DInit(A, B, C, D, E); tempA = cos(tempB) * tempC * tempD * tempE; tempA = cos(tempA) * tempE; tempA = cos(tempA) * tempE; DFinal(A)
#define Dfunction4(A,B,C,D,E) DInit(A, B, C, D, E); tempA = cos(tempB) * tempC * tempD * tempE; tempA = cos(tempA) * tempE; tempA = cos(tempA) * tempE; tempA = cos(tempA) * tempE; DFinal(A)
#define Dfunction5(A,B,C,D,E) DInit(A, B, C, D, E); tempA = cos(tempB) * tempC * tempD * tempE; tempA = cos(tempA) * tempE; tempA = cos(tempA) * tempE; tempA = cos(tempA) * tempE; tempA = cos(tempA) * tempE; DFinal(A)
