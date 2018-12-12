//
// (c) August 1, 2018 Saman Biookaghazadeh @ Arizona State University
//

// Mainly for S1119 test case and other thing like that

#define megaIter 1
#define a 1.385
#define b 1.8

#define OP(input1,input2,input3) input1 = input1 * input2 + input3
#define OP2(input1,input2,input3) OP(input1,input2,input3); OP(input1,input2,input3)
#define OP3(input1,input2,input3) OP2(input1,input2,input3); OP(input1,input2,input3)
#define OP4(input1,input2,input3) OP3(input1,input2,input3); OP(input1,input2,input3)
#define OP5(input1,input2,input3) OP4(input1,input2,input3); OP(input1,input2,input3)
#define OP6(input1,input2,input3) OP5(input1,input2,input3); OP(input1,input2,input3)
#define OP7(input1,input2,input3) OP6(input1,input2,input3); OP(input1,input2,input3)
#define OP8(input1,input2,input3) OP7(input1,input2,input3); OP(input1,input2,input3)

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

#define CInit(A,B,C,D) DTYPE tempA = 0; DTYPE tempB = B; DTYPE tempC = C; DTYPE tempD = D
#define CFinal(A) A = tempA

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

#define Bfunction(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; AFinal (A)
#define Bfunction2(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; AFinal(A)
#define Bfunction3(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; AFinal(A)
#define Bfunction4(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; AFinal(A)
#define Bfunction5(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; AFinal(A)
#define Bfunction6(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; AFinal(A)
#define Bfunction7(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; AFinal(A)
#define Bfunction8(A,B,C) AInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; AFinal(A)

#define megaBfunction(A,B,C) AInit(A,B,C); OP8(tempA,tempB,tempC); AFinal (A)
#define megaBfunction2(A,B,C) AInit(A,B,C); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); AFinal (A)
#define megaBfunction3(A,B,C) AInit(A,B,C); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); AFinal (A)
#define megaBfunction4(A,B,C) AInit(A,B,C); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); Op8(tempA,tempB,tempC); AFinal(A)
#define megaBfunction5(A,B,C) AInit(A,B,C); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); AFinal(A)
#define megaBfunction6(A,B,C) AInit(A,B,C); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); AFinal(A)
#define megaBfunction7(A,B,C) AInit(A,B,C); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); AFinal(A)
#define megaBfunction8(A,B,C) AInit(A,B,C); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); OP8(tempA,tempB,tempC); AFinal(A)

#define BBfunction(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; BBfinal (A)
#define BBfunction2(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; BBfinal(A)
#define BBfunction3(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; BBfinal(A)
#define BBfunction4(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; BBfinal(A)
#define BBfunction5(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; BBfinal(A)
#define BBfunction6(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; BBfinal(A)
#define BBfunction7(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; BBfinal(A)
#define BBfunction8(A,B,C) BBInit(A, B, C); tempA = cos(tempB) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; tempA = cos(tempA) * tempC; BBfinal(A)


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

#define Dfunction(A,B,C,D,E) DInit(A, B, C, D, E); tempA = cos(tempB) * tempC * tempD * tempE; DFinal(A)
#define Dfunction2(A,B,C,D,E) DInit(A, B, C, D, E); tempA = cos(tempB) * tempC * tempD * tempE; tempA = cos(tempA) * tempE; DFinal(A)
#define Dfunction3(A,B,C,D,E) DInit(A, B, C, D, E); tempA = cos(tempB) * tempC * tempD * tempE; tempA = cos(tempA) * tempE; tempA = cos(tempA) * tempE; DFinal(A)
#define Dfunction4(A,B,C,D,E) DInit(A, B, C, D, E); tempA = cos(tempB) * tempC * tempD * tempE; tempA = cos(tempA) * tempE; tempA = cos(tempA) * tempE; tempA = cos(tempA) * tempE; DFinal(A)
#define Dfunction5(A,B,C,D,E) DInit(A, B, C, D, E); tempA = cos(tempB) * tempC * tempD * tempE; tempA = cos(tempA) * tempE; tempA = cos(tempA) * tempE; tempA = cos(tempA) * tempE; tempA = cos(tempA) * tempE; DFinal(A)
