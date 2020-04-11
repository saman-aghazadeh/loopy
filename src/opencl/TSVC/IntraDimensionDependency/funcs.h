#define AInit(A,B) DTYPE tempA = A; DTYPE tempB = B
#define AFinal(A) A = tempA

#define ACalc1(A,B) tempA += tempA * tempB
#define ACalc2(A,B) ACalc1(A,B); ACalc1(A,B)
#define ACalc3(A,B) ACalc2(A,B); ACalc1(A,B)
#define ACalc4(A,B) ACalc3(A,B); ACalc1(A,B)
#define ACalc5(A,B) ACalc4(A,B); ACalc1(A,B)

#define Afunction1(A,B) AInit(A,B); ACalc1(A,B); AFinal(A)
#define Afunction2(A,B) AInit(A,B); ACalc2(A,B); AFinal(A)
#define Afunction3(A,B) AInit(A,B); ACalc3(A,B); AFinal(A)
#define Afunction4(A,B) AInit(A,B); ACalc4(A,B); AFinal(A)
#define Afunction5(A,B) AInit(A,B); ACalc5(A,B); AFinal(A)
