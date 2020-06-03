#ifndef FUNCS_H_
#define FUNCS_H_

#define AInit(A,B,C) DTYPE tempA = 0; DTYPE tempB = B; DTYPE tempC = C
#define AFinal(A) A = tempA
#define FMAInit	tempA = tempB * tempC + tempA
#define FMA1 	tempA = tempA * tempC + tempA
#define FMA2   	FMA1; FMA1
#define FMA3   	FMA2; FMA1
#define FMA4   	FMA3; FMA1
#define FMA5   	FMA4; FMA1
#define FMA6   	FMA5; FMA1
#define FMA7   	FMA6; FMA1
#define FMA8   	FMA7; FMA1
#define FMA9   	FMA8; FMA1
#define FMA10  	FMA9; FMA1
#define FMA11  	FMA10; FMA1
#define FMA12  	FMA11; FMA1
#define FMA13  	FMA12; FMA1
#define FMA14  	FMA13; FMA1
#define FMA15  	FMA14; FMA1

#define functionI(A,B,C) AInit(A,B,C); FMAInit; FMA1; AFinal(A)
#define functionII(A,B,C) AInit(A,B,C); FMAInit; FMA3; AFinal(A)
#define functionIII(A,B,C) AInit(A,B,C); FMAInit; FMA5; AFinal(A)
#define functionIV(A,B,C) AInit(A,B,C); FMAInit; FMA7; AFinal(A)
#define functionV(A,B,C) AInit(A,B,C); FMAInit; FMA9; AFinal(A)

#define CInit(A,B,C, D) DTYPE tempA = 0; DTYPE tempB = B; DTYPE tempC = C; DTYPE tempD = C
#define CFinal(A) A = tempA
#define CFMAInit	tempA = tempA + tempB * tempC + tempD
#define CFMA1		tempA = tempA + tempB * tempC
#define CFMA2		CFMA1; CFMA1
#define CFMA3		CFMA2; CFMA1
#define CFMA4		CFMA3; CFMA1
#define CFMA5		CFMA4; CFMA1
#define CFMA6		CFMA5; CFMA1
#define CFMA7		CFMA6; CFMA1
#define CFMA8		CFMA7; CFMA1
#define CFMA9		CFMA8; CFMA1
#define CFMA10		CFMA9; CFMA1
#define CFMA11		CFMA10; CFMA1
#define CFMA12		CFMA11; CFMA1
#define CFMA13		CFMA12; CFMA1
#define CFMA14		CFMA13; CFMA1
#define CFMA15		CFMA14; CFMA1

#define	CfunctionI(A,B,C,D)	CInit(A,B,C,D); CFMAInit; CFMA1; CFinal(A)
#define	CfunctionII(A,B,C,D)	CInit(A,B,C,D); CFMAInit; CFMA3; CFinal(A)
#define	CfunctionIII(A,B,C,D)	CInit(A,B,C,D); CFMAInit; CFMA5; CFinal(A)
#define	CfunctionIV(A,B,C,D)	CInit(A,B,C,D); CFMAInit; CFMA7; CFinal(A)
#define	CfunctionV(A,B,C,D)	CInit(A,B,C,D); CFMAInit; CFMA9; CFinal(A)

#endif 
