#include "funcs.h"

ulong AntiDependent2 (void *AA, void *BB, void *CC,
                              int size) {

        DTYPE* AAptr = (DTYPE *) AA;
        DTYPE* BBptr = (DTYPE *) BB;
        DTYPE* CCptr = (DTYPE *) CC;

        auto start = GETTIME;

	DTYPE multiplier = rand();

        #pragma omp parallel for
        for (int y = 1; y < size; y++) {
		
#if INTENSITY1
		{
			functionI(BBptr[y], BBptr[y+1], multiplier);
		} 
		{
			functionI(AAptr[y], BBptr[y-1], multiplier);
		}
#elif INTENSITY2
		{
			functionII(BBptr[y], BBptr[y+1], multiplier);
		} 
		{
			functionII(AAptr[y], BBptr[y-1], multiplier);
		}
#elif INTENSITY3
		{		
			functionIII(BBptr[y], BBptr[y+1], multiplier);
		}
		{
			functionIII(AAptr[y], BBptr[y-1], multiplier);
		}
#elif INTENSITY4
		{
			functionIV(BBptr[y], BBptr[y+1], multiplier);
		}
		{
			functionIV(AAptr[y], BBptr[y-1], multiplier);
		}
#elif INTENSITY5
		{
			functionIII(BBptr[y], BBptr[y+1], multiplier);
		}
		{
			functionIII(AAptr[y], BBptr[y-1], multiplier);
		}
#endif
        }

        auto end = GETTIME;
        return DIFFTIME(end, start);
}

ulong AntiDependent4 (void *AA, void *BB, void *CC,
                              int size) {

        DTYPE* AAptr = (DTYPE *) AA;
        DTYPE* BBptr = (DTYPE *) BB;
        DTYPE* CCptr = (DTYPE *) CC;

        auto start = GETTIME;

	DTYPE multiplier = rand();

        #pragma omp parallel for
        for (int y = 1; y < size; y++) {
		
#if INTENSITY1
		{
			functionI(BBptr[y+4], BBptr[y+5], multiplier);
		}
		{
			functionI(BBptr[y+2], BBptr[y+3], multiplier);
		}
		{
			functionI(BBptr[y], BBptr[y+1], multiplier);
		}
		{
			functionI(AAptr[y], BBptr[y-1], multiplier);
		}
#elif INTENSITY2
		{
			functionII(BBptr[y+4], BBptr[y+5], multiplier);
		}
		{
			functionII(BBptr[y+2], BBptr[y+3], multiplier);
		}
		{
			functionII(BBptr[y], BBptr[y+1], multiplier);
		}
		{
			functionII(AAptr[y], BBptr[y-1], multiplier);
		}
#elif INTENSITY3
		{
			functionIII(BBptr[y+4], BBptr[y+5], multiplier);
		}
		{
			functionIII(BBptr[y+2], BBptr[y+3], multiplier);
		}
		{
			functionIII(BBptr[y], BBptr[y+1], multiplier);
		}
		{
			functionIII(AAptr[y], BBptr[y-1], multiplier);
		}
#elif INTENSITY4
		{
			functionIV(BBptr[y+4], BBptr[y+5], multiplier);
		}
		{
			functionIV(BBptr[y+2], BBptr[y+3], multiplier);
		}
		{
			functionIV(BBptr[y], BBptr[y+1], multiplier);
		}
		{
			functionIV(AAptr[y], BBptr[y-1], multiplier);
		}
#elif INTENSITY5
		{
			functionIII(BBptr[y+4], BBptr[y+5], multiplier);
		}
		{
			functionIII(BBptr[y+2], BBptr[y+3], multiplier);
		}
		{
			functionIII(BBptr[y], BBptr[y+1], multiplier);
		}
		{
			functionIII(AAptr[y], BBptr[y-1], multiplier);
		}
#endif
        }

        auto end = GETTIME;
        return DIFFTIME(end, start);
}

ulong AntiDependent6 (void *AA, void *BB, void *CC,
                              int size) {

        DTYPE* AAptr = (DTYPE *) AA;
        DTYPE* BBptr = (DTYPE *) BB;
        DTYPE* CCptr = (DTYPE *) CC;

        auto start = GETTIME;

	DTYPE multiplier = rand();

        #pragma omp parallel for
        for (int y = 1; y < size; y++) {
		
#if INTENSITY1
		{
			functionI(BBptr[y+8], BBptr[y+9], multiplier);
		}
		{
			functionI(BBptr[y+6], BBptr[y+7], multiplier);
		}
		{
			functionI(BBptr[y+4], BBptr[y+5], multiplier);
		}
		{
			functionI(BBptr[y+2], BBptr[y+3], multiplier);
		}
		{
			functionI(BBptr[y], BBptr[y+1], multiplier);
		}
		{
			functionI(AAptr[y], BBptr[y-1], multiplier);
		}
#elif INTENSITY2
		{
			functionII(BBptr[y+8], BBptr[y+9], multiplier);
		}
		{
			functionII(BBptr[y+6], BBptr[y+7], multiplier);
		}
		{
			functionII(BBptr[y+4], BBptr[y+5], multiplier);
		}
		{
			functionII(BBptr[y+2], BBptr[y+3], multiplier);
		}
		{
			functionII(BBptr[y], BBptr[y+1], multiplier);
		}
		{
			functionII(AAptr[y], BBptr[y-1], multiplier);
		}
#elif INTENSITY3
		{
			functionIII(BBptr[y+8], BBptr[y+9], multiplier);
		}
		{
			functionIII(BBptr[y+6], BBptr[y+7], multiplier);
		}
		{
			functionIII(BBptr[y+4], BBptr[y+5], multiplier);
		}
		{
			functionIII(BBptr[y+2], BBptr[y+3], multiplier);
		}
		{		
			functionIII(BBptr[y], BBptr[y+1], multiplier);
		}
		{
			functionIII(AAptr[y], BBptr[y-1], multiplier);
		}
#elif INTENSITY4
		{
			functionIV(BBptr[y+8], BBptr[y+9], multiplier);
		}
		{
			functionIV(BBptr[y+6], BBptr[y+7], multiplier);
		}
		{
			functionIV(BBptr[y+4], BBptr[y+5], multiplier);
		}
		{
			functionIV(BBptr[y+2], BBptr[y+3], multiplier);
		}
		{
			functionIV(BBptr[y], BBptr[y+1], multiplier);
		}
		{
			functionIV(AAptr[y], BBptr[y-1], multiplier);
		}
#elif INTENSITY5
		{
			functionIII(BBptr[y+8], BBptr[y+9], multiplier);
		}
		{
			functionIII(BBptr[y+6], BBptr[y+7], multiplier);
		}
		{
			functionIII(BBptr[y+4], BBptr[y+5], multiplier);
		}
		{
			functionIII(BBptr[y+2], BBptr[y+3], multiplier);
		}
		{
			functionIII(BBptr[y], BBptr[y+1], multiplier);
		}
		{
			functionIII(AAptr[y], BBptr[y-1], multiplier);
		}
#endif
        }

        auto end = GETTIME;
        return DIFFTIME(end, start);
}
