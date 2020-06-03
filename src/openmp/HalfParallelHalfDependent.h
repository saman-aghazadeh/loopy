ulong HalfParallelHalfDependentSec1 (void *AA, void *BB, void *CC,
                              int size) {

        DTYPE* AAptr = (DTYPE *) AA;
        DTYPE* BBptr = (DTYPE *) BB;
        DTYPE* CCptr = (DTYPE *) CC;

        auto start = GETTIME;

        #pragma omp parallel for
        for (int y = 0; y < size; y++) {

#if INTENSITY1
        	functionI(AAptr[y], AAptr[y], BBptr[y]);
#elif INTENSITY2
                functionII(AAptr[y], AAptr[y], BBptr[y]);
#elif INTENSITY3
                functionIII(AAptr[y], AAptr[y], BBptr[y]);
#elif INTENSITY4        
               	functionIV(AAptr[y], AAptr[y], BBptr[y]);
#elif INTENSITY5
                functionV(AAptr[y], AAptr[y], BBptr[y]);
#endif
        }

        auto end = GETTIME;
        return DIFFTIME(end, start);
}

ulong HalfParallelHalfDependentSec2 (void *AA, void *BB, void *CC,
                              int size) {

        DTYPE* AAptr = (DTYPE *) AA;
        DTYPE* BBptr = (DTYPE *) BB;
        DTYPE* CCptr = (DTYPE *) CC;

        auto start = GETTIME;

        for (int y = 0; y < size; y++) {

#if INTENSITY1
        	functionI(AAptr[y], AAptr[y], BBptr[y]);
#elif INTENSITY2
                functionII(AAptr[y], AAptr[y], BBptr[y]);
#elif INTENSITY3
                functionIII(AAptr[y], AAptr[y], BBptr[y]);
#elif INTENSITY4        
               	functionIV(AAptr[y], AAptr[y], BBptr[y]);
#elif INTENSITY5
                functionV(AAptr[y], AAptr[y], BBptr[y]);
#endif
        }

        auto end = GETTIME;
        return DIFFTIME(end, start);
}


