ulong intraDimensionDependent (void *AA, void *BB, void *CC,
                              int sizeY, int sizeX) {

        DTYPE* AAptr = (DTYPE *) AA;
        DTYPE* BBptr = (DTYPE *) BB;
        DTYPE* CCptr = (DTYPE *) CC;

        auto start = GETTIME;

        #pragma omp parallel for
        for (int y = 0; y < sizeY; y++) {

                DTYPE temp = AAptr[y];
                for (int x = 0; x < sizeX; x++) {
#if INTENSITY1
                        functionI(temp, temp, BBptr[x*sizeY+y]);
#elif INTENSITY2
                        functionII(temp, temp, BBptr[x*sizeY+y]);
#elif INTENSITY3
                        functionIII(temp, temp, BBptr[x*sizeY+y]);
#elif INTENSITY4        
                        functionIV(temp, temp, BBptr[x*sizeY+y]);
#elif INTENSITY5
                        functionV(temp, temp, BBptr[x*sizeY+y]);
#endif
                        AAptr[x*sizeY+y] = temp;
                }
        }

        auto end = GETTIME;
	return DIFFTIME(end, start);

}
