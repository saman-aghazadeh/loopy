ulong conditionalDependent2 (void *AA, void *BB, void *CC,
                            int size) {

        DTYPE* AAptr = (DTYPE *) AA;
        DTYPE* BBptr = (DTYPE *) BB;
        DTYPE* CCptr = (DTYPE *) CC;

        DTYPE multiplier = rand();
        DTYPE additive = rand();
        auto start = GETTIME;

        #pragma omp parallel for                
        for (int y = 0; y < size; y++) {

                if (y % 2 == 0) {
#if INTENSITY1
                        CfunctionI(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], BBptr[y], multiplier, additive);
#endif
                } else if (y % 2 == 1) {
#if INTENSITY1
                        CfunctionI(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], BBptr[y], multiplier, -additive);
#endif

                }

        }

	auto end = GETTIME;

	return DIFFTIME(end, start);
}

ulong conditionalDependent4 (void *AA, void *BB, void *CC,
                            int size) {

        DTYPE* AAptr = (DTYPE *) AA;
        DTYPE* BBptr = (DTYPE *) BB;
        DTYPE* CCptr = (DTYPE *) CC;

        DTYPE multiplier = rand();
        DTYPE additive = rand();
        auto start = GETTIME;

        #pragma omp parallel for                
        for (int y = 0; y < size; y++) {

                if (y % 4 == 0) {
#if INTENSITY1
                        CfunctionI(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], BBptr[y], multiplier, additive);
#endif
                } else if (y % 4 == 1) {
#if INTENSITY1
                        CfunctionI(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], BBptr[y], multiplier, -additive);
#endif

                } else if (y % 4 == 2) {
#if INTENSITY1
                        CfunctionI(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], BBptr[y], -multiplier, -additive);
#endif

                } else if (y % 4 == 3) {
#if INTENSITY1
                        CfunctionI(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], BBptr[y], -multiplier, -additive);
#endif

                }
        }

	auto end = GETTIME;

	return DIFFTIME(end, start);
}


ulong conditionalDependent6 (void *AA, void *BB, void *CC,
                            int size) {

        DTYPE* AAptr = (DTYPE *) AA;
        DTYPE* BBptr = (DTYPE *) BB;
        DTYPE* CCptr = (DTYPE *) CC;

        DTYPE multiplier = rand();
        DTYPE additive = rand();
        auto start = GETTIME;

        #pragma omp parallel for                
        for (int y = 0; y < size; y++) {

                if (y % 6 == 0) {
#if INTENSITY1
                        CfunctionI(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], BBptr[y], multiplier, additive);
#endif
                } else if (y % 6 == 1) {
#if INTENSITY1
                        CfunctionI(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], BBptr[y], multiplier, -additive);
#endif
                } else if (y % 6 == 2) {
#if INTENSITY1
                        CfunctionI(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], BBptr[y], -multiplier, -additive);
#endif

                } else if (y % 6 == 3) {
#if INTENSITY1
                        CfunctionI(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], BBptr[y], -multiplier, -additive);
#endif
                } else if (y % 6 == 4) {
#if INTENSITY1
                        CfunctionI(AAptr[y], CCptr[y], multiplier, additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], CCptr[y], multiplier, additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], CCptr[y], multiplier, additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], CCptr[y], multiplier, additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], CCptr[y], multiplier, additive);
#endif
		} else if (y % 6 == 5) {
#if INTENSITY1
                        CfunctionI(AAptr[y], CCptr[y], multiplier, -additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], CCptr[y], multiplier, -additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], CCptr[y], multiplier, -additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], CCptr[y], multiplier, -additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], CCptr[y], multiplier, -additive);
#endif
		}

        }
	auto end = GETTIME;

	return DIFFTIME(end, start);
}

ulong conditionalDependent8 (void *AA, void *BB, void *CC,
                            int size) {

        DTYPE* AAptr = (DTYPE *) AA;
        DTYPE* BBptr = (DTYPE *) BB;
        DTYPE* CCptr = (DTYPE *) CC;

        DTYPE multiplier = rand();
        DTYPE additive = rand();
        auto start = GETTIME;

        #pragma omp parallel for                
        for (int y = 0; y < size; y++) {

                if (y % 8 == 0) {
#if INTENSITY1
                        CfunctionI(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], BBptr[y], multiplier, additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], BBptr[y], multiplier, additive);
#endif
                } else if (y % 8 == 1) {
#if INTENSITY1
                        CfunctionI(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], BBptr[y], multiplier, -additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], BBptr[y], multiplier, -additive);
#endif
                } else if (y % 8 == 2) {
#if INTENSITY1
                        CfunctionI(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], BBptr[y], -multiplier, -additive);
#endif

                } else if (y % 8 == 3) {
#if INTENSITY1
                        CfunctionI(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], BBptr[y], -multiplier, -additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], BBptr[y], -multiplier, -additive);
#endif
                } else if (y % 8 == 4) {
#if INTENSITY1
                        CfunctionI(AAptr[y], CCptr[y], multiplier, additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], CCptr[y], multiplier, additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], CCptr[y], multiplier, additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], CCptr[y], multiplier, additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], CCptr[y], multiplier, additive);
#endif
		} else if (y % 8 == 5) {
#if INTENSITY1
                        CfunctionI(AAptr[y], CCptr[y], multiplier, -additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], CCptr[y], multiplier, -additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], CCptr[y], multiplier, -additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], CCptr[y], multiplier, -additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], CCptr[y], multiplier, -additive);
#endif
		} else if (y % 8 == 6) {
#if INTENSITY1
                        CfunctionI(AAptr[y], CCptr[y], -multiplier, -additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], CCptr[y], -multiplier, -additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], CCptr[y], -multiplier, -additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], CCptr[y], -multiplier, -additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], CCptr[y], -multiplier, -additive);
#endif

		} else if (y % 8 == 7) {
#if INTENSITY1
                        CfunctionI(AAptr[y], CCptr[y], -multiplier, additive);
#elif INTENSITY2
                        CfunctionII(AAptr[y], CCptr[y], -multiplier, additive);
#elif INTENSITY3
                        CfunctionIII(AAptr[y], CCptr[y], -multiplier, additive);
#elif INTENSITY4
                        CfunctionIV(AAptr[y], CCptr[y], -multiplier, additive);
#elif INTENSITY5
                        CfunctionV(AAptr[y], CCptr[y], -multiplier, additive);
#endif
		}

        }
	auto end = GETTIME;

	return DIFFTIME(end, start);
}


