#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_COPIES 19

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <loop_length_per_array>\n", argv[0]);
        return 1;
    }

    size_t N = atoll(argv[1]);
    size_t array_size = N * sizeof(double);
    
    // Allocate 19 source and 19 destination arrays
    double *src[NUM_COPIES];
    double *dst[NUM_COPIES];

    for (int i = 0; i < NUM_COPIES; i++) {
        src[i] = (double *)malloc(array_size);
        dst[i] = (double *)malloc(array_size);
        if (!src[i] || !dst[i]) {
            fprintf(stderr, "Memory allocation failed!\n");
            return 1;
        }
        // Warm up memory and prevent lazy allocation
	#pragma omp parallel for
        for (size_t j = 0; j < N; j++) {
            src[i][j] = (double)j;
            dst[i][j] = 0.0;
        }
    }

    //    printf("Benchmarking %d parallel copies with N=%zu...\n", NUM_COPIES, N);

    int NITER;
    double duration;

#undef NUM_COPIES
#define NUM_COPIES 1
    
    for(NITER = 1; ; NITER*=2) {
      double start_time = get_time();
      for(int k=0; k<NITER; k++) {
	// The Benchmark Loop: 19 concurrent streams
#pragma omp parallel for
	for (size_t j = 0; j < N; j++) {
	  dst[0][j]  = src[0][j];
#ifndef S01
#undef NUM_COPIES
#define NUM_COPIES 2
	  dst[1][j]  = src[1][j];
#ifndef S02
#undef NUM_COPIES
#define NUM_COPIES 3
	  dst[2][j]  = src[2][j];
#ifndef S03
#undef NUM_COPIES
#define NUM_COPIES 4
	  dst[3][j]  = src[3][j];
#ifndef S04
#undef NUM_COPIES
#define NUM_COPIES 5
	  dst[4][j]  = src[4][j];
#ifndef S05
#undef NUM_COPIES
#define NUM_COPIES 6
	  dst[5][j]  = src[5][j];
#ifndef S06
#undef NUM_COPIES
#define NUM_COPIES 7
	  dst[6][j]  = src[6][j];
#ifndef S07
#undef NUM_COPIES
#define NUM_COPIES 8
	  dst[7][j]  = src[7][j];
#ifndef S08
#undef NUM_COPIES
#define NUM_COPIES 9
	  dst[8][j]  = src[8][j];
#ifndef S09
#undef NUM_COPIES
#define NUM_COPIES 10
	  dst[9][j]  = src[9][j];
#ifndef S10
#undef NUM_COPIES
#define NUM_COPIES 11
	  dst[10][j] = src[10][j];
#ifndef S11
#undef NUM_COPIES
#define NUM_COPIES 12
	  dst[11][j] = src[11][j];
#ifndef S12
#undef NUM_COPIES
#define NUM_COPIES 13
	  dst[12][j] = src[12][j];
#ifndef S13
#undef NUM_COPIES
#define NUM_COPIES 14
	  dst[13][j] = src[13][j];
#ifndef S14
#undef NUM_COPIES
#define NUM_COPIES 15
	  dst[14][j] = src[14][j];
#ifndef S15
#undef NUM_COPIES
#define NUM_COPIES 16
	  dst[15][j] = src[15][j];
#ifndef S16
#undef NUM_COPIES
#define NUM_COPIES 17
	  dst[16][j] = src[16][j];
#ifndef S17
#undef NUM_COPIES
#define NUM_COPIES 18
	  dst[17][j] = src[17][j];
#ifndef S18
#undef NUM_COPIES
#define NUM_COPIES 19
	  dst[18][j] = src[18][j];
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
	}
	if(dst[1][N>>1]<0.) printf("%lf",dst[1][N>>1]);
      }
      
      double end_time = get_time();
      duration = end_time - start_time;

      if(duration > 2.0) break;
      //      printf("Doubling, NITER=%d\n",NITER);
    }

    // Performance Calculation
    // (19 reads + 19 writes) * N * 8 bytes
    double total_bytes = (double)NUM_COPIES * 2 * NITER * N * sizeof(double);
    double gb_per_sec = (total_bytes / (1000.0 * 1000.0 * 1000.0)) / duration;

    //    printf("Results (NUM_COPIES = %d):\n",NUM_COPIES);
    //    printf("  Duration:    %.6f seconds\n", duration);
    printf("NUM_COPIES: %d   Bandwidth:   %.2f GiB/s\n", NUM_COPIES, gb_per_sec);

    // Cleanup
    for (int i = 0; i < NUM_COPIES; i++) {
        free(src[i]);
        free(dst[i]);
    }

    return 0;
}
