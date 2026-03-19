/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of LBM-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cli.h"
#include "lbm.h"
#include "timing.h"

int main(int argc, char *argv[])
{
  int dims[3]              = { 20, 20, 20 };
  const char *geometryType = "channel";
  int verify               = 0;
  char *kernelToUse        = "push-soa";
  int nThreads             = 1;
  int periodic[3]          = { 0, 0, 0 };

  CaseDataType cd;
  cd.MaxIterations     = 10;
  cd.RhoIn             = F(1.0);
  cd.RhoOut            = F(1.0);
  cd.Omega             = F(1.0);
  cd.XForce            = F(0.00001);
  cd.StatisticsModulus = 100;

  printf("LBM Benchmark Kernels, compiled %s %s, type: %s\n",
      __DATE__,
      __TIME__,
#ifdef VERIFICATION
      "verification"
#else
      "benchmark"
#endif
  );

  parseArguments(
      argc, argv, dims, &geometryType, &kernelToUse, &nThreads, periodic, &verify, &cd);

#ifdef _OPENMP
  omp_set_num_threads(nThreads);
  (void)nThreads;
#else
  (void)nThreads;
#endif

  // Create geometry
  LatticeDescType ld;
  geometryCreate(geometryType, dims, periodic, &ld);

  printf("#\n");
  printf("# - iterations:        %d\n", cd.MaxIterations);
  printf("# - geometry:\n");
  printf("#   type:              %s\n", ld.Name);
  printf("#   dimensions:        %d x %d x %d\n", ld.Dims[0], ld.Dims[1], ld.Dims[2]);
  printf("#   nodes total:       %10d\n", ld.nObst + ld.nFluid);
  printf("#   nodes fluid:       %10d\n", ld.nFluid);
  printf("#   nodes obstacles:   %10d\n", ld.nObst);
  printf("#   periodicity:       x: %d y: %d z: %d\n",
      ld.PeriodicX,
      ld.PeriodicY,
      ld.PeriodicZ);
  printf("# - flow:\n");
  printf("#   omega:             %f\n", cd.Omega);
  printf("#   rho in:            %e\n", cd.RhoIn);
  printf("#   rho out:           %e\n", cd.RhoOut);
#ifdef _OPENMP
  printf("# - OpenMP threads:    %d\n", omp_get_max_threads());
#endif

  // Find kernel
  KernelFunctionsType *kf = NULL;
  for (int j = 0; j < G_N_KERNELS; ++j) {
    if (!strcasecmp(kernelToUse, GKernels[j].Name)) {
      kf = &GKernels[j];
      break;
    }
  }

  if (kf == NULL) {
    printf("ERROR: kernel \"%s\" not found.\n", kernelToUse);
    return 1;
  }

  printf("#\n# - kernel:            %s\n#\n", kf->Name);

  // Initialize kernel
  KernelDataType *kd = NULL;
  kf->Init(&ld, &kd, &cd);

#ifdef VERIFICATION
  if (verify) {
    kernelSetInitialDensity(&ld, kd, &cd);
    kernelSetInitialVelocity(&ld, kd, &cd);
  }
#endif

  printf("# starting kernel...\n");

  // Run kernel
  kd->Kernel(&ld, kd, &cd);

  // Statistics
  kernelStatistics(kd, &ld, &cd, cd.MaxIterations);

#ifdef VERIFICATION
  PdfType errorNorm = -1.0;
  kernelVerify(&ld, kd, &cd, &errorNorm);
#endif

  double duration    = kd->Duration;
  double loopBalance = kd->LoopBalance;
  double dataVolGByte =
      loopBalance * ld.nFluid * cd.MaxIterations / 1024.0 / 1024.0 / 1024.0;
  double bandwidthGBytePerS = dataVolGByte / duration;
  double perf = (double)ld.nFluid * (double)cd.MaxIterations / duration / 1.0e6;

  // Deinitialize kernel
  kf->Deinit(&ld, &kd);

  printf("#\n");
  printf("# Evaluation Stats\n");
  printf("#   runtime:           \t%.3f s\n", duration);
  printf("#   iterations:        \t%d\n", cd.MaxIterations);
  printf("#   fluid cells:       \t%d\n", ld.nFluid);
  printf("# Derived metrics\n");
  printf("#   MEM data vol.:     \t%.2f GByte\n", dataVolGByte);
  printf("#   MEM bandwidth:     \t%.2f GByte/s\n", bandwidthGBytePerS);
  printf("#   performance:       \t%.3f MFLUP/s\n", perf);

  printf("P:   %f MFLUP/s  d: %f s  iter: %d  fnodes: %f x1e6  geo: %s  "
         "kernel: %s\n",
      perf,
      duration,
      cd.MaxIterations,
      ld.nFluid / 1e6,
      geometryType,
      kernelToUse);

  int exitCode = 0;
#ifdef VERIFICATION
  if (verify) {
    printf("# VERIFICATION: deviation from analytical solution: %e\n", errorNorm);
    if (errorNorm > 0.1) {
      printf("# VERIFICATION FAILED.\n");
      exitCode = 1;
    } else {
      printf("# VERIFICATION SUCCEEDED.\n");
    }
  }
#endif

  free(ld.Lattice);
  return exitCode;
}
