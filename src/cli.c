/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of LBM-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "cli.h"

static int parseDimensions(const char *param, int *nX, int *nY, int *nZ)
{
  char *tmp;
  *nX = atoi(param);
  if (*nX <= 0) {
    printf("ERROR: X dimension must be > 0.\n");
    return 0;
  }
  tmp = strchr(param, 'x');
  if (tmp == NULL) {
    printf("ERROR: Y dimension is missing.\n");
    return 0;
  }
  *nY = atoi(tmp + 1);
  if (*nY <= 0) {
    printf("ERROR: Y dimension must be > 0.\n");
    return 0;
  }
  tmp = strchr(tmp + 1, 'x');
  if (tmp == NULL) {
    printf("ERROR: Z dimension is missing.\n");
    return 0;
  }
  *nZ = atoi(tmp + 1);
  if (*nZ <= 0) {
    printf("ERROR: Z dimension must be > 0.\n");
    return 0;
  }
  return 1;
}

void parseArguments(int argc,
    char **argv,
    int dims[3],
    const char **geometryType,
    char **kernelToUse,
    int *nThreads,
    int periodic[3],
    int *verify,
    CaseDataType *cd)
{
  int opt;
  while ((opt = getopt(argc, argv, BASE_ARGS)) != -1) {
    switch (opt) {
    case 'd':
      if (!parseDimensions(optarg, &dims[0], &dims[1], &dims[2]))
        exit(EXIT_FAILURE);
      break;
    case 'g':
      *geometryType = optarg;
      break;
    case 'i':
      cd->MaxIterations = (int)strtol(optarg, NULL, 0);
      if (cd->MaxIterations <= 0) {
        printf("ERROR: iterations must be > 0.\n");
        exit(EXIT_FAILURE);
      }
      break;
    case 'o':
      cd->Omega = F(strtod(optarg, NULL));
      break;
    case 'f':
      cd->XForce = F(strtod(optarg, NULL));
      break;
    case 'k':
      *kernelToUse = optarg;
      break;
    case 'l':
      printf("Available kernels:\n");
      for (int j = 0; j < G_N_KERNELS; ++j)
        printf("   %s\n", GKernels[j].Name);
      exit(EXIT_SUCCESS);
    case 'V':
#ifdef VERIFICATION
      *verify           = 1;
      cd->Omega         = F(1.0);
      cd->RhoIn         = F(1.0);
      cd->RhoOut        = F(1.0);
      *geometryType     = "box";
      dims[0]           = 16;
      dims[1]           = 16;
      dims[2]           = 16;
      cd->XForce        = F(0.00001);
      cd->MaxIterations = 1000;
      periodic[0]       = 1;
      periodic[1]       = 1;
      periodic[2]       = 0;
      printf("#\n# VERIFICATION: verifying flow profile of channel flow.\n#\n");
#else
      printf("ERROR: recompile with VERIFICATION=on to use -V.\n");
      exit(EXIT_FAILURE);
#endif
      break;
    case 't':
#ifdef _OPENMP
      *nThreads = atoi(optarg);
      if (*nThreads <= 0) {
        printf("ERROR: threads must be > 0.\n");
        exit(EXIT_FAILURE);
      }
#else
      printf("ERROR: threads requires OpenMP.\n");
      exit(EXIT_FAILURE);
#endif
      break;
    case 'x':
      periodic[0] = 1;
      break;
    case 'y':
      periodic[1] = 1;
      break;
    case 'z':
      periodic[2] = 1;
      break;
    case 'h':
      printf(HELPTEXT);
      exit(EXIT_SUCCESS);
    default:
      printf("ERROR: unknown option.\n");
      printf(HELPTEXT);
      exit(EXIT_FAILURE);
    }
  }
}
