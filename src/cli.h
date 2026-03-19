/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of LBM-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __CLI_H_
#define __CLI_H_

#include "lbm.h"

#define BASE_ARGS "d:g:i:o:f:k:lVt:xyzh"

#define HELPTEXT                                                                         \
  "Usage: lbmbench [options]\n"                                                          \
  "  -d XxYxZ             Geometry dimensions\n"                                         \
  "  -g TYPE              box|channel|pipe|blocks-N|fluid\n"                             \
  "  -i N                 Number of iterations\n"                                        \
  "  -o V                 Relaxation parameter (omega)\n"                                \
  "  -f V                 X-direction body force\n"                                      \
  "  -k NAME              Kernel to use\n"                                               \
  "  -l                   List available kernels\n"                                      \
  "  -V                   Run verification\n"                                            \
  "  -t N                 Number of threads\n"                                           \
  "  -x/-y/-z             Enable periodic BC in x/y/z\n"                                 \
  "  -h                   Print this help\n"

void parseArguments(int argc,
    char **argv,
    int dims[3],
    const char **geometryType,
    char **kernelToUse,
    int *nThreads,
    int periodic[3],
    int *verify,
    CaseDataType *cd);

#endif
