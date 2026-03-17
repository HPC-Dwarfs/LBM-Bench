/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of LBM-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __LBM_H_
#define __LBM_H_

#include <stdlib.h>

typedef double PdfT;

#define F(number) (PdfT)(number)

#define N_D3Q19 19

#define D3Q19_N  0
#define D3Q19_S  1
#define D3Q19_E  2
#define D3Q19_W  3
#define D3Q19_NE 4
#define D3Q19_SE 5
#define D3Q19_NW 6
#define D3Q19_SW 7
#define D3Q19_T  8
#define D3Q19_TN 9
#define D3Q19_TE 10
#define D3Q19_TW 11
#define D3Q19_TS 12
#define D3Q19_B  13
#define D3Q19_BS 14
#define D3Q19_BN 15
#define D3Q19_BW 16
#define D3Q19_BE 17
#define D3Q19_C  18

extern int D3Q19_X[N_D3Q19];
extern int D3Q19_Y[N_D3Q19];
extern int D3Q19_Z[N_D3Q19];
extern int D3Q19_INV[N_D3Q19];

typedef int LatticeT;

enum {
  LAT_CELL_OBSTACLE = 0,
  LAT_CELL_FLUID = 1,
  LAT_CELL_INLET = 2,
  LAT_CELL_OUTLET = 4
};

typedef struct LatticeDesc {
  int Dims[3];
  LatticeT *Lattice;
  int nCells;
  int nFluid;
  int nObst;
  int nInlet;
  int nOutlet;
  int PeriodicX;
  int PeriodicY;
  int PeriodicZ;
  const char *Name;
} LatticeDesc;

typedef struct CaseData {
  PdfT Omega;
  PdfT RhoIn;
  PdfT RhoOut;
  PdfT XForce;
  int MaxIterations;
  int StatisticsModulus;
} CaseData;

typedef struct KernelData {
  PdfT *Pdfs[2];
  PdfT *PdfsActive;
  int Dims[3];
  int GlobalDims[3];
  int Offsets[3];
  int *BounceBackPdfsSrc;
  int *BounceBackPdfsDst;
  int nBounceBackPdfs;
  int Blk[3];
  int Iteration;
  double LoopBalance;
  double Duration;
  void (*GetNode)(struct KernelData *kd, int x, int y, int z, PdfT *pdfs);
  void (*SetNode)(struct KernelData *kd, int x, int y, int z, PdfT *pdfs);
  void (*BoundaryConditionsGetPdf)(struct KernelData *kd, int x, int y, int z,
                                   int dir, PdfT *pdf);
  void (*BoundaryConditionsSetPdf)(struct KernelData *kd, int x, int y, int z,
                                   int dir, PdfT pdf);
  void (*Kernel)(LatticeDesc *ld, struct KernelData *kd, CaseData *cd);
} KernelData;

typedef struct KernelFunctions {
  const char *Name;
  void (*Init)(LatticeDesc *ld, KernelData **kd, CaseData *cd);
  void (*Deinit)(LatticeDesc *ld, KernelData **kd);
} KernelFunctions;

extern KernelFunctions g_kernels[];
extern const int g_nKernels;

static inline int latticeIndex(int dims[3], int x, int y, int z) {
  return z * dims[0] * dims[1] + y * dims[0] + x;
}

static inline int indexSoA(int gDims[3], int x, int y, int z, int d) {
  return d * gDims[0] * gDims[1] * gDims[2] + x * gDims[1] * gDims[2] +
         y * gDims[2] + z;
}

static inline int indexAoS(int gDims[3], int x, int y, int z, int d) {
  return x * gDims[1] * gDims[2] * N_D3Q19 + y * gDims[2] * N_D3Q19 +
         z * N_D3Q19 + d;
}

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

void geometryCreate(const char *type, int dims[3], int periodic[3],
                    LatticeDesc *ld);
void kernelInit(LatticeDesc *ld, KernelData **kd, int propModel,
                int dataLayout);
void kernelDeinit(LatticeDesc *ld, KernelData **kd);
void kernelSetInitialDensity(LatticeDesc *ld, KernelData *kd, CaseData *cd);
void kernelSetInitialVelocity(LatticeDesc *ld, KernelData *kd, CaseData *cd);
void kernelComputeBoundaryConditions(KernelData *kd, LatticeDesc *ld,
                                     CaseData *cd);
void kernelAddBodyForce(KernelData *kd, LatticeDesc *ld, CaseData *cd);
PdfT kernelDensity(KernelData *kd, LatticeDesc *ld);
void kernelStatistics(KernelData *kd, LatticeDesc *ld, CaseData *cd, int iter);
void kernelVerify(LatticeDesc *ld, KernelData *kd, CaseData *cd,
                  PdfT *errorNorm);

enum { PROP_PUSH = 0, PROP_PULL = 1, PROP_AA = 2 };
enum { LAYOUT_SOA = 0, LAYOUT_AOS = 1 };

void kernelInitPushSoA(LatticeDesc *ld, KernelData **kd, CaseData *cd);
void kernelInitPushAoS(LatticeDesc *ld, KernelData **kd, CaseData *cd);
void kernelInitPullSoA(LatticeDesc *ld, KernelData **kd, CaseData *cd);
void kernelInitPullAoS(LatticeDesc *ld, KernelData **kd, CaseData *cd);
void kernelInitBlkPushSoA(LatticeDesc *ld, KernelData **kd, CaseData *cd);
void kernelInitBlkPullSoA(LatticeDesc *ld, KernelData **kd, CaseData *cd);
void kernelInitAaSoA(LatticeDesc *ld, KernelData **kd, CaseData *cd);

void kernelPushSoA(LatticeDesc *ld, KernelData *kd, CaseData *cd);
void kernelPushAoS(LatticeDesc *ld, KernelData *kd, CaseData *cd);
void kernelPullSoA(LatticeDesc *ld, KernelData *kd, CaseData *cd);
void kernelPullAoS(LatticeDesc *ld, KernelData *kd, CaseData *cd);
void kernelBlkPushSoA(LatticeDesc *ld, KernelData *kd, CaseData *cd);
void kernelBlkPullSoA(LatticeDesc *ld, KernelData *kd, CaseData *cd);
void kernelAaSoA(LatticeDesc *ld, KernelData *kd, CaseData *cd);

#endif
