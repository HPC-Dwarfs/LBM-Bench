/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of LBM-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __LBM_H_
#define __LBM_H_

#include <stdlib.h>

#ifdef PRECISION_SP
typedef float PdfType;
#else
typedef double PdfType;
#endif

#define F(number) (PdfType)(number)

#define N_D3Q19 19

#define D3Q19_N 0
#define D3Q19_S 1
#define D3Q19_E 2
#define D3Q19_W 3
#define D3Q19_NE 4
#define D3Q19_SE 5
#define D3Q19_NW 6
#define D3Q19_SW 7
#define D3Q19_T 8
#define D3Q19_TN 9
#define D3Q19_TE 10
#define D3Q19_TW 11
#define D3Q19_TS 12
#define D3Q19_B 13
#define D3Q19_BS 14
#define D3Q19_BN 15
#define D3Q19_BW 16
#define D3Q19_BE 17
#define D3Q19_C 18

extern int D3Q19X[N_D3Q19];
extern int D3Q19Y[N_D3Q19];
extern int D3Q19Z[N_D3Q19];
extern int D3Q19Inv[N_D3Q19];

typedef int LatticeType;

enum {
  LAT_CELL_OBSTACLE = 0,
  LAT_CELL_FLUID    = 1,
  LAT_CELL_INLET    = 2,
  LAT_CELL_OUTLET   = 4
};

typedef struct LatticeDesc {
  int Dims[3];
  LatticeType *Lattice;
  int nCells;
  int nFluid;
  int nObst;
  int nInlet;
  int nOutlet;
  int PeriodicX;
  int PeriodicY;
  int PeriodicZ;
  const char *Name;
} LatticeDescType;

typedef struct CaseData {
  PdfType Omega;
  PdfType RhoIn;
  PdfType RhoOut;
  PdfType XForce;
  int MaxIterations;
  int StatisticsModulus;
} CaseDataType;

typedef struct KernelData {
  PdfType *Pdfs[2];
  PdfType *PdfsActive;
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
  void (*GetNode)(struct KernelData *kd, int x, int y, int z, PdfType *pdfs);
  void (*SetNode)(struct KernelData *kd, int x, int y, int z, PdfType *pdfs);
  void (*BoundaryConditionsGetPdf)(
      struct KernelData *kd, int x, int y, int z, int dir, PdfType *pdf);
  void (*BoundaryConditionsSetPdf)(
      struct KernelData *kd, int x, int y, int z, int dir, PdfType pdf);
  void (*Kernel)(LatticeDescType *ld, struct KernelData *kd, CaseDataType *cd);
} KernelDataType;

typedef struct KernelFunctions {
  const char *Name;
  void (*Init)(LatticeDescType *ld, KernelDataType **kd, CaseDataType *cd);
  void (*Deinit)(LatticeDescType *ld, KernelDataType **kd);
} KernelFunctionsType;

extern KernelFunctionsType GKernels[];
extern const int G_N_KERNELS;

static inline int latticeIndex(int dims[3], int x, int y, int z)
{
  return z * dims[0] * dims[1] + y * dims[0] + x;
}

static inline int indexSoA(int gDims[3], int x, int y, int z, int d)
{
  return d * gDims[0] * gDims[1] * gDims[2] + x * gDims[1] * gDims[2] + y * gDims[2] + z;
}

static inline int indexAoS(int gDims[3], int x, int y, int z, int d)
{
  return x * gDims[1] * gDims[2] * N_D3Q19 + y * gDims[2] * N_D3Q19 + z * N_D3Q19 + d;
}

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

void geometryCreate(const char *type, int dims[3], int periodic[3], LatticeDescType *ld);
void kernelInit(LatticeDescType *ld, KernelDataType **kd, int propModel, int dataLayout);
void kernelDeinit(LatticeDescType *ld, KernelDataType **kd);
void kernelSetInitialDensity(LatticeDescType *ld, KernelDataType *kd, CaseDataType *cd);
void kernelSetInitialVelocity(LatticeDescType *ld, KernelDataType *kd, CaseDataType *cd);
void kernelComputeBoundaryConditions(
    KernelDataType *kd, LatticeDescType *ld, CaseDataType *cd);
void kernelAddBodyForce(KernelDataType *kd, LatticeDescType *ld, CaseDataType *cd);
PdfType kernelDensity(KernelDataType *kd, LatticeDescType *ld);
void kernelStatistics(
    KernelDataType *kd, LatticeDescType *ld, CaseDataType *cd, int iter);
void kernelVerify(
    LatticeDescType *ld, KernelDataType *kd, CaseDataType *cd, PdfType *errorNorm);

enum { PROP_PUSH = 0, PROP_PULL = 1, PROP_AA = 2 };
enum { LAYOUT_SOA = 0, LAYOUT_AOS = 1 };

void kernelInitPushSoA(LatticeDescType *ld, KernelDataType **kd, CaseDataType *cd);
void kernelInitPushAoS(LatticeDescType *ld, KernelDataType **kd, CaseDataType *cd);
void kernelInitPullSoA(LatticeDescType *ld, KernelDataType **kd, CaseDataType *cd);
void kernelInitPullAoS(LatticeDescType *ld, KernelDataType **kd, CaseDataType *cd);
void kernelInitBlkPushSoA(LatticeDescType *ld, KernelDataType **kd, CaseDataType *cd);
void kernelInitBlkPullSoA(LatticeDescType *ld, KernelDataType **kd, CaseDataType *cd);
void kernelInitAaSoA(LatticeDescType *ld, KernelDataType **kd, CaseDataType *cd);

void kernelPushSoA(LatticeDescType *ld, KernelDataType *kd, CaseDataType *cd);
void kernelPushAoS(LatticeDescType *ld, KernelDataType *kd, CaseDataType *cd);
void kernelPullSoA(LatticeDescType *ld, KernelDataType *kd, CaseDataType *cd);
void kernelPullAoS(LatticeDescType *ld, KernelDataType *kd, CaseDataType *cd);
void kernelBlkPushSoA(LatticeDescType *ld, KernelDataType *kd, CaseDataType *cd);
void kernelBlkPullSoA(LatticeDescType *ld, KernelDataType *kd, CaseDataType *cd);
void kernelAaSoA(LatticeDescType *ld, KernelDataType *kd, CaseDataType *cd);

#endif
