/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of LBM-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "allocate.h"
#include "lbm.h"

// D3Q19 velocity vectors
// clang-format off
int D3Q19_X[N_D3Q19] = { 0,  0,  1, -1,  1,  1, -1, -1,  0,  0,  1, -1,  0,  0,  0,  0, -1,  1,  0};
int D3Q19_Y[N_D3Q19] = { 1, -1,  0,  0,  1, -1,  1, -1,  0,  1,  0,  0, -1,  0, -1,  1,  0,  0,  0};
int D3Q19_Z[N_D3Q19] = { 0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  0};
int D3Q19_INV[N_D3Q19] = {
  D3Q19_S,  D3Q19_N,  D3Q19_W,  D3Q19_E,
  D3Q19_SW, D3Q19_NW, D3Q19_SE, D3Q19_NE,
  D3Q19_B,  D3Q19_BS, D3Q19_BW, D3Q19_BE, D3Q19_BN,
  D3Q19_T,  D3Q19_TN, D3Q19_TS, D3Q19_TE, D3Q19_TW,
  D3Q19_C
};
// clang-format on

// Kernel function table
KernelFunctions g_kernels[] = {
    {"pull-soa", kernelInitPullSoA, kernelDeinit},
    {"pull-aos", kernelInitPullAoS, kernelDeinit},
    {"push-soa", kernelInitPushSoA, kernelDeinit},
    {"push-aos", kernelInitPushAoS, kernelDeinit},
    {"blk-push-soa", kernelInitBlkPushSoA, kernelDeinit},
    {"blk-pull-soa", kernelInitBlkPullSoA, kernelDeinit},
    {"aa-soa", kernelInitAaSoA, kernelDeinit},
};

const int g_nKernels = sizeof(g_kernels) / sizeof(g_kernels[0]);

// -----------------------------------------------------------------------
// Geometry creation
// -----------------------------------------------------------------------

static const char *g_geoTypeStr[] = {"box", "channel", "pipe", "blocks",
                                     "fluid"};
enum { GEO_BOX = 0, GEO_CHANNEL = 1, GEO_PIPE = 2, GEO_BLOCKS = 3,
       GEO_FLUID = 4 };

void geometryCreate(const char *geometryType, int dims[3], int periodic[3],
                    LatticeDesc *ld) {
  int type = -1;
  int blockSize = 8;

  if (strncasecmp("channel", geometryType, 7) == 0) {
    type = GEO_CHANNEL;
  } else if (strncasecmp("box", geometryType, 3) == 0) {
    type = GEO_BOX;
  } else if (strncasecmp("pipe", geometryType, 4) == 0) {
    type = GEO_PIPE;
  } else if (strncasecmp("blocks", geometryType, 6) == 0) {
    type = GEO_BLOCKS;
    if (strlen(geometryType) > 7) {
      blockSize = atoi(&geometryType[7]);
      if (blockSize <= 0) blockSize = 8;
    }
  } else if (strncasecmp("fluid", geometryType, 5) == 0) {
    type = GEO_FLUID;
  } else {
    printf("ERROR: unknown geometry type '%s'.\n", geometryType);
    exit(1);
  }

  ld->Dims[0] = dims[0];
  ld->Dims[1] = dims[1];
  ld->Dims[2] = dims[2];
  ld->nCells = dims[0] * dims[1] * dims[2];
  ld->PeriodicX = periodic[0];
  ld->PeriodicY = periodic[1];
  ld->PeriodicZ = periodic[2];
  ld->Name = g_geoTypeStr[type];

  LatticeT *lattice =
      (LatticeT *)allocate(64, sizeof(LatticeT) * ld->nCells);

  ld->Lattice = lattice;

  for (int z = 0; z < dims[2]; ++z)
    for (int y = 0; y < dims[1]; ++y)
      for (int x = 0; x < dims[0]; ++x)
        lattice[latticeIndex(dims, x, y, z)] = LAT_CELL_FLUID;

  if (type == GEO_CHANNEL || type == GEO_BLOCKS || type == GEO_PIPE)
    periodic[0] = 1;
  else if (type == GEO_FLUID) {
    periodic[0] = 1;
    periodic[1] = 1;
    periodic[2] = 1;
  }

  // X boundaries
  for (int z = 0; z < dims[2]; ++z)
    for (int y = 0; y < dims[1]; ++y) {
      if (!periodic[0]) {
        lattice[latticeIndex(dims, 0, y, z)] = LAT_CELL_OBSTACLE;
        lattice[latticeIndex(dims, dims[0] - 1, y, z)] = LAT_CELL_OBSTACLE;
      }
    }

  // Y boundaries
  for (int z = 0; z < dims[2]; ++z)
    for (int x = 0; x < dims[0]; ++x) {
      if (!periodic[1]) {
        lattice[latticeIndex(dims, x, 0, z)] = LAT_CELL_OBSTACLE;
        lattice[latticeIndex(dims, x, dims[1] - 1, z)] = LAT_CELL_OBSTACLE;
      }
    }

  // Z boundaries
  for (int y = 0; y < dims[1]; ++y)
    for (int x = 0; x < dims[0]; ++x) {
      if (!periodic[2]) {
        lattice[latticeIndex(dims, x, y, 0)] = LAT_CELL_OBSTACLE;
        lattice[latticeIndex(dims, x, y, dims[2] - 1)] = LAT_CELL_OBSTACLE;
      }
    }

  if (type == GEO_PIPE) {
#define SQR(a) ((a) * (a))
    double centerZ = dims[2] / 2.0 - 0.5;
    double centerY = dims[1] / 2.0 - 0.5;
    double minDiameter = MIN(dims[1], dims[2]);
    double minRadiusSquared = SQR(minDiameter / 2 - 1);
    for (int z = 0; z < dims[2]; ++z)
      for (int y = 0; y < dims[1]; ++y)
        if ((SQR(z - centerZ) + SQR(y - centerY)) >= minRadiusSquared)
          for (int x = 0; x < dims[0]; ++x)
            lattice[latticeIndex(dims, x, y, z)] = LAT_CELL_OBSTACLE;
#undef SQR
  } else if (type == GEO_BLOCKS && blockSize > 0) {
    int nb = blockSize;
    for (int z = 0; z < dims[2]; ++z) {
      if ((z % (2 * nb)) < nb) continue;
      for (int y = 0; y < dims[1]; ++y) {
        if ((y % (2 * nb)) < nb) continue;
        for (int x = 0; x < dims[0]; ++x)
          if ((x % (2 * nb)) >= nb)
            lattice[latticeIndex(dims, x, y, z)] = LAT_CELL_OBSTACLE;
      }
    }
  }

  ld->PeriodicX = periodic[0];
  ld->PeriodicY = periodic[1];
  ld->PeriodicZ = periodic[2];

  ld->nObst = 0;
  ld->nFluid = 0;
  ld->nInlet = 0;
  ld->nOutlet = 0;

  for (int z = 0; z < dims[2]; ++z)
    for (int y = 0; y < dims[1]; ++y)
      for (int x = 0; x < dims[0]; ++x) {
        switch (lattice[latticeIndex(dims, x, y, z)]) {
        case LAT_CELL_OBSTACLE: ld->nObst++; break;
        case LAT_CELL_FLUID: ld->nFluid++; break;
        case LAT_CELL_INLET:
          ld->nInlet++;
          ld->nFluid++;
          break;
        case LAT_CELL_OUTLET:
          ld->nOutlet++;
          ld->nFluid++;
          break;
        }
      }
}

// -----------------------------------------------------------------------
// GetNode / SetNode helpers for SoA and AoS layouts (push and pull)
// -----------------------------------------------------------------------

static void getNodePushSoA(KernelData *kd, int x, int y, int z, PdfT *pdfs) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  int *gDims = kd->GlobalDims;
  for (int d = 0; d < N_D3Q19; ++d)
    pdfs[d] = kd->PdfsActive[indexSoA(gDims, x + oX, y + oY, z + oZ, d)];
}

static void setNodePushSoA(KernelData *kd, int x, int y, int z, PdfT *pdfs) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  int *gDims = kd->GlobalDims;
  for (int d = 0; d < N_D3Q19; ++d)
    kd->PdfsActive[indexSoA(gDims, x + oX, y + oY, z + oZ, d)] = pdfs[d];
}

static void getNodePullSoA(KernelData *kd, int x, int y, int z, PdfT *pdfs) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  int *gDims = kd->GlobalDims;
  for (int d = 0; d < N_D3Q19; ++d)
    pdfs[d] = kd->PdfsActive[indexSoA(gDims, x + oX - D3Q19_X[d],
                                      y + oY - D3Q19_Y[d],
                                      z + oZ - D3Q19_Z[d], d)];
}

static void setNodePullSoA(KernelData *kd, int x, int y, int z, PdfT *pdfs) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  int *gDims = kd->GlobalDims;
  for (int d = 0; d < N_D3Q19; ++d)
    kd->PdfsActive[indexSoA(gDims, x + oX - D3Q19_X[d], y + oY - D3Q19_Y[d],
                            z + oZ - D3Q19_Z[d], d)] = pdfs[d];
}

static void getNodePushAoS(KernelData *kd, int x, int y, int z, PdfT *pdfs) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  int *gDims = kd->GlobalDims;
  for (int d = 0; d < N_D3Q19; ++d)
    pdfs[d] = kd->PdfsActive[indexAoS(gDims, x + oX, y + oY, z + oZ, d)];
}

static void setNodePushAoS(KernelData *kd, int x, int y, int z, PdfT *pdfs) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  int *gDims = kd->GlobalDims;
  for (int d = 0; d < N_D3Q19; ++d)
    kd->PdfsActive[indexAoS(gDims, x + oX, y + oY, z + oZ, d)] = pdfs[d];
}

static void getNodePullAoS(KernelData *kd, int x, int y, int z, PdfT *pdfs) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  int *gDims = kd->GlobalDims;
  for (int d = 0; d < N_D3Q19; ++d)
    pdfs[d] = kd->PdfsActive[indexAoS(gDims, x + oX - D3Q19_X[d],
                                      y + oY - D3Q19_Y[d],
                                      z + oZ - D3Q19_Z[d], d)];
}

static void setNodePullAoS(KernelData *kd, int x, int y, int z, PdfT *pdfs) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  int *gDims = kd->GlobalDims;
  for (int d = 0; d < N_D3Q19; ++d)
    kd->PdfsActive[indexAoS(gDims, x + oX - D3Q19_X[d], y + oY - D3Q19_Y[d],
                            z + oZ - D3Q19_Z[d], d)] = pdfs[d];
}

// AA SoA GetNode/SetNode: iteration-dependent
static void getNodeAaSoA(KernelData *kd, int x, int y, int z, PdfT *pdfs) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  int *gDims = kd->GlobalDims;
  if (kd->Iteration % 2 == 0) {
    for (int d = 0; d < N_D3Q19; ++d)
      pdfs[d] = kd->PdfsActive[indexSoA(gDims, x + oX - D3Q19_X[d],
                                        y + oY - D3Q19_Y[d],
                                        z + oZ - D3Q19_Z[d], D3Q19_INV[d])];
  } else {
    for (int d = 0; d < N_D3Q19; ++d)
      pdfs[d] = kd->PdfsActive[indexSoA(gDims, x + oX, y + oY, z + oZ, d)];
  }
}

static void setNodeAaSoA(KernelData *kd, int x, int y, int z, PdfT *pdfs) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  int *gDims = kd->GlobalDims;
  if (kd->Iteration % 2 == 0) {
    for (int d = 0; d < N_D3Q19; ++d)
      kd->PdfsActive[indexSoA(gDims, x + oX - D3Q19_X[d],
                              y + oY - D3Q19_Y[d], z + oZ - D3Q19_Z[d],
                              D3Q19_INV[d])] = pdfs[d];
  } else {
    for (int d = 0; d < N_D3Q19; ++d)
      kd->PdfsActive[indexSoA(gDims, x + oX, y + oY, z + oZ, d)] = pdfs[d];
  }
}

// BcGetPdf / BcSetPdf for push/pull SoA/AoS
static void bcGetPdfPushSoA(KernelData *kd, int x, int y, int z, int dir,
                            PdfT *pdf) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  *pdf = kd->PdfsActive[indexSoA(kd->GlobalDims, x + oX, y + oY, z + oZ,
                                 dir)];
}

static void bcSetPdfPushSoA(KernelData *kd, int x, int y, int z, int dir,
                            PdfT pdf) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  kd->PdfsActive[indexSoA(kd->GlobalDims, x + oX, y + oY, z + oZ, dir)] = pdf;
}

static void bcGetPdfPullSoA(KernelData *kd, int x, int y, int z, int dir,
                            PdfT *pdf) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  *pdf = kd->PdfsActive[indexSoA(kd->GlobalDims, x + oX - D3Q19_X[dir],
                                 y + oY - D3Q19_Y[dir],
                                 z + oZ - D3Q19_Z[dir], dir)];
}

static void bcSetPdfPullSoA(KernelData *kd, int x, int y, int z, int dir,
                            PdfT pdf) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  kd->PdfsActive[indexSoA(kd->GlobalDims, x + oX - D3Q19_X[dir],
                          y + oY - D3Q19_Y[dir], z + oZ - D3Q19_Z[dir],
                          dir)] = pdf;
}

static void bcGetPdfPushAoS(KernelData *kd, int x, int y, int z, int dir,
                            PdfT *pdf) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  *pdf = kd->PdfsActive[indexAoS(kd->GlobalDims, x + oX, y + oY, z + oZ,
                                 dir)];
}

static void bcSetPdfPushAoS(KernelData *kd, int x, int y, int z, int dir,
                            PdfT pdf) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  kd->PdfsActive[indexAoS(kd->GlobalDims, x + oX, y + oY, z + oZ, dir)] = pdf;
}

static void bcGetPdfPullAoS(KernelData *kd, int x, int y, int z, int dir,
                            PdfT *pdf) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  *pdf = kd->PdfsActive[indexAoS(kd->GlobalDims, x + oX - D3Q19_X[dir],
                                 y + oY - D3Q19_Y[dir],
                                 z + oZ - D3Q19_Z[dir], dir)];
}

static void bcSetPdfPullAoS(KernelData *kd, int x, int y, int z, int dir,
                            PdfT pdf) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  kd->PdfsActive[indexAoS(kd->GlobalDims, x + oX - D3Q19_X[dir],
                          y + oY - D3Q19_Y[dir], z + oZ - D3Q19_Z[dir],
                          dir)] = pdf;
}

static void bcGetPdfAaSoA(KernelData *kd, int x, int y, int z, int dir,
                          PdfT *pdf) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  if (kd->Iteration % 2 == 0) {
    *pdf = kd->PdfsActive[indexSoA(kd->GlobalDims, x + oX - D3Q19_X[dir],
                                   y + oY - D3Q19_Y[dir],
                                   z + oZ - D3Q19_Z[dir], D3Q19_INV[dir])];
  } else {
    *pdf =
        kd->PdfsActive[indexSoA(kd->GlobalDims, x + oX, y + oY, z + oZ, dir)];
  }
}

static void bcSetPdfAaSoA(KernelData *kd, int x, int y, int z, int dir,
                          PdfT pdf) {
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  if (kd->Iteration % 2 == 0) {
    kd->PdfsActive[indexSoA(kd->GlobalDims, x + oX - D3Q19_X[dir],
                            y + oY - D3Q19_Y[dir], z + oZ - D3Q19_Z[dir],
                            D3Q19_INV[dir])] = pdf;
  } else {
    kd->PdfsActive[indexSoA(kd->GlobalDims, x + oX, y + oY, z + oZ, dir)] =
        pdf;
  }
}

// -----------------------------------------------------------------------
// Kernel init / deinit
// -----------------------------------------------------------------------

// Index function pointer type for bounce-back computation
typedef int (*IndexFn)(int gDims[3], int x, int y, int z, int d);

static void kernelInitInternal(LatticeDesc *ld, KernelData **kernelData,
                               int propModel, int dataLayout) {
  KernelData *kd = (KernelData *)allocate(64, sizeof(KernelData));
  memset(kd, 0, sizeof(KernelData));
  *kernelData = kd;

  kd->Dims[0] = ld->Dims[0];
  kd->Dims[1] = ld->Dims[1];
  kd->Dims[2] = ld->Dims[2];

  int *lDims = ld->Dims;
  int *gDims = kd->GlobalDims;

  gDims[0] = lDims[0] + 2;
  gDims[1] = lDims[1] + 2;
  gDims[2] = lDims[2] + 2;

  kd->Offsets[0] = 1;
  kd->Offsets[1] = 1;
  kd->Offsets[2] = 1;

  kd->Blk[0] = gDims[0];
  kd->Blk[1] = gDims[1];
  kd->Blk[2] = gDims[2];

  kd->Iteration = -1;

  int lX = lDims[0], lY = lDims[1], lZ = lDims[2];
  int gX = gDims[0], gY = gDims[1], gZ = gDims[2];
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];

  int nCells = gX * gY * gZ;

  IndexFn idxFn = (dataLayout == LAYOUT_SOA) ? indexSoA : indexAoS;

  int numPdfArrays = (propModel == PROP_AA) ? 1 : 2;

  printf("# allocating data for %d LB nodes with padding (%lu bytes = %.1f "
         "MiB for %s)\n",
         nCells, (unsigned long)numPdfArrays * sizeof(PdfT) * nCells * N_D3Q19,
         (double)numPdfArrays * sizeof(PdfT) * nCells * N_D3Q19 / 1024.0 /
             1024.0,
         numPdfArrays == 1 ? "single lattice" : "both lattices");

  kd->Pdfs[0] = (PdfT *)allocate(2 * 1024 * 1024,
                                  sizeof(PdfT) * nCells * N_D3Q19);
  if (numPdfArrays == 2)
    kd->Pdfs[1] = (PdfT *)allocate(2 * 1024 * 1024,
                                    sizeof(PdfT) * nCells * N_D3Q19);
  else
    kd->Pdfs[1] = NULL;

  // Initialize PDFs to zero
  for (int x = 0; x < gX; ++x)
    for (int y = 0; y < gY; ++y)
      for (int z = 0; z < gZ; ++z)
        for (int d = 0; d < N_D3Q19; ++d) {
          kd->Pdfs[0][idxFn(gDims, x, y, z, d)] = 0.0;
          if (kd->Pdfs[1])
            kd->Pdfs[1][idxFn(gDims, x, y, z, d)] = 0.0;
        }

  // Count bounce-back PDFs
  int nBounceBackPdfs = 0;
  int nx, ny, nz;

  for (int x = 0; x < lX; ++x)
    for (int y = 0; y < lY; ++y)
      for (int z = 0; z < lZ; ++z) {
        if (ld->Lattice[latticeIndex(lDims, x, y, z)] == LAT_CELL_OBSTACLE)
          continue;
        for (int d = 0; d < N_D3Q19; ++d) {
          if (propModel == PROP_PUSH) {
            nx = x + D3Q19_X[d]; ny = y + D3Q19_Y[d]; nz = z + D3Q19_Z[d];
          } else if (propModel == PROP_PULL) {
            nx = x - D3Q19_X[d]; ny = y - D3Q19_Y[d]; nz = z - D3Q19_Z[d];
          } else { // AA: uses push-like neighbor for counting
            if (propModel == PROP_AA) {
              nx = x + D3Q19_X[d]; ny = y + D3Q19_Y[d]; nz = z + D3Q19_Z[d];
            } else {
              nx = x - D3Q19_X[d]; ny = y - D3Q19_Y[d]; nz = z - D3Q19_Z[d];
            }
          }

          if ((nx < 0 || nx >= lX) && ld->PeriodicX)
            ++nBounceBackPdfs;
          else if ((ny < 0 || ny >= lY) && ld->PeriodicY)
            ++nBounceBackPdfs;
          else if ((nz < 0 || nz >= lZ) && ld->PeriodicZ)
            ++nBounceBackPdfs;
          else if (nx < 0 || ny < 0 || nz < 0 || nx >= lX || ny >= lY ||
                   nz >= lZ)
            continue;
          else if (ld->Lattice[latticeIndex(lDims, nx, ny, nz)] ==
                   LAT_CELL_OBSTACLE)
            ++nBounceBackPdfs;
        }
      }

  kd->BounceBackPdfsSrc =
      (int *)allocate(64, sizeof(int) * (nBounceBackPdfs + 1));
  kd->BounceBackPdfsDst =
      (int *)allocate(64, sizeof(int) * (nBounceBackPdfs + 1));
  kd->nBounceBackPdfs = nBounceBackPdfs;

  printf("# bounce-back PDFs: %d\n", nBounceBackPdfs);

  // Build bounce-back index arrays
  nBounceBackPdfs = 0;
  int srcIndex, dstIndex;
  int px, py, pz;

  for (int x = 0; x < lX; ++x)
    for (int y = 0; y < lY; ++y)
      for (int z = 0; z < lZ; ++z) {
        if (ld->Lattice[latticeIndex(lDims, x, y, z)] == LAT_CELL_OBSTACLE)
          continue;
        for (int d = 0; d < N_D3Q19; ++d) {
          if (propModel == PROP_PUSH) {
            nx = x + D3Q19_X[d]; ny = y + D3Q19_Y[d]; nz = z + D3Q19_Z[d];
          } else if (propModel == PROP_PULL) {
            nx = x - D3Q19_X[d]; ny = y - D3Q19_Y[d]; nz = z - D3Q19_Z[d];
          } else {
            nx = x + D3Q19_X[d]; ny = y + D3Q19_Y[d]; nz = z + D3Q19_Z[d];
          }

          if (((nx < 0 || nx >= lX) && ld->PeriodicX) ||
              ((ny < 0 || ny >= lY) && ld->PeriodicY) ||
              ((nz < 0 || nz >= lZ) && ld->PeriodicZ)) {
            // Periodic boundary
            px = (nx < 0) ? lX - 1 : (nx >= lX) ? 0 : nx;
            py = (ny < 0) ? lY - 1 : (ny >= lY) ? 0 : ny;
            pz = (nz < 0) ? lZ - 1 : (nz >= lZ) ? 0 : nz;

            if (ld->Lattice[latticeIndex(lDims, px, py, pz)] ==
                LAT_CELL_OBSTACLE) {
              if (propModel == PROP_PUSH) {
                srcIndex = idxFn(gDims, nx + oX, ny + oY, nz + oZ, d);
                dstIndex =
                    idxFn(gDims, x + oX, y + oY, z + oZ, D3Q19_INV[d]);
              } else if (propModel == PROP_PULL) {
                srcIndex =
                    idxFn(gDims, x + oX, y + oY, z + oZ, D3Q19_INV[d]);
                dstIndex = idxFn(gDims, nx + oX, ny + oY, nz + oZ, d);
              } else { // AA
                srcIndex = idxFn(gDims, nx + oX, ny + oY, nz + oZ, d);
                dstIndex =
                    idxFn(gDims, x + oX, y + oY, z + oZ, D3Q19_INV[d]);
              }
            } else {
              if (propModel == PROP_PUSH) {
                srcIndex = idxFn(gDims, nx + oX, ny + oY, nz + oZ, d);
                dstIndex = idxFn(gDims, px + oX, py + oY, pz + oZ, d);
              } else if (propModel == PROP_PULL) {
                srcIndex = idxFn(gDims, px + oX, py + oY, pz + oZ, d);
                dstIndex = idxFn(gDims, nx + oX, ny + oY, nz + oZ, d);
              } else { // AA
                srcIndex = idxFn(gDims, nx + oX, ny + oY, nz + oZ, d);
                dstIndex = idxFn(gDims, px + oX, py + oY, pz + oZ, d);
              }
            }

            kd->BounceBackPdfsSrc[nBounceBackPdfs] = srcIndex;
            kd->BounceBackPdfsDst[nBounceBackPdfs] = dstIndex;
            ++nBounceBackPdfs;
          } else if (nx < 0 || ny < 0 || nz < 0 || nx >= lX || ny >= lY ||
                     nz >= lZ) {
            continue;
          } else if (ld->Lattice[latticeIndex(lDims, nx, ny, nz)] ==
                     LAT_CELL_OBSTACLE) {
            if (propModel == PROP_PUSH) {
              srcIndex = idxFn(gDims, nx + oX, ny + oY, nz + oZ, d);
              dstIndex = idxFn(gDims, x + oX, y + oY, z + oZ, D3Q19_INV[d]);
            } else if (propModel == PROP_PULL) {
              srcIndex = idxFn(gDims, x + oX, y + oY, z + oZ, D3Q19_INV[d]);
              dstIndex = idxFn(gDims, nx + oX, ny + oY, nz + oZ, d);
            } else { // AA
              srcIndex = idxFn(gDims, nx + oX, ny + oY, nz + oZ, d);
              dstIndex = idxFn(gDims, x + oX, y + oY, z + oZ, D3Q19_INV[d]);
            }

            kd->BounceBackPdfsSrc[nBounceBackPdfs] = srcIndex;
            kd->BounceBackPdfsDst[nBounceBackPdfs] = dstIndex;
            ++nBounceBackPdfs;
          }
        }
      }

  // Set function pointers based on prop model and data layout
  if (propModel == PROP_PUSH && dataLayout == LAYOUT_SOA) {
    kd->GetNode = getNodePushSoA;
    kd->SetNode = setNodePushSoA;
    kd->BoundaryConditionsGetPdf = bcGetPdfPushSoA;
    kd->BoundaryConditionsSetPdf = bcSetPdfPushSoA;
  } else if (propModel == PROP_PUSH && dataLayout == LAYOUT_AOS) {
    kd->GetNode = getNodePushAoS;
    kd->SetNode = setNodePushAoS;
    kd->BoundaryConditionsGetPdf = bcGetPdfPushAoS;
    kd->BoundaryConditionsSetPdf = bcSetPdfPushAoS;
  } else if (propModel == PROP_PULL && dataLayout == LAYOUT_SOA) {
    kd->GetNode = getNodePullSoA;
    kd->SetNode = setNodePullSoA;
    kd->BoundaryConditionsGetPdf = bcGetPdfPullSoA;
    kd->BoundaryConditionsSetPdf = bcSetPdfPullSoA;
  } else if (propModel == PROP_PULL && dataLayout == LAYOUT_AOS) {
    kd->GetNode = getNodePullAoS;
    kd->SetNode = setNodePullAoS;
    kd->BoundaryConditionsGetPdf = bcGetPdfPullAoS;
    kd->BoundaryConditionsSetPdf = bcSetPdfPullAoS;
  } else if (propModel == PROP_AA && dataLayout == LAYOUT_SOA) {
    kd->GetNode = getNodeAaSoA;
    kd->SetNode = setNodeAaSoA;
    kd->BoundaryConditionsGetPdf = bcGetPdfAaSoA;
    kd->BoundaryConditionsSetPdf = bcSetPdfAaSoA;
  }

  kd->PdfsActive = kd->Pdfs[0];
}

void kernelInitPushSoA(LatticeDesc *ld, KernelData **kd, CaseData *cd) {
  kernelInitInternal(ld, kd, PROP_PUSH, LAYOUT_SOA);
  (*kd)->Kernel = kernelPushSoA;
  (*kd)->LoopBalance = 2.0 * N_D3Q19 * sizeof(PdfT);
}

void kernelInitPushAoS(LatticeDesc *ld, KernelData **kd, CaseData *cd) {
  kernelInitInternal(ld, kd, PROP_PUSH, LAYOUT_AOS);
  (*kd)->Kernel = kernelPushAoS;
  (*kd)->LoopBalance = 2.0 * N_D3Q19 * sizeof(PdfT);
}

void kernelInitPullSoA(LatticeDesc *ld, KernelData **kd, CaseData *cd) {
  kernelInitInternal(ld, kd, PROP_PULL, LAYOUT_SOA);
  (*kd)->Kernel = kernelPullSoA;
  (*kd)->LoopBalance = 2.0 * N_D3Q19 * sizeof(PdfT);
}

void kernelInitPullAoS(LatticeDesc *ld, KernelData **kd, CaseData *cd) {
  kernelInitInternal(ld, kd, PROP_PULL, LAYOUT_AOS);
  (*kd)->Kernel = kernelPullAoS;
  (*kd)->LoopBalance = 2.0 * N_D3Q19 * sizeof(PdfT);
}

void kernelInitBlkPushSoA(LatticeDesc *ld, KernelData **kd, CaseData *cd) {
  kernelInitInternal(ld, kd, PROP_PUSH, LAYOUT_SOA);
  (*kd)->Kernel = kernelBlkPushSoA;
  (*kd)->Blk[0] = 8;
  (*kd)->Blk[1] = 8;
  (*kd)->Blk[2] = 8;
  (*kd)->LoopBalance = 2.0 * N_D3Q19 * sizeof(PdfT);
  printf("# blocking x: %3d y: %3d z: %3d\n", (*kd)->Blk[0], (*kd)->Blk[1],
         (*kd)->Blk[2]);
}

void kernelInitBlkPullSoA(LatticeDesc *ld, KernelData **kd, CaseData *cd) {
  kernelInitInternal(ld, kd, PROP_PULL, LAYOUT_SOA);
  (*kd)->Kernel = kernelBlkPullSoA;
  (*kd)->Blk[0] = 8;
  (*kd)->Blk[1] = 8;
  (*kd)->Blk[2] = 8;
  (*kd)->LoopBalance = 2.0 * N_D3Q19 * sizeof(PdfT);
  printf("# blocking x: %3d y: %3d z: %3d\n", (*kd)->Blk[0], (*kd)->Blk[1],
         (*kd)->Blk[2]);
}

void kernelInitAaSoA(LatticeDesc *ld, KernelData **kd, CaseData *cd) {
  kernelInitInternal(ld, kd, PROP_AA, LAYOUT_SOA);
  (*kd)->Kernel = kernelAaSoA;
  (*kd)->LoopBalance = N_D3Q19 * sizeof(PdfT);
}

void kernelDeinit(LatticeDesc *ld, KernelData **kd) {
  if (*kd == NULL) return;
  free((*kd)->Pdfs[0]);
  free((*kd)->Pdfs[1]);
  free((*kd)->BounceBackPdfsSrc);
  free((*kd)->BounceBackPdfsDst);
  free(*kd);
  *kd = NULL;
}

// -----------------------------------------------------------------------
// Boundary conditions (Zou-He)
// -----------------------------------------------------------------------

void kernelComputeBoundaryConditions(KernelData *kd, LatticeDesc *ld,
                                     CaseData *cd) {
  PdfT rhoIn = cd->RhoIn;
  PdfT rhoOut = cd->RhoOut;
  PdfT rhoInInv = F(1.0) / rhoIn;
  PdfT rhoOutInv = F(1.0) / rhoOut;

  const PdfT oneThird = F(1.0) / F(3.0);
  const PdfT oneFourth = F(1.0) / F(4.0);
  const PdfT oneSixth = F(1.0) / F(6.0);

  PdfT pdfs[N_D3Q19];
  PdfT dens, ux, indepUx;

  int nX = kd->Dims[0], nY = kd->Dims[1], nZ = kd->Dims[2];
  int xIn = 0, xOut = nX - 1;

  for (int z = 1; z < nZ - 1; ++z) {
    for (int y = 1; y < nY - 1; ++y) {
      // Inlet
      if (ld->Lattice[latticeIndex(ld->Dims, xIn, y, z)] == LAT_CELL_INLET) {
        kd->BoundaryConditionsGetPdf(kd, xIn, y, z, D3Q19_C, pdfs + D3Q19_C);
        kd->BoundaryConditionsGetPdf(kd, xIn, y, z, D3Q19_T, pdfs + D3Q19_T);
        kd->BoundaryConditionsGetPdf(kd, xIn, y, z, D3Q19_B, pdfs + D3Q19_B);
        kd->BoundaryConditionsGetPdf(kd, xIn, y, z, D3Q19_S, pdfs + D3Q19_S);
        kd->BoundaryConditionsGetPdf(kd, xIn, y, z, D3Q19_N, pdfs + D3Q19_N);
        kd->BoundaryConditionsGetPdf(kd, xIn, y, z, D3Q19_TS,
                                     pdfs + D3Q19_TS);
        kd->BoundaryConditionsGetPdf(kd, xIn, y, z, D3Q19_BS,
                                     pdfs + D3Q19_BS);
        kd->BoundaryConditionsGetPdf(kd, xIn, y, z, D3Q19_TN,
                                     pdfs + D3Q19_TN);
        kd->BoundaryConditionsGetPdf(kd, xIn, y, z, D3Q19_BN,
                                     pdfs + D3Q19_BN);
        kd->BoundaryConditionsGetPdf(kd, xIn, y, z, D3Q19_SW,
                                     pdfs + D3Q19_SW);
        kd->BoundaryConditionsGetPdf(kd, xIn, y, z, D3Q19_TW,
                                     pdfs + D3Q19_TW);
        kd->BoundaryConditionsGetPdf(kd, xIn, y, z, D3Q19_W, pdfs + D3Q19_W);
        kd->BoundaryConditionsGetPdf(kd, xIn, y, z, D3Q19_BW,
                                     pdfs + D3Q19_BW);
        kd->BoundaryConditionsGetPdf(kd, xIn, y, z, D3Q19_NW,
                                     pdfs + D3Q19_NW);

        dens = rhoIn;
        ux = F(1.0) -
             (pdfs[D3Q19_C] +
              (pdfs[D3Q19_T] + pdfs[D3Q19_B] + pdfs[D3Q19_S] +
               pdfs[D3Q19_N]) +
              (pdfs[D3Q19_TS] + pdfs[D3Q19_BS] + pdfs[D3Q19_TN] +
               pdfs[D3Q19_BN]) +
              F(2.0) * (pdfs[D3Q19_SW] + pdfs[D3Q19_TW] + pdfs[D3Q19_W] +
                        pdfs[D3Q19_BW] + pdfs[D3Q19_NW])) *
                 rhoInInv;

        indepUx = oneSixth * dens * ux;
        pdfs[D3Q19_E] = pdfs[D3Q19_W] + oneThird * dens * ux;
        pdfs[D3Q19_NE] =
            pdfs[D3Q19_SW] -
            oneFourth * (pdfs[D3Q19_N] - pdfs[D3Q19_S]) + indepUx;
        pdfs[D3Q19_SE] =
            pdfs[D3Q19_NW] +
            oneFourth * (pdfs[D3Q19_N] - pdfs[D3Q19_S]) + indepUx;
        pdfs[D3Q19_TE] =
            pdfs[D3Q19_BW] -
            oneFourth * (pdfs[D3Q19_T] - pdfs[D3Q19_B]) + indepUx;
        pdfs[D3Q19_BE] =
            pdfs[D3Q19_TW] +
            oneFourth * (pdfs[D3Q19_T] - pdfs[D3Q19_B]) + indepUx;

        kd->BoundaryConditionsSetPdf(kd, xIn, y, z, D3Q19_E, pdfs[D3Q19_E]);
        kd->BoundaryConditionsSetPdf(kd, xIn, y, z, D3Q19_NE, pdfs[D3Q19_NE]);
        kd->BoundaryConditionsSetPdf(kd, xIn, y, z, D3Q19_SE, pdfs[D3Q19_SE]);
        kd->BoundaryConditionsSetPdf(kd, xIn, y, z, D3Q19_TE, pdfs[D3Q19_TE]);
        kd->BoundaryConditionsSetPdf(kd, xIn, y, z, D3Q19_BE, pdfs[D3Q19_BE]);
      }

      // Outlet
      if (ld->Lattice[latticeIndex(ld->Dims, xOut, y, z)] ==
          LAT_CELL_OUTLET) {
        kd->BoundaryConditionsGetPdf(kd, xOut, y, z, D3Q19_C, pdfs + D3Q19_C);
        kd->BoundaryConditionsGetPdf(kd, xOut, y, z, D3Q19_T, pdfs + D3Q19_T);
        kd->BoundaryConditionsGetPdf(kd, xOut, y, z, D3Q19_B, pdfs + D3Q19_B);
        kd->BoundaryConditionsGetPdf(kd, xOut, y, z, D3Q19_S, pdfs + D3Q19_S);
        kd->BoundaryConditionsGetPdf(kd, xOut, y, z, D3Q19_N, pdfs + D3Q19_N);
        kd->BoundaryConditionsGetPdf(kd, xOut, y, z, D3Q19_TS,
                                     pdfs + D3Q19_TS);
        kd->BoundaryConditionsGetPdf(kd, xOut, y, z, D3Q19_BS,
                                     pdfs + D3Q19_BS);
        kd->BoundaryConditionsGetPdf(kd, xOut, y, z, D3Q19_TN,
                                     pdfs + D3Q19_TN);
        kd->BoundaryConditionsGetPdf(kd, xOut, y, z, D3Q19_BN,
                                     pdfs + D3Q19_BN);
        kd->BoundaryConditionsGetPdf(kd, xOut, y, z, D3Q19_NE,
                                     pdfs + D3Q19_NE);
        kd->BoundaryConditionsGetPdf(kd, xOut, y, z, D3Q19_BE,
                                     pdfs + D3Q19_BE);
        kd->BoundaryConditionsGetPdf(kd, xOut, y, z, D3Q19_E, pdfs + D3Q19_E);
        kd->BoundaryConditionsGetPdf(kd, xOut, y, z, D3Q19_TE,
                                     pdfs + D3Q19_TE);
        kd->BoundaryConditionsGetPdf(kd, xOut, y, z, D3Q19_SE,
                                     pdfs + D3Q19_SE);

        dens = rhoOut;
        ux = F(-1.0) +
             (pdfs[D3Q19_C] +
              (pdfs[D3Q19_T] + pdfs[D3Q19_B] + pdfs[D3Q19_S] +
               pdfs[D3Q19_N]) +
              (pdfs[D3Q19_TS] + pdfs[D3Q19_BS] + pdfs[D3Q19_TN] +
               pdfs[D3Q19_BN]) +
              F(2.0) * (pdfs[D3Q19_NE] + pdfs[D3Q19_BE] + pdfs[D3Q19_E] +
                        pdfs[D3Q19_TE] + pdfs[D3Q19_SE])) *
                 rhoOutInv;

        indepUx = oneSixth * dens * ux;
        pdfs[D3Q19_W] = pdfs[D3Q19_E] - oneThird * dens * ux;
        pdfs[D3Q19_SW] =
            pdfs[D3Q19_NE] +
            oneFourth * (pdfs[D3Q19_N] - pdfs[D3Q19_S]) - indepUx;
        pdfs[D3Q19_NW] =
            pdfs[D3Q19_SE] -
            oneFourth * (pdfs[D3Q19_N] - pdfs[D3Q19_S]) - indepUx;
        pdfs[D3Q19_BW] =
            pdfs[D3Q19_TE] +
            oneFourth * (pdfs[D3Q19_T] - pdfs[D3Q19_B]) - indepUx;
        pdfs[D3Q19_TW] =
            pdfs[D3Q19_BE] -
            oneFourth * (pdfs[D3Q19_T] - pdfs[D3Q19_B]) - indepUx;

        kd->BoundaryConditionsSetPdf(kd, xOut, y, z, D3Q19_W, pdfs[D3Q19_W]);
        kd->BoundaryConditionsSetPdf(kd, xOut, y, z, D3Q19_NW,
                                     pdfs[D3Q19_NW]);
        kd->BoundaryConditionsSetPdf(kd, xOut, y, z, D3Q19_SW,
                                     pdfs[D3Q19_SW]);
        kd->BoundaryConditionsSetPdf(kd, xOut, y, z, D3Q19_TW,
                                     pdfs[D3Q19_TW]);
        kd->BoundaryConditionsSetPdf(kd, xOut, y, z, D3Q19_BW,
                                     pdfs[D3Q19_BW]);
      }
    }
  }
}

// -----------------------------------------------------------------------
// Initial density / velocity
// -----------------------------------------------------------------------

void kernelSetInitialDensity(LatticeDesc *ld, KernelData *kd, CaseData *cd) {
  int *lDims = ld->Dims;
  PdfT rhoIn = cd->RhoIn, rhoOut = cd->RhoOut;
  PdfT ux = F(0.0), uy = F(0.0), uz = F(0.0), dens;
  PdfT omega = cd->Omega;

  PdfT w_0 = F(1.0) / F(3.0), w_1 = F(1.0) / F(18.0), w_2 = F(1.0) / F(36.0);
  PdfT omegaW0 = F(3.0) * w_0 * omega;
  PdfT omegaW1 = F(3.0) * w_1 * omega;
  PdfT omegaW2 = F(3.0) * w_2 * omega;
  PdfT oneThird = F(1.0) / F(3.0);
  PdfT dirIndepTrm;

  int nX = lDims[0], nY = lDims[1], nZ = lDims[2];
  PdfT pdfs[N_D3Q19];

#define SQR(a) ((a) * (a))
  for (int z = 0; z < nZ; ++z)
    for (int y = 0; y < nY; ++y)
      for (int x = 0; x < nX; ++x) {
        if (ld->Lattice[latticeIndex(lDims, x, y, z)] == LAT_CELL_OBSTACLE)
          continue;
        dens = rhoIn + (rhoOut - rhoIn) * x / (nX - F(1.0));
        dirIndepTrm =
            oneThird * dens - F(0.5) * (ux * ux + uy * uy + uz * uz);

        pdfs[D3Q19_C] = omegaW0 * dirIndepTrm;
        pdfs[D3Q19_N] = omegaW1 * (dirIndepTrm + uy + F(1.5) * SQR(uy));
        pdfs[D3Q19_S] = omegaW1 * (dirIndepTrm - uy + F(1.5) * SQR(uy));
        pdfs[D3Q19_E] = omegaW1 * (dirIndepTrm + ux + F(1.5) * SQR(ux));
        pdfs[D3Q19_W] = omegaW1 * (dirIndepTrm - ux + F(1.5) * SQR(ux));
        pdfs[D3Q19_T] = omegaW1 * (dirIndepTrm + uz + F(1.5) * SQR(uz));
        pdfs[D3Q19_B] = omegaW1 * (dirIndepTrm - uz + F(1.5) * SQR(uz));

        pdfs[D3Q19_NE] =
            omegaW2 * (dirIndepTrm + (ux + uy) + F(1.5) * SQR(ux + uy));
        pdfs[D3Q19_SW] =
            omegaW2 * (dirIndepTrm - (ux + uy) + F(1.5) * SQR(ux + uy));
        pdfs[D3Q19_SE] =
            omegaW2 * (dirIndepTrm + (ux - uy) + F(1.5) * SQR(ux - uy));
        pdfs[D3Q19_NW] =
            omegaW2 * (dirIndepTrm - (ux - uy) + F(1.5) * SQR(ux - uy));

        pdfs[D3Q19_TE] =
            omegaW2 * (dirIndepTrm + (ux + uz) + F(1.5) * SQR(ux + uz));
        pdfs[D3Q19_BW] =
            omegaW2 * (dirIndepTrm - (ux + uz) + F(1.5) * SQR(ux + uz));
        pdfs[D3Q19_BE] =
            omegaW2 * (dirIndepTrm + (ux - uz) + F(1.5) * SQR(ux - uz));
        pdfs[D3Q19_TW] =
            omegaW2 * (dirIndepTrm - (ux - uz) + F(1.5) * SQR(ux - uz));

        pdfs[D3Q19_TN] =
            omegaW2 * (dirIndepTrm + (uy + uz) + F(1.5) * SQR(uy + uz));
        pdfs[D3Q19_BS] =
            omegaW2 * (dirIndepTrm - (uy + uz) + F(1.5) * SQR(uy + uz));
        pdfs[D3Q19_BN] =
            omegaW2 * (dirIndepTrm + (uy - uz) + F(1.5) * SQR(uy - uz));
        pdfs[D3Q19_TS] =
            omegaW2 * (dirIndepTrm - (uy - uz) + F(1.5) * SQR(uy - uz));

        kd->SetNode(kd, x, y, z, pdfs);
      }
#undef SQR
}

void kernelSetInitialVelocity(LatticeDesc *ld, KernelData *kd, CaseData *cd) {
  int *lDims = ld->Dims;
  PdfT ux, uy, uz, dens = F(1.0);
  PdfT omega = cd->Omega;

  PdfT w_0 = F(1.0) / F(3.0), w_1 = F(1.0) / F(18.0), w_2 = F(1.0) / F(36.0);
  PdfT omegaW0 = F(3.0) * w_0 * omega;
  PdfT omegaW1 = F(3.0) * w_1 * omega;
  PdfT omegaW2 = F(3.0) * w_2 * omega;
  PdfT oneThird = F(1.0) / F(3.0);
  PdfT dirIndepTrm;

  int nX = lDims[0], nY = lDims[1], nZ = lDims[2];
  PdfT pdfs[N_D3Q19];

#define SQR(a) ((a) * (a))
  for (int z = 0; z < nZ; ++z)
    for (int y = 0; y < nY; ++y)
      for (int x = 0; x < nX; ++x) {
        if (ld->Lattice[latticeIndex(lDims, x, y, z)] != LAT_CELL_FLUID)
          continue;
        ux = uy = uz = F(0.0);
        dirIndepTrm =
            oneThird * dens - F(0.5) * (ux * ux + uy * uy + uz * uz);

        pdfs[D3Q19_C] = omegaW0 * dirIndepTrm;
        pdfs[D3Q19_N] = omegaW1 * (dirIndepTrm + uy + F(1.5) * SQR(uy));
        pdfs[D3Q19_S] = omegaW1 * (dirIndepTrm - uy + F(1.5) * SQR(uy));
        pdfs[D3Q19_E] = omegaW1 * (dirIndepTrm + ux + F(1.5) * SQR(ux));
        pdfs[D3Q19_W] = omegaW1 * (dirIndepTrm - ux + F(1.5) * SQR(ux));
        pdfs[D3Q19_T] = omegaW1 * (dirIndepTrm + uz + F(1.5) * SQR(uz));
        pdfs[D3Q19_B] = omegaW1 * (dirIndepTrm - uz + F(1.5) * SQR(uz));

        pdfs[D3Q19_NE] =
            omegaW2 * (dirIndepTrm + (ux + uy) + F(1.5) * SQR(ux + uy));
        pdfs[D3Q19_SW] =
            omegaW2 * (dirIndepTrm - (ux + uy) + F(1.5) * SQR(ux + uy));
        pdfs[D3Q19_SE] =
            omegaW2 * (dirIndepTrm + (ux - uy) + F(1.5) * SQR(ux - uy));
        pdfs[D3Q19_NW] =
            omegaW2 * (dirIndepTrm - (ux - uy) + F(1.5) * SQR(ux - uy));

        pdfs[D3Q19_TE] =
            omegaW2 * (dirIndepTrm + (ux + uz) + F(1.5) * SQR(ux + uz));
        pdfs[D3Q19_BW] =
            omegaW2 * (dirIndepTrm - (ux + uz) + F(1.5) * SQR(ux + uz));
        pdfs[D3Q19_BE] =
            omegaW2 * (dirIndepTrm + (ux - uz) + F(1.5) * SQR(ux - uz));
        pdfs[D3Q19_TW] =
            omegaW2 * (dirIndepTrm - (ux - uz) + F(1.5) * SQR(ux - uz));

        pdfs[D3Q19_TN] =
            omegaW2 * (dirIndepTrm + (uy + uz) + F(1.5) * SQR(uy + uz));
        pdfs[D3Q19_BS] =
            omegaW2 * (dirIndepTrm - (uy + uz) + F(1.5) * SQR(uy + uz));
        pdfs[D3Q19_BN] =
            omegaW2 * (dirIndepTrm + (uy - uz) + F(1.5) * SQR(uy - uz));
        pdfs[D3Q19_TS] =
            omegaW2 * (dirIndepTrm - (uy - uz) + F(1.5) * SQR(uy - uz));

        kd->SetNode(kd, x, y, z, pdfs);
      }
#undef SQR
}

// -----------------------------------------------------------------------
// Body force (Luo method)
// -----------------------------------------------------------------------

void kernelAddBodyForce(KernelData *kd, LatticeDesc *ld, CaseData *cd) {
  PdfT w_0 = F(1.0) / F(3.0), w_1 = F(1.0) / F(18.0), w_2 = F(1.0) / F(36.0);
  PdfT w[] = {w_1, w_1, w_1, w_1, w_2, w_2, w_2, w_2, w_1, w_2,
              w_2, w_2, w_2, w_1, w_2, w_2, w_2, w_2, w_0};
  PdfT xForce = cd->XForce;
  PdfT pdfs[N_D3Q19];
  int nX = kd->Dims[0], nY = kd->Dims[1], nZ = kd->Dims[2];

  for (int z = 0; z < nZ; ++z)
    for (int y = 0; y < nY; ++y)
      for (int x = 0; x < nX; ++x) {
        if (ld->Lattice[latticeIndex(ld->Dims, x, y, z)] == LAT_CELL_OBSTACLE)
          continue;
        kd->GetNode(kd, x, y, z, pdfs);
        for (int d = 0; d < N_D3Q19; ++d)
          pdfs[d] += F(3.0) * w[d] * D3Q19_X[d] * xForce;
        kd->SetNode(kd, x, y, z, pdfs);
      }
}

// -----------------------------------------------------------------------
// Density
// -----------------------------------------------------------------------

PdfT kernelDensity(KernelData *kd, LatticeDesc *ld) {
  int *lDims = ld->Dims;
  PdfT pdfs[N_D3Q19];
  PdfT density = F(0.0);

  for (int z = 0; z < lDims[2]; ++z)
    for (int y = 0; y < lDims[1]; ++y)
      for (int x = 0; x < lDims[0]; ++x) {
        if (ld->Lattice[latticeIndex(lDims, x, y, z)] == LAT_CELL_OBSTACLE)
          continue;
        kd->GetNode(kd, x, y, z, pdfs);
        PdfT localDensity = F(0.0);
        for (int d = 0; d < N_D3Q19; ++d)
          localDensity += pdfs[d];
        density += localDensity;
      }

  return density / ld->nFluid;
}

// -----------------------------------------------------------------------
// Statistics
// -----------------------------------------------------------------------

void kernelStatistics(KernelData *kd, LatticeDesc *ld, CaseData *cd,
                      int iter) {
  if (iter % cd->StatisticsModulus == 0 || iter == cd->MaxIterations) {
    printf("# iter: %4d   avg density: %e\n", iter, kernelDensity(kd, ld));
  }
}

// -----------------------------------------------------------------------
// Verification
// -----------------------------------------------------------------------

void kernelVerify(LatticeDesc *ld, KernelData *kd, CaseData *cd,
                  PdfT *errorNorm) {
  int nX = ld->Dims[0], nY = ld->Dims[1], nZ = ld->Dims[2];
  PdfT omega = cd->Omega;
  PdfT viscosity = (F(1.0) / omega - F(0.5)) / F(3.0);

  PdfT *outputArray = (PdfT *)malloc(nZ * nY * sizeof(PdfT));
  memset(outputArray, 0, nZ * nY * sizeof(PdfT));

  PdfT pdfs[N_D3Q19];
  int xPos = nX / 2;

  for (int z = 0; z < nZ; ++z)
    for (int y = 0; y < nY; ++y) {
      if (ld->Lattice[latticeIndex(ld->Dims, xPos, y, z)] !=
          LAT_CELL_OBSTACLE) {
        kd->GetNode(kd, xPos, y, z, pdfs);
        PdfT ux = pdfs[D3Q19_E] + pdfs[D3Q19_NE] + pdfs[D3Q19_SE] +
                  pdfs[D3Q19_TE] + pdfs[D3Q19_BE] - pdfs[D3Q19_W] -
                  pdfs[D3Q19_NW] - pdfs[D3Q19_SW] - pdfs[D3Q19_TW] -
                  pdfs[D3Q19_BW];
#ifdef VERIFICATION
        ux += F(0.5) * cd->XForce;
#endif
        outputArray[y * nZ + z] = ux;
      }
    }

#define SQR(a) ((a) * (a))
  PdfT center = nY / F(2.0);
  PdfT minDiameter = (PdfT)nY;
  PdfT minRadiusSquared = SQR(minDiameter / F(2.0) - F(1.0));

  PdfT deviation = F(0.0);
  int flagEvenNy = (nY % 2 == 0) ? 1 : 0;
  int y = (nY - flagEvenNy - 1) / 2;

  for (int z = 0; z < nZ; ++z) {
    PdfT curRadSq = SQR(z - center + F(0.5));
    PdfT analyUx;
    if (curRadSq >= minRadiusSquared)
      analyUx = F(0.0);
    else
      analyUx = cd->XForce * (minRadiusSquared - curRadSq) / (F(2.0) * viscosity);

    PdfT avgUx;
    if (flagEvenNy)
      avgUx = (outputArray[y * nZ + z] + outputArray[(y + 1) * nZ + z]) / F(2.0);
    else
      avgUx = outputArray[y * nZ + z];

    if (analyUx != F(0.0))
      deviation += SQR(fabs(analyUx - avgUx) / analyUx);
  }
#undef SQR

  *errorNorm = (PdfT)sqrt(deviation);
  printf("# Kernel validation: L2 error norm of relative error: %e\n",
         *errorNorm);

  free(outputArray);
}
