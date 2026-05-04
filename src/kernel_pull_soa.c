/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of LBM-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include "lbm.h"
#include "timing.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void kernelPullSoA(LatticeDescType *ld, KernelDataType *kd, CaseDataType *cd)
{
  int nX = ld->Dims[0], nY = ld->Dims[1], nZ = ld->Dims[2];
  int *gDims = kd->GlobalDims;
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];

  PdfType omega      = cd->Omega;
  PdfType omegaEven  = omega;
  PdfType magicParam = F(1.0) / F(12.0);
  PdfType omegaOdd   = F(1.0) / (F(0.5) + magicParam / (F(1.0) / omega - F(0.5)));

  PdfType evenPart, oddPart, dirIndepTrm;
  PdfType w0   = F(1.0) / F(3.0);
  PdfType w1   = F(1.0) / F(18.0);
  PdfType w2   = F(1.0) / F(36.0);
  PdfType w1X3 = w1 * F(3.0), w1NineHalf = w1 * F(9.0) / F(2.0);
  PdfType w2X3 = w2 * F(3.0), w2NineHalf = w2 * F(9.0) / F(2.0);
  PdfType w1Indep, w2Indep;
  PdfType ux, uy, uz, ui, dens;
  PdfType pdfN, pdfS, pdfE, pdfW, pdfNE, pdfSE, pdfNW, pdfSW;
  PdfType pdfT, pdfTN, pdfTE, pdfTW, pdfTS;
  PdfType pdfB, pdfBN, pdfBE, pdfBW, pdfBS, pdfC;

  PdfType *src = kd->Pdfs[0];
  PdfType *dst = kd->Pdfs[1];
  PdfType *tmp;
  int maxIterations = cd->MaxIterations;

#define I(x, y, z, dir) indexSoA(gDims, (x), (y), (z), (dir))

  kd->Duration = -getTimeStamp();

  for (int iter = 0; iter < maxIterations; ++iter) {

#ifdef _OPENMP
#pragma omp parallel for collapse(2) default(none) shared(gDims,                         \
        src,                                                                             \
        dst,                                                                             \
        w0,                                                                              \
        w1,                                                                              \
        w2,                                                                              \
        omegaEven,                                                                       \
        omegaOdd,                                                                        \
        w1X3,                                                                            \
        w2X3,                                                                            \
        w1NineHalf,                                                                      \
        w2NineHalf,                                                                      \
        oX,                                                                              \
        oY,                                                                              \
        oZ,                                                                              \
        nX,                                                                              \
        nY,                                                                              \
        nZ) private(ux,                                                                  \
        uy,                                                                              \
        uz,                                                                              \
        ui,                                                                              \
        dens,                                                                            \
        dirIndepTrm,                                                                     \
        pdfC,                                                                            \
        pdfN,                                                                            \
        pdfE,                                                                            \
        pdfS,                                                                            \
        pdfW,                                                                            \
        pdfNE,                                                                           \
        pdfSE,                                                                           \
        pdfSW,                                                                           \
        pdfNW,                                                                           \
        pdfT,                                                                            \
        pdfTN,                                                                           \
        pdfTE,                                                                           \
        pdfTS,                                                                           \
        pdfTW,                                                                           \
        pdfB,                                                                            \
        pdfBN,                                                                           \
        pdfBE,                                                                           \
        pdfBS,                                                                           \
        pdfBW,                                                                           \
        evenPart,                                                                        \
        oddPart,                                                                         \
        w1Indep,                                                                         \
        w2Indep)
#endif
    for (int x = oX; x < nX + oX; ++x) {
      for (int y = oY; y < nY + oY; ++y) {
        for (int z = oZ; z < nZ + oZ; ++z) {
          // Pull: read from neighbors
          pdfN  = src[I(x, y - 1, z, D3Q19_N)];
          pdfS  = src[I(x, y + 1, z, D3Q19_S)];
          pdfE  = src[I(x - 1, y, z, D3Q19_E)];
          pdfW  = src[I(x + 1, y, z, D3Q19_W)];
          pdfNE = src[I(x - 1, y - 1, z, D3Q19_NE)];
          pdfSE = src[I(x - 1, y + 1, z, D3Q19_SE)];
          pdfNW = src[I(x + 1, y - 1, z, D3Q19_NW)];
          pdfSW = src[I(x + 1, y + 1, z, D3Q19_SW)];
          pdfT  = src[I(x, y, z - 1, D3Q19_T)];
          pdfTN = src[I(x, y - 1, z - 1, D3Q19_TN)];
          pdfTE = src[I(x - 1, y, z - 1, D3Q19_TE)];
          pdfTW = src[I(x + 1, y, z - 1, D3Q19_TW)];
          pdfTS = src[I(x, y + 1, z - 1, D3Q19_TS)];
          pdfB  = src[I(x, y, z + 1, D3Q19_B)];
          pdfBS = src[I(x, y + 1, z + 1, D3Q19_BS)];
          pdfBN = src[I(x, y - 1, z + 1, D3Q19_BN)];
          pdfBW = src[I(x + 1, y, z + 1, D3Q19_BW)];
          pdfBE = src[I(x - 1, y, z + 1, D3Q19_BE)];
          pdfC  = src[I(x, y, z, D3Q19_C)];

          ux =
              pdfE + pdfNE + pdfSE + pdfTE + pdfBE - pdfW - pdfNW - pdfSW - pdfTW - pdfBW;
          uy =
              pdfN + pdfNE + pdfNW + pdfTN + pdfBN - pdfS - pdfSE - pdfSW - pdfTS - pdfBS;
          uz =
              pdfT + pdfTE + pdfTW + pdfTN + pdfTS - pdfB - pdfBE - pdfBW - pdfBN - pdfBS;

          dens = pdfC + pdfN + pdfE + pdfS + pdfW + pdfNE + pdfSE + pdfSW + pdfNW + pdfT +
                 pdfTN + pdfTE + pdfTS + pdfTW + pdfB + pdfBN + pdfBE + pdfBS + pdfBW;

          dirIndepTrm = dens - (ux * ux + uy * uy + uz * uz) * F(3.0) / F(2.0);

          // Pull: write to local cell
          dst[I(x, y, z, D3Q19_C)] = pdfC - omegaEven * (pdfC - w0 * dirIndepTrm);

          w1Indep                  = w1 * dirIndepTrm;

          ui                       = uy;
          evenPart =
              omegaEven * (F(0.5) * (pdfN + pdfS) - ui * ui * w1NineHalf - w1Indep);
          oddPart                  = omegaOdd * (F(0.5) * (pdfN - pdfS) - ui * w1X3);
          dst[I(x, y, z, D3Q19_N)] = pdfN - evenPart - oddPart;
          dst[I(x, y, z, D3Q19_S)] = pdfS - evenPart + oddPart;

          ui                       = ux;
          evenPart =
              omegaEven * (F(0.5) * (pdfE + pdfW) - ui * ui * w1NineHalf - w1Indep);
          oddPart                  = omegaOdd * (F(0.5) * (pdfE - pdfW) - ui * w1X3);
          dst[I(x, y, z, D3Q19_E)] = pdfE - evenPart - oddPart;
          dst[I(x, y, z, D3Q19_W)] = pdfW - evenPart + oddPart;

          ui                       = uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfT + pdfB) - ui * ui * w1NineHalf - w1Indep);
          oddPart                  = omegaOdd * (F(0.5) * (pdfT - pdfB) - ui * w1X3);
          dst[I(x, y, z, D3Q19_T)] = pdfT - evenPart - oddPart;
          dst[I(x, y, z, D3Q19_B)] = pdfB - evenPart + oddPart;

          w2Indep                  = w2 * dirIndepTrm;

          ui                       = -ux + uy;
          evenPart =
              omegaEven * (F(0.5) * (pdfNW + pdfSE) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfNW - pdfSE) - ui * w2X3);
          dst[I(x, y, z, D3Q19_NW)] = pdfNW - evenPart - oddPart;
          dst[I(x, y, z, D3Q19_SE)] = pdfSE - evenPart + oddPart;

          ui                        = ux + uy;
          evenPart =
              omegaEven * (F(0.5) * (pdfNE + pdfSW) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfNE - pdfSW) - ui * w2X3);
          dst[I(x, y, z, D3Q19_NE)] = pdfNE - evenPart - oddPart;
          dst[I(x, y, z, D3Q19_SW)] = pdfSW - evenPart + oddPart;

          ui                        = -ux + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTW + pdfBE) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfTW - pdfBE) - ui * w2X3);
          dst[I(x, y, z, D3Q19_TW)] = pdfTW - evenPart - oddPart;
          dst[I(x, y, z, D3Q19_BE)] = pdfBE - evenPart + oddPart;

          ui                        = ux + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTE + pdfBW) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfTE - pdfBW) - ui * w2X3);
          dst[I(x, y, z, D3Q19_TE)] = pdfTE - evenPart - oddPart;
          dst[I(x, y, z, D3Q19_BW)] = pdfBW - evenPart + oddPart;

          ui                        = -uy + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTS + pdfBN) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfTS - pdfBN) - ui * w2X3);
          dst[I(x, y, z, D3Q19_TS)] = pdfTS - evenPart - oddPart;
          dst[I(x, y, z, D3Q19_BN)] = pdfBN - evenPart + oddPart;

          ui                        = uy + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTN + pdfBS) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfTN - pdfBS) - ui * w2X3);
          dst[I(x, y, z, D3Q19_TN)] = pdfTN - evenPart - oddPart;
          dst[I(x, y, z, D3Q19_BS)] = pdfBS - evenPart + oddPart;
        }
      }
    }

    for (int i = 0; i < kd->nBounceBackPdfs; ++i)
      dst[kd->BounceBackPdfsDst[i]] = dst[kd->BounceBackPdfsSrc[i]];

#ifdef VERIFICATION
    kd->PdfsActive = dst;
    kernelAddBodyForce(kd, ld, cd);
#endif

    tmp = src;
    src = dst;
    dst = tmp;
  }

  kd->Duration += getTimeStamp();
  kd->PdfsActive = src;

#undef I
}
