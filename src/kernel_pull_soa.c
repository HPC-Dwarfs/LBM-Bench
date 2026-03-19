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
  PdfType pdfN, pdfS, pdfE, pdfW, pdfNe, pdfSe, pdfNw, pdfSw;
  PdfType pdfT, pdfTn, pdfTe, pdfTw, pdfTs;
  PdfType pdfB, pdfBn, pdfBe, pdfBw, pdfBs, pdfC;

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
        w_0,                                                                             \
        w_1,                                                                             \
        w_2,                                                                             \
        omegaEven,                                                                       \
        omegaOdd,                                                                        \
        w_1_x3,                                                                          \
        w_2_x3,                                                                          \
        w_1_nine_half,                                                                   \
        w_2_nine_half,                                                                   \
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
        dir_indep_trm,                                                                   \
        pdf_C,                                                                           \
        pdf_N,                                                                           \
        pdf_E,                                                                           \
        pdf_S,                                                                           \
        pdf_W,                                                                           \
        pdf_NE,                                                                          \
        pdf_SE,                                                                          \
        pdf_SW,                                                                          \
        pdf_NW,                                                                          \
        pdf_T,                                                                           \
        pdf_TN,                                                                          \
        pdf_TE,                                                                          \
        pdf_TS,                                                                          \
        pdf_TW,                                                                          \
        pdf_B,                                                                           \
        pdf_BN,                                                                          \
        pdf_BE,                                                                          \
        pdf_BS,                                                                          \
        pdf_BW,                                                                          \
        evenPart,                                                                        \
        oddPart,                                                                         \
        w_1_indep,                                                                       \
        w_2_indep)
#endif
    for (int x = oX; x < nX + oX; ++x) {
      for (int y = oY; y < nY + oY; ++y) {
        for (int z = oZ; z < nZ + oZ; ++z) {
          // Pull: read from neighbors
          pdfN  = src[I(x, y - 1, z, D3Q19_N)];
          pdfS  = src[I(x, y + 1, z, D3Q19_S)];
          pdfE  = src[I(x - 1, y, z, D3Q19_E)];
          pdfW  = src[I(x + 1, y, z, D3Q19_W)];
          pdfNe = src[I(x - 1, y - 1, z, D3Q19_NE)];
          pdfSe = src[I(x - 1, y + 1, z, D3Q19_SE)];
          pdfNw = src[I(x + 1, y - 1, z, D3Q19_NW)];
          pdfSw = src[I(x + 1, y + 1, z, D3Q19_SW)];
          pdfT  = src[I(x, y, z - 1, D3Q19_T)];
          pdfTn = src[I(x, y - 1, z - 1, D3Q19_TN)];
          pdfTe = src[I(x - 1, y, z - 1, D3Q19_TE)];
          pdfTw = src[I(x + 1, y, z - 1, D3Q19_TW)];
          pdfTs = src[I(x, y + 1, z - 1, D3Q19_TS)];
          pdfB  = src[I(x, y, z + 1, D3Q19_B)];
          pdfBs = src[I(x, y + 1, z + 1, D3Q19_BS)];
          pdfBn = src[I(x, y - 1, z + 1, D3Q19_BN)];
          pdfBw = src[I(x + 1, y, z + 1, D3Q19_BW)];
          pdfBe = src[I(x - 1, y, z + 1, D3Q19_BE)];
          pdfC  = src[I(x, y, z, D3Q19_C)];

          ux =
              pdfE + pdfNe + pdfSe + pdfTe + pdfBe - pdfW - pdfNw - pdfSw - pdfTw - pdfBw;
          uy =
              pdfN + pdfNe + pdfNw + pdfTn + pdfBn - pdfS - pdfSe - pdfSw - pdfTs - pdfBs;
          uz =
              pdfT + pdfTe + pdfTw + pdfTn + pdfTs - pdfB - pdfBe - pdfBw - pdfBn - pdfBs;

          dens = pdfC + pdfN + pdfE + pdfS + pdfW + pdfNe + pdfSe + pdfSw + pdfNw + pdfT +
                 pdfTn + pdfTe + pdfTs + pdfTw + pdfB + pdfBn + pdfBe + pdfBs + pdfBw;

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
              omegaEven * (F(0.5) * (pdfNw + pdfSe) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfNw - pdfSe) - ui * w2X3);
          dst[I(x, y, z, D3Q19_NW)] = pdfNw - evenPart - oddPart;
          dst[I(x, y, z, D3Q19_SE)] = pdfSe - evenPart + oddPart;

          ui                        = ux + uy;
          evenPart =
              omegaEven * (F(0.5) * (pdfNe + pdfSw) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfNe - pdfSw) - ui * w2X3);
          dst[I(x, y, z, D3Q19_NE)] = pdfNe - evenPart - oddPart;
          dst[I(x, y, z, D3Q19_SW)] = pdfSw - evenPart + oddPart;

          ui                        = -ux + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTw + pdfBe) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfTw - pdfBe) - ui * w2X3);
          dst[I(x, y, z, D3Q19_TW)] = pdfTw - evenPart - oddPart;
          dst[I(x, y, z, D3Q19_BE)] = pdfBe - evenPart + oddPart;

          ui                        = ux + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTe + pdfBw) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfTe - pdfBw) - ui * w2X3);
          dst[I(x, y, z, D3Q19_TE)] = pdfTe - evenPart - oddPart;
          dst[I(x, y, z, D3Q19_BW)] = pdfBw - evenPart + oddPart;

          ui                        = -uy + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTs + pdfBn) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfTs - pdfBn) - ui * w2X3);
          dst[I(x, y, z, D3Q19_TS)] = pdfTs - evenPart - oddPart;
          dst[I(x, y, z, D3Q19_BN)] = pdfBn - evenPart + oddPart;

          ui                        = uy + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTn + pdfBs) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfTn - pdfBs) - ui * w2X3);
          dst[I(x, y, z, D3Q19_TN)] = pdfTn - evenPart - oddPart;
          dst[I(x, y, z, D3Q19_BS)] = pdfBs - evenPart + oddPart;
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
