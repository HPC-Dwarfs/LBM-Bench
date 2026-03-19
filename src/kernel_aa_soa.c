/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of LBM-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include "lbm.h"
#include "timing.h"

void kernelAaSoA(LatticeDescType *ld, KernelDataType *kd, CaseDataType *cd)
{
  int nX = ld->Dims[0], nY = ld->Dims[1], nZ = ld->Dims[2];
  int *gDims = kd->GlobalDims;
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];

  PdfType omega      = cd->Omega;
  PdfType omegaEven  = omega;
  PdfType magicParam = F(1.0) / F(12.0);
  PdfType omegaOdd   = F(1.0) / (F(0.5) + magicParam / (F(1.0) / omega - F(0.5)));

  PdfType evenPart, oddPart, dirIndepTrm;
  PdfType w0 = F(1.0) / F(3.0), w1 = F(1.0) / F(18.0), w2 = F(1.0) / F(36.0);
  PdfType w1X3 = w1 * F(3.0), w1NineHalf = w1 * F(9.0) / F(2.0);
  PdfType w2X3 = w2 * F(3.0), w2NineHalf = w2 * F(9.0) / F(2.0);
  PdfType w1Indep, w2Indep;
  PdfType ux, uy, uz, ui, dens;
  PdfType pdfN, pdfS, pdfE, pdfW, pdfNe, pdfSe, pdfNw, pdfSw;
  PdfType pdfT, pdfTn, pdfTe, pdfTw, pdfTs;
  PdfType pdfB, pdfBn, pdfBe, pdfBw, pdfBs, pdfC;

  PdfType *src      = kd->Pdfs[0];
  int maxIterations = cd->MaxIterations;

#define I(x, y, z, dir) indexSoA(gDims, (x), (y), (z), (dir))

  kd->Duration = -getTimeStamp();

  for (int iter = 0; iter < maxIterations; iter += 2) {

    // ----------------------------------------------------------------
    // Even time step: read local cell, write to local cell with
    // swapped directions (in-place collision)
    // ----------------------------------------------------------------
    for (int x = oX; x < nX + oX; ++x) {
      for (int y = oY; y < nY + oY; ++y) {
        for (int z = oZ; z < nZ + oZ; ++z) {

          if (ld->Lattice[latticeIndex(ld->Dims, x - oX, y - oY, z - oZ)] ==
              LAT_CELL_OBSTACLE)
            continue;

          pdfN  = src[I(x, y, z, D3Q19_N)];
          pdfS  = src[I(x, y, z, D3Q19_S)];
          pdfE  = src[I(x, y, z, D3Q19_E)];
          pdfW  = src[I(x, y, z, D3Q19_W)];
          pdfNe = src[I(x, y, z, D3Q19_NE)];
          pdfSe = src[I(x, y, z, D3Q19_SE)];
          pdfNw = src[I(x, y, z, D3Q19_NW)];
          pdfSw = src[I(x, y, z, D3Q19_SW)];
          pdfT  = src[I(x, y, z, D3Q19_T)];
          pdfTn = src[I(x, y, z, D3Q19_TN)];
          pdfTe = src[I(x, y, z, D3Q19_TE)];
          pdfTw = src[I(x, y, z, D3Q19_TW)];
          pdfTs = src[I(x, y, z, D3Q19_TS)];
          pdfB  = src[I(x, y, z, D3Q19_B)];
          pdfBs = src[I(x, y, z, D3Q19_BS)];
          pdfBn = src[I(x, y, z, D3Q19_BN)];
          pdfBw = src[I(x, y, z, D3Q19_BW)];
          pdfBe = src[I(x, y, z, D3Q19_BE)];
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

          // Even: write to local cell with swapped (inverse) directions
          src[I(x, y, z, D3Q19_C)] = pdfC - omegaEven * (pdfC - w0 * dirIndepTrm);
          w1Indep                  = w1 * dirIndepTrm;

          ui                       = uy;
          evenPart =
              omegaEven * (F(0.5) * (pdfN + pdfS) - ui * ui * w1NineHalf - w1Indep);
          oddPart                  = omegaOdd * (F(0.5) * (pdfN - pdfS) - ui * w1X3);
          src[I(x, y, z, D3Q19_S)] = pdfN - evenPart - oddPart;
          src[I(x, y, z, D3Q19_N)] = pdfS - evenPart + oddPart;

          ui                       = ux;
          evenPart =
              omegaEven * (F(0.5) * (pdfE + pdfW) - ui * ui * w1NineHalf - w1Indep);
          oddPart                  = omegaOdd * (F(0.5) * (pdfE - pdfW) - ui * w1X3);
          src[I(x, y, z, D3Q19_W)] = pdfE - evenPart - oddPart;
          src[I(x, y, z, D3Q19_E)] = pdfW - evenPart + oddPart;

          ui                       = uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfT + pdfB) - ui * ui * w1NineHalf - w1Indep);
          oddPart                  = omegaOdd * (F(0.5) * (pdfT - pdfB) - ui * w1X3);
          src[I(x, y, z, D3Q19_B)] = pdfT - evenPart - oddPart;
          src[I(x, y, z, D3Q19_T)] = pdfB - evenPart + oddPart;

          w2Indep                  = w2 * dirIndepTrm;

          ui                       = -ux + uy;
          evenPart =
              omegaEven * (F(0.5) * (pdfNw + pdfSe) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfNw - pdfSe) - ui * w2X3);
          src[I(x, y, z, D3Q19_SE)] = pdfNw - evenPart - oddPart;
          src[I(x, y, z, D3Q19_NW)] = pdfSe - evenPart + oddPart;

          ui                        = ux + uy;
          evenPart =
              omegaEven * (F(0.5) * (pdfNe + pdfSw) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfNe - pdfSw) - ui * w2X3);
          src[I(x, y, z, D3Q19_SW)] = pdfNe - evenPart - oddPart;
          src[I(x, y, z, D3Q19_NE)] = pdfSw - evenPart + oddPart;

          ui                        = -ux + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTw + pdfBe) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfTw - pdfBe) - ui * w2X3);
          src[I(x, y, z, D3Q19_BE)] = pdfTw - evenPart - oddPart;
          src[I(x, y, z, D3Q19_TW)] = pdfBe - evenPart + oddPart;

          ui                        = ux + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTe + pdfBw) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfTe - pdfBw) - ui * w2X3);
          src[I(x, y, z, D3Q19_BW)] = pdfTe - evenPart - oddPart;
          src[I(x, y, z, D3Q19_TE)] = pdfBw - evenPart + oddPart;

          ui                        = -uy + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTs + pdfBn) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfTs - pdfBn) - ui * w2X3);
          src[I(x, y, z, D3Q19_BN)] = pdfTs - evenPart - oddPart;
          src[I(x, y, z, D3Q19_TS)] = pdfBn - evenPart + oddPart;

          ui                        = uy + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTn + pdfBs) - ui * ui * w2NineHalf - w2Indep);
          oddPart                   = omegaOdd * (F(0.5) * (pdfTn - pdfBs) - ui * w2X3);
          src[I(x, y, z, D3Q19_BS)] = pdfTn - evenPart - oddPart;
          src[I(x, y, z, D3Q19_TN)] = pdfBs - evenPart + oddPart;
        }
      }
    }

    // Bounce back after even step
    for (int i = 0; i < kd->nBounceBackPdfs; ++i)
      src[kd->BounceBackPdfsSrc[i]] = src[kd->BounceBackPdfsDst[i]];

    kd->Iteration = iter;

#ifdef VERIFICATION
    kd->PdfsActive = src;
    kernelAddBodyForce(kd, ld, cd);
#endif

    // ----------------------------------------------------------------
    // Odd time step: read from neighbors (inverse dir), write to
    // neighbors (push-like propagation)
    // ----------------------------------------------------------------
    for (int x = oX; x < nX + oX; ++x) {
      for (int y = oY; y < nY + oY; ++y) {
        for (int z = oZ; z < nZ + oZ; ++z) {

          // Odd: load from neighbor cells using inverse directions
          pdfN  = src[I(x, y - 1, z, D3Q19_S)];
          pdfS  = src[I(x, y + 1, z, D3Q19_N)];
          pdfE  = src[I(x - 1, y, z, D3Q19_W)];
          pdfW  = src[I(x + 1, y, z, D3Q19_E)];
          pdfNe = src[I(x - 1, y - 1, z, D3Q19_SW)];
          pdfSe = src[I(x - 1, y + 1, z, D3Q19_NW)];
          pdfNw = src[I(x + 1, y - 1, z, D3Q19_SE)];
          pdfSw = src[I(x + 1, y + 1, z, D3Q19_NE)];
          pdfT  = src[I(x, y, z - 1, D3Q19_B)];
          pdfTn = src[I(x, y - 1, z - 1, D3Q19_BS)];
          pdfTe = src[I(x - 1, y, z - 1, D3Q19_BW)];
          pdfTw = src[I(x + 1, y, z - 1, D3Q19_BE)];
          pdfTs = src[I(x, y + 1, z - 1, D3Q19_BN)];
          pdfB  = src[I(x, y, z + 1, D3Q19_T)];
          pdfBs = src[I(x, y + 1, z + 1, D3Q19_TN)];
          pdfBn = src[I(x, y - 1, z + 1, D3Q19_TS)];
          pdfBw = src[I(x + 1, y, z + 1, D3Q19_TE)];
          pdfBe = src[I(x - 1, y, z + 1, D3Q19_TW)];
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

          // Odd: write to neighbor cells (push propagation)
          src[I(x, y, z, D3Q19_C)] = pdfC - omegaEven * (pdfC - w0 * dirIndepTrm);
          w1Indep                  = w1 * dirIndepTrm;

          ui                       = uy;
          evenPart =
              omegaEven * (F(0.5) * (pdfN + pdfS) - ui * ui * w1NineHalf - w1Indep);
          oddPart                      = omegaOdd * (F(0.5) * (pdfN - pdfS) - ui * w1X3);
          src[I(x, y + 1, z, D3Q19_N)] = pdfN - evenPart - oddPart;
          src[I(x, y - 1, z, D3Q19_S)] = pdfS - evenPart + oddPart;

          ui                           = ux;
          evenPart =
              omegaEven * (F(0.5) * (pdfE + pdfW) - ui * ui * w1NineHalf - w1Indep);
          oddPart                      = omegaOdd * (F(0.5) * (pdfE - pdfW) - ui * w1X3);
          src[I(x + 1, y, z, D3Q19_E)] = pdfE - evenPart - oddPart;
          src[I(x - 1, y, z, D3Q19_W)] = pdfW - evenPart + oddPart;

          ui                           = uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfT + pdfB) - ui * ui * w1NineHalf - w1Indep);
          oddPart                      = omegaOdd * (F(0.5) * (pdfT - pdfB) - ui * w1X3);
          src[I(x, y, z + 1, D3Q19_T)] = pdfT - evenPart - oddPart;
          src[I(x, y, z - 1, D3Q19_B)] = pdfB - evenPart + oddPart;

          w2Indep                      = w2 * dirIndepTrm;

          ui                           = -ux + uy;
          evenPart =
              omegaEven * (F(0.5) * (pdfNw + pdfSe) - ui * ui * w2NineHalf - w2Indep);
          oddPart = omegaOdd * (F(0.5) * (pdfNw - pdfSe) - ui * w2X3);
          src[I(x - 1, y + 1, z, D3Q19_NW)] = pdfNw - evenPart - oddPart;
          src[I(x + 1, y - 1, z, D3Q19_SE)] = pdfSe - evenPart + oddPart;

          ui                                = ux + uy;
          evenPart =
              omegaEven * (F(0.5) * (pdfNe + pdfSw) - ui * ui * w2NineHalf - w2Indep);
          oddPart = omegaOdd * (F(0.5) * (pdfNe - pdfSw) - ui * w2X3);
          src[I(x + 1, y + 1, z, D3Q19_NE)] = pdfNe - evenPart - oddPart;
          src[I(x - 1, y - 1, z, D3Q19_SW)] = pdfSw - evenPart + oddPart;

          ui                                = -ux + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTw + pdfBe) - ui * ui * w2NineHalf - w2Indep);
          oddPart = omegaOdd * (F(0.5) * (pdfTw - pdfBe) - ui * w2X3);
          src[I(x - 1, y, z + 1, D3Q19_TW)] = pdfTw - evenPart - oddPart;
          src[I(x + 1, y, z - 1, D3Q19_BE)] = pdfBe - evenPart + oddPart;

          ui                                = ux + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTe + pdfBw) - ui * ui * w2NineHalf - w2Indep);
          oddPart = omegaOdd * (F(0.5) * (pdfTe - pdfBw) - ui * w2X3);
          src[I(x + 1, y, z + 1, D3Q19_TE)] = pdfTe - evenPart - oddPart;
          src[I(x - 1, y, z - 1, D3Q19_BW)] = pdfBw - evenPart + oddPart;

          ui                                = -uy + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTs + pdfBn) - ui * ui * w2NineHalf - w2Indep);
          oddPart = omegaOdd * (F(0.5) * (pdfTs - pdfBn) - ui * w2X3);
          src[I(x, y - 1, z + 1, D3Q19_TS)] = pdfTs - evenPart - oddPart;
          src[I(x, y + 1, z - 1, D3Q19_BN)] = pdfBn - evenPart + oddPart;

          ui                                = uy + uz;
          evenPart =
              omegaEven * (F(0.5) * (pdfTn + pdfBs) - ui * ui * w2NineHalf - w2Indep);
          oddPart = omegaOdd * (F(0.5) * (pdfTn - pdfBs) - ui * w2X3);
          src[I(x, y + 1, z + 1, D3Q19_TN)] = pdfTn - evenPart - oddPart;
          src[I(x, y - 1, z - 1, D3Q19_BS)] = pdfBs - evenPart + oddPart;
        }
      }
    }

    // Bounce back after odd step
    for (int i = 0; i < kd->nBounceBackPdfs; ++i)
      src[kd->BounceBackPdfsDst[i]] = src[kd->BounceBackPdfsSrc[i]];

    kd->Iteration = iter + 1;

#ifdef VERIFICATION
    kd->PdfsActive = src;
    kernelAddBodyForce(kd, ld, cd);
#endif
  }

  kd->Duration += getTimeStamp();
  kd->PdfsActive = src;

#undef I
}
