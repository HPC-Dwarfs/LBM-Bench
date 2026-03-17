/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of LBM-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include "lbm.h"
#include "timing.h"

void kernelBlkPullSoA(LatticeDesc *ld, KernelData *kd, CaseData *cd) {
  int nX = ld->Dims[0], nY = ld->Dims[1], nZ = ld->Dims[2];
  int *gDims = kd->GlobalDims;
  int oX = kd->Offsets[0], oY = kd->Offsets[1], oZ = kd->Offsets[2];
  int *blk = kd->Blk;

  PdfT omega = cd->Omega;
  PdfT omegaEven = omega;
  PdfT magicParam = F(1.0) / F(12.0);
  PdfT omegaOdd = F(1.0) / (F(0.5) + magicParam / (F(1.0) / omega - F(0.5)));

  PdfT evenPart, oddPart, dir_indep_trm;
  PdfT w_0 = F(1.0) / F(3.0), w_1 = F(1.0) / F(18.0), w_2 = F(1.0) / F(36.0);
  PdfT w_1_x3 = w_1 * F(3.0), w_1_nine_half = w_1 * F(9.0) / F(2.0);
  PdfT w_2_x3 = w_2 * F(3.0), w_2_nine_half = w_2 * F(9.0) / F(2.0);
  PdfT w_1_indep, w_2_indep;
  PdfT ux, uy, uz, ui, dens;
  PdfT pdf_N, pdf_S, pdf_E, pdf_W, pdf_NE, pdf_SE, pdf_NW, pdf_SW;
  PdfT pdf_T, pdf_TN, pdf_TE, pdf_TW, pdf_TS;
  PdfT pdf_B, pdf_BN, pdf_BE, pdf_BW, pdf_BS, pdf_C;

  PdfT *src = kd->Pdfs[0], *dst = kd->Pdfs[1], *tmp;
  int maxIterations = cd->MaxIterations;

#define I(x, y, z, dir) indexSoA(gDims, (x), (y), (z), (dir))

  kd->Duration = -getTimeStamp();

  for (int iter = 0; iter < maxIterations; ++iter) {
    for (int bX = oX; bX < nX + oX; bX += blk[0]) {
      for (int bY = oY; bY < nY + oY; bY += blk[1]) {
        for (int bZ = oZ; bZ < nZ + oZ; bZ += blk[2]) {
          int eX = MIN(bX + blk[0], nX + oX);
          int eY = MIN(bY + blk[1], nY + oY);
          int eZ = MIN(bZ + blk[2], nZ + oZ);

          for (int x = bX; x < eX; ++x) {
            for (int y = bY; y < eY; ++y) {
              for (int z = bZ; z < eZ; ++z) {
                // Pull: read from neighbors
                pdf_N  = src[I(x, y - 1, z, D3Q19_N)];
                pdf_S  = src[I(x, y + 1, z, D3Q19_S)];
                pdf_E  = src[I(x - 1, y, z, D3Q19_E)];
                pdf_W  = src[I(x + 1, y, z, D3Q19_W)];
                pdf_NE = src[I(x - 1, y - 1, z, D3Q19_NE)];
                pdf_SE = src[I(x - 1, y + 1, z, D3Q19_SE)];
                pdf_NW = src[I(x + 1, y - 1, z, D3Q19_NW)];
                pdf_SW = src[I(x + 1, y + 1, z, D3Q19_SW)];
                pdf_T  = src[I(x, y, z - 1, D3Q19_T)];
                pdf_TN = src[I(x, y - 1, z - 1, D3Q19_TN)];
                pdf_TE = src[I(x - 1, y, z - 1, D3Q19_TE)];
                pdf_TW = src[I(x + 1, y, z - 1, D3Q19_TW)];
                pdf_TS = src[I(x, y + 1, z - 1, D3Q19_TS)];
                pdf_B  = src[I(x, y, z + 1, D3Q19_B)];
                pdf_BS = src[I(x, y + 1, z + 1, D3Q19_BS)];
                pdf_BN = src[I(x, y - 1, z + 1, D3Q19_BN)];
                pdf_BW = src[I(x + 1, y, z + 1, D3Q19_BW)];
                pdf_BE = src[I(x - 1, y, z + 1, D3Q19_BE)];
                pdf_C  = src[I(x, y, z, D3Q19_C)];

                ux = pdf_E + pdf_NE + pdf_SE + pdf_TE + pdf_BE - pdf_W - pdf_NW - pdf_SW - pdf_TW - pdf_BW;
                uy = pdf_N + pdf_NE + pdf_NW + pdf_TN + pdf_BN - pdf_S - pdf_SE - pdf_SW - pdf_TS - pdf_BS;
                uz = pdf_T + pdf_TE + pdf_TW + pdf_TN + pdf_TS - pdf_B - pdf_BE - pdf_BW - pdf_BN - pdf_BS;
                dens = pdf_C + pdf_N + pdf_E + pdf_S + pdf_W + pdf_NE + pdf_SE + pdf_SW + pdf_NW +
                       pdf_T + pdf_TN + pdf_TE + pdf_TS + pdf_TW + pdf_B + pdf_BN + pdf_BE + pdf_BS + pdf_BW;

                dir_indep_trm = dens - (ux * ux + uy * uy + uz * uz) * F(3.0) / F(2.0);

                // Pull: write to local cell
                dst[I(x, y, z, D3Q19_C)] = pdf_C - omegaEven * (pdf_C - w_0 * dir_indep_trm);
                w_1_indep = w_1 * dir_indep_trm;

                ui = uy;
                evenPart = omegaEven * (F(0.5) * (pdf_N + pdf_S) - ui * ui * w_1_nine_half - w_1_indep);
                oddPart = omegaOdd * (F(0.5) * (pdf_N - pdf_S) - ui * w_1_x3);
                dst[I(x, y, z, D3Q19_N)] = pdf_N - evenPart - oddPart;
                dst[I(x, y, z, D3Q19_S)] = pdf_S - evenPart + oddPart;

                ui = ux;
                evenPart = omegaEven * (F(0.5) * (pdf_E + pdf_W) - ui * ui * w_1_nine_half - w_1_indep);
                oddPart = omegaOdd * (F(0.5) * (pdf_E - pdf_W) - ui * w_1_x3);
                dst[I(x, y, z, D3Q19_E)] = pdf_E - evenPart - oddPart;
                dst[I(x, y, z, D3Q19_W)] = pdf_W - evenPart + oddPart;

                ui = uz;
                evenPart = omegaEven * (F(0.5) * (pdf_T + pdf_B) - ui * ui * w_1_nine_half - w_1_indep);
                oddPart = omegaOdd * (F(0.5) * (pdf_T - pdf_B) - ui * w_1_x3);
                dst[I(x, y, z, D3Q19_T)] = pdf_T - evenPart - oddPart;
                dst[I(x, y, z, D3Q19_B)] = pdf_B - evenPart + oddPart;

                w_2_indep = w_2 * dir_indep_trm;

                ui = -ux + uy;
                evenPart = omegaEven * (F(0.5) * (pdf_NW + pdf_SE) - ui * ui * w_2_nine_half - w_2_indep);
                oddPart = omegaOdd * (F(0.5) * (pdf_NW - pdf_SE) - ui * w_2_x3);
                dst[I(x, y, z, D3Q19_NW)] = pdf_NW - evenPart - oddPart;
                dst[I(x, y, z, D3Q19_SE)] = pdf_SE - evenPart + oddPart;

                ui = ux + uy;
                evenPart = omegaEven * (F(0.5) * (pdf_NE + pdf_SW) - ui * ui * w_2_nine_half - w_2_indep);
                oddPart = omegaOdd * (F(0.5) * (pdf_NE - pdf_SW) - ui * w_2_x3);
                dst[I(x, y, z, D3Q19_NE)] = pdf_NE - evenPart - oddPart;
                dst[I(x, y, z, D3Q19_SW)] = pdf_SW - evenPart + oddPart;

                ui = -ux + uz;
                evenPart = omegaEven * (F(0.5) * (pdf_TW + pdf_BE) - ui * ui * w_2_nine_half - w_2_indep);
                oddPart = omegaOdd * (F(0.5) * (pdf_TW - pdf_BE) - ui * w_2_x3);
                dst[I(x, y, z, D3Q19_TW)] = pdf_TW - evenPart - oddPart;
                dst[I(x, y, z, D3Q19_BE)] = pdf_BE - evenPart + oddPart;

                ui = ux + uz;
                evenPart = omegaEven * (F(0.5) * (pdf_TE + pdf_BW) - ui * ui * w_2_nine_half - w_2_indep);
                oddPart = omegaOdd * (F(0.5) * (pdf_TE - pdf_BW) - ui * w_2_x3);
                dst[I(x, y, z, D3Q19_TE)] = pdf_TE - evenPart - oddPart;
                dst[I(x, y, z, D3Q19_BW)] = pdf_BW - evenPart + oddPart;

                ui = -uy + uz;
                evenPart = omegaEven * (F(0.5) * (pdf_TS + pdf_BN) - ui * ui * w_2_nine_half - w_2_indep);
                oddPart = omegaOdd * (F(0.5) * (pdf_TS - pdf_BN) - ui * w_2_x3);
                dst[I(x, y, z, D3Q19_TS)] = pdf_TS - evenPart - oddPart;
                dst[I(x, y, z, D3Q19_BN)] = pdf_BN - evenPart + oddPart;

                ui = uy + uz;
                evenPart = omegaEven * (F(0.5) * (pdf_TN + pdf_BS) - ui * ui * w_2_nine_half - w_2_indep);
                oddPart = omegaOdd * (F(0.5) * (pdf_TN - pdf_BS) - ui * w_2_x3);
                dst[I(x, y, z, D3Q19_TN)] = pdf_TN - evenPart - oddPart;
                dst[I(x, y, z, D3Q19_BS)] = pdf_BS - evenPart + oddPart;
              }
            }
          }
        }
      }
    }

    for (int i = 0; i < kd->nBounceBackPdfs; ++i)
      dst[kd->BounceBackPdfsDst[i]] = dst[kd->BounceBackPdfsSrc[i]];

#ifdef VERIFICATION
    kd->PdfsActive = dst;
    kernelAddBodyForce(kd, ld, cd);
#endif

    tmp = src; src = dst; dst = tmp;
  }

  kd->Duration += getTimeStamp();
  kd->PdfsActive = src;
#undef I
}
