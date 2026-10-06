// tiny single tile n:m test for poking at the rtl
// changed: new test. runs one squashed weight tile and scores the output against a few
// "what if it's broken this way" references, so one sim run says which way it's broken

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/gemmini_nm.h"

#define K2 (2 * DIM)

static elem_t a[DIM][K2] row_align(1);
static elem_t a_lo[DIM][DIM] row_align(1);
static elem_t a_hi[DIM][DIM] row_align(1);
static elem_t w[K2][DIM] row_align(1);
static elem_t wc[DIM][DIM] row_align(1);
static nm_idx_t idx[DIM][DIM];
static acc_t out[DIM][DIM] row_align_acc(1);
static acc_t ref[DIM][DIM];

#define ACC_BIT ((uint32_t)1 << (ADDR_LEN - 1))
#define FULL_BIT ((uint32_t)1 << (ADDR_LEN - 3))

// score = how many of the DIM*DIM outputs match this guess
static int score(const char * name, int mode) {
  int hits = 0;
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++) {
      acc_t s = 0;
      for (int r = 0; r < DIM; r++) {
        int g = r / NM_N, id = idx[r][j];
        if (mode == 1) id = 0;                       // index ignored
        if (mode == 2) id = idx[DIM - 1 - r][j];     // index rows flipped
        if (mode == 3) id = idx[j][r];               // index transposed
        int k = g * NM_M + id;
        if (mode == 4) k = r;                        // plain dense with the squashed weights, no a_hi
        s += (acc_t)a[i][k] * wc[r][j];
      }
      if (mode == 5) { // dense + a_hi treated as bias
        s = 0;
        for (int r = 0; r < DIM; r++) s += (acc_t)a[i][r] * wc[r][j];
        s += a_hi[i][j];
      }
      if (mode == 0) ref[i][j] = s;
      hits += out[i][j] == s;
    }
  printf("  %-26s %3d / %d\n", name, hits, DIM * DIM);
  return hits;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif
  printf("nm_sparse_debug %d:%d\n", NM_N, NM_M);
  gemmini_flush(0);

  for (int i = 0; i < DIM; i++)
    for (int k = 0; k < K2; k++)
      a[i][k] = (rand() % 16) - 8;
  for (int k = 0; k < K2; k++)
    for (int j = 0; j < DIM; j++)
      w[k][j] = (rand() % 16) - 8;
  nm_prune(K2, DIM, &w[0][0], DIM, NM_N, NM_M);
  nm_compress(K2, DIM, &w[0][0], DIM, &wc[0][0], &idx[0][0], DIM, NM_N, NM_M);
  for (int i = 0; i < DIM; i++)
    for (int k = 0; k < DIM; k++) {
      a_lo[i][k] = a[i][k];
      a_hi[i][k] = a[i][DIM + k];
    }

  // a_lo bank 0, a_hi bank 1, weights bank 2
  const uint32_t lo_sp = 0, hi_sp = BANK_ROWS, w_sp = 2 * BANK_ROWS;
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_mvin(a_lo, lo_sp);
  gemmini_mvin(a_hi, hi_sp);
  gemmini_mvin(wc, w_sp);

  gemmini_nm_config_ex(1, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, 1);
  nm_load_meta(&idx[0][0], DIM, NM_M);
  gemmini_preload(w_sp, ACC_BIT);
  gemmini_compute_preloaded(lo_sp, hi_sp);
  gemmini_fence();

  gemmini_config_st(DIM * sizeof(acc_t));
  gemmini_mvout(out, ACC_BIT | FULL_BIT);
  gemmini_fence();
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_fence();

  int ok = score("correct sparse", 0);
  score("index ignored (all 0)", 1);
  score("index rows flipped", 2);
  score("index transposed", 3);
  score("dense, no a_hi", 4);
  score("dense + a_hi as bias", 5);

  printf("row 0 got: ");
  for (int j = 0; j < DIM; j++) printf("%d ", (int)out[0][j]);
  printf("\nrow 0 want: ");
  for (int j = 0; j < DIM; j++) printf("%d ", (int)ref[0][j]);
  printf("\n");

  if (ok != DIM * DIM) {
    printf("nm_sparse_debug FAILED\n");
    exit(1);
  }
  printf("nm_sparse_debug passed\n");
  exit(0);
}
