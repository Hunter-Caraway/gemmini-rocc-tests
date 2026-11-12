// stage 1 test for n:m sparsity, cpu only, no gemmini instructions
// changed: new test, checks prune -> compress -> decompress -> matmul all line up
// runs 2:4, 4:8 and 1:2 (also with k not a multiple of m), plus index packing of edge tiles

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

#define TI 16
#define TK 64
#define TJ 16

static elem_t a[TI][TK];
static elem_t w[TK][TJ];
static elem_t w_back[TK][TJ];
static elem_t wc[TK][TJ];
static nm_idx_t idx[TK][TJ];
static acc_t c_dense[TI][TJ];
static acc_t c_sparse[TI][TJ];

static int run_one(int n, int m, size_t k) {
  // fresh full-range int8 data each round
  for (size_t y = 0; y < TI; y++)
    for (size_t z = 0; z < TK; z++)
      a[y][z] = (rand() % 256) - 128;
  for (size_t z = 0; z < TK; z++)
    for (size_t x = 0; x < TJ; x++)
      w[z][x] = (rand() % 256) - 128;

  nm_prune(k, TJ, &w[0][0], TJ, n, m);
  if (!nm_check(k, TJ, &w[0][0], TJ, n, m)) {
    printf("%d:%d k=%d prune left too many nonzeros\n", n, m, (int)k);
    return 1;
  }

  nm_compress(k, TJ, &w[0][0], TJ, &wc[0][0], &idx[0][0], TJ, n, m);
  nm_decompress(k, TJ, &wc[0][0], &idx[0][0], TJ, &w_back[0][0], TJ, n, m);
  for (size_t z = 0; z < k; z++)
    for (size_t x = 0; x < TJ; x++)
      if (w_back[z][x] != w[z][x]) {
        printf("%d:%d k=%d decompress mismatch at (%d,%d): %d vs %d\n", n, m, (int)k, (int)z, (int)x, w_back[z][x], w[z][x]);
        return 1;
      }

  nm_ref_dense(TI, k, TJ, &a[0][0], TK, &w[0][0], TJ, &c_dense[0][0], TJ);
  nm_ref_sparse(TI, k, TJ, &a[0][0], TK, &wc[0][0], &idx[0][0], TJ, &c_sparse[0][0], TJ, n, m);
  for (size_t y = 0; y < TI; y++)
    for (size_t x = 0; x < TJ; x++)
      if (c_dense[y][x] != c_sparse[y][x]) {
        printf("%d:%d k=%d matmul mismatch at (%d,%d): dense %d sparse %d\n", n, m, (int)k, (int)y, (int)x, c_dense[y][x], c_sparse[y][x]);
        return 1;
      }

  printf("%d:%d ok (k=%d squashed to %d rows)\n", n, m, (int)k, (int)(nm_groups(k, m) * n));
  return 0;
}

// an edge tile (rows x cols) must pack exactly like the same tile zero padded to DIM x DIM
#define BIG (DIM + 5)
static nm_idx_t big[BIG][BIG];
static nm_idx_t padded[DIM][DIM];

static int pack_one(size_t r0, size_t c0) {
  const size_t rows = BIG - r0 < DIM ? BIG - r0 : DIM, cols = BIG - c0 < DIM ? BIG - c0 : DIM;
  uint64_t got[2 * DIM], want[2 * DIM];
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++)
      padded[r][c] = r < rows && c < cols ? big[r0 + r][c0 + c] : 0;
  nm_pack_meta(&big[r0][c0], BIG, rows, cols, NM_M, got);
  nm_pack_meta(&padded[0][0], DIM, DIM, DIM, NM_M, want);
  for (int i = 0; i < 2 * nm_meta_cmds(NM_M); i++)
    if (got[i] != want[i]) {
      printf("edge tile at (%d,%d) %dx%d packs wrong in word %d\n", (int)r0, (int)c0, (int)rows, (int)cols, i);
      return 1;
    }
  return 0;
}

static int pack_edges() {
  for (size_t r = 0; r < BIG; r++)
    for (size_t c = 0; c < BIG; c++)
      big[r][c] = 1 + rand() % (NM_M - 1); // never 0, so a dropped or extra index shows
  int fails = pack_one(0, 0) + pack_one(0, DIM) + pack_one(DIM, 0) + pack_one(DIM, DIM);
  if (!fails) printf("edge tile packing ok\n");
  return fails;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  int fails = 0;
  fails += run_one(2, 4, TK);
  fails += run_one(4, 8, TK);
  fails += run_one(1, 2, TK);
  fails += run_one(2, 4, TK - 1);
  fails += run_one(4, 8, TK - 3);
  fails += pack_edges();

  if (fails) {
    printf("nm_sparse_sw FAILED\n");
    exit(1);
  }
  printf("nm_sparse_sw passed\n");
  exit(0);
}
