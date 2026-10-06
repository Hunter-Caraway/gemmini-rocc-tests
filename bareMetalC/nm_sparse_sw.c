// stage 1 test for n:m sparsity, cpu only, no gemmini instructions
// changed: new test, checks prune -> compress -> decompress -> matmul all line up
// runs both 2:4 and 4:8 so the generic n:m path gets exercised

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

static int run_one(int n, int m) {
  // fresh random data each round
  for (size_t y = 0; y < TI; y++)
    for (size_t z = 0; z < TK; z++)
      a[y][z] = (rand() % 64) - 32;
  for (size_t z = 0; z < TK; z++)
    for (size_t x = 0; x < TJ; x++)
      w[z][x] = (rand() % 64) - 32;

  nm_prune(TK, TJ, &w[0][0], TJ, n, m);
  if (!nm_check(TK, TJ, &w[0][0], TJ, n, m)) {
    printf("%d:%d prune left too many nonzeros\n", n, m);
    return 1;
  }

  nm_compress(TK, TJ, &w[0][0], TJ, &wc[0][0], &idx[0][0], TJ, n, m);
  nm_decompress(TK, TJ, &wc[0][0], &idx[0][0], TJ, &w_back[0][0], TJ, n, m);
  for (size_t z = 0; z < TK; z++)
    for (size_t x = 0; x < TJ; x++)
      if (w_back[z][x] != w[z][x]) {
        printf("%d:%d decompress mismatch at (%d,%d): %d vs %d\n", n, m, (int)z, (int)x, w_back[z][x], w[z][x]);
        return 1;
      }

  nm_ref_dense(TI, TK, TJ, &a[0][0], TK, &w[0][0], TJ, &c_dense[0][0], TJ);
  nm_ref_sparse(TI, TK, TJ, &a[0][0], TK, &wc[0][0], &idx[0][0], TJ, &c_sparse[0][0], TJ, n, m);
  for (size_t y = 0; y < TI; y++)
    for (size_t x = 0; x < TJ; x++)
      if (c_dense[y][x] != c_sparse[y][x]) {
        printf("%d:%d matmul mismatch at (%d,%d): dense %d sparse %d\n", n, m, (int)y, (int)x, c_dense[y][x], c_sparse[y][x]);
        return 1;
      }

  printf("%d:%d ok (k=%d squashed to %d rows)\n", n, m, TK, TK / m * n);
  return 0;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  int fails = 0;
  fails += run_one(2, 4);
  fails += run_one(4, 8);
  fails += run_one(1, 2);

  if (fails) {
    printf("nm_sparse_sw FAILED\n");
    exit(1);
  }
  printf("nm_sparse_sw passed\n");
  exit(0);
}
