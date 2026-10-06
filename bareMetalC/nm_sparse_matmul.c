// n:m sparse matmul on gemmini vs dense matmul on the same pruned weights
// changed: new test. checks both against the cpu reference and prints cycles for each
// c (i x j) = a (i x k) * w (k x j), w pruned n:m down each column
// sizes are small by default so rtl sim doesnt take forever, bump with -DNM_I=.. etc

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

#ifndef NM_I
#define NM_I (2*DIM)
#endif
#ifndef NM_K
#define NM_K (4*DIM)
#endif
#ifndef NM_J
#define NM_J (2*DIM)
#endif

#define IT (NM_I / DIM)
#define KT (NM_K / DIM)
#define JT (NM_J / DIM)
#define KC (NM_K / NM_M * NM_N)   // squashed weight rows
#define KCT (KC / DIM)            // squashed weight tiles

#if (NM_M != 2 * NM_N)
#error "sparse mode needs m == 2*n"
#endif
#if (NM_K % (2 * DIM)) != 0
#error "k has to be a multiple of 2*DIM"
#endif

static elem_t a[NM_I][NM_K] row_align(1);
static elem_t w[NM_K][NM_J] row_align(1);
static elem_t wc[KC][NM_J] row_align(1);
static nm_idx_t idx[KC][NM_J];
static acc_t gold[NM_I][NM_J];
static acc_t c_dense[NM_I][NM_J] row_align_acc(1);
static acc_t c_sparse[NM_I][NM_J] row_align_acc(1);

// spad layout: a tiles first, then weight tiles
#define A_SP(it, kt) (((it) * KT + (kt)) * DIM)
#define W_BASE (IT * KT * DIM)
#define W_SP(kt, jt) (W_BASE + ((kt) * JT + (jt)) * DIM)
#define ACC_BIT ((uint32_t)1 << (ADDR_LEN - 1))
#define ACCUM_BIT ((uint32_t)1 << (ADDR_LEN - 2))
#define FULL_BIT ((uint32_t)1 << (ADDR_LEN - 3))
#define C_ACC(it, jt) (((it) * JT + (jt)) * DIM)

static void load_a() {
  gemmini_config_ld(NM_K * sizeof(elem_t));
  for (int it = 0; it < IT; it++)
    for (int kt = 0; kt < KT; kt++)
      gemmini_mvin(&a[it * DIM][kt * DIM], A_SP(it, kt));
}

static void store_c(acc_t c[NM_I][NM_J]) {
  gemmini_config_st(NM_J * sizeof(acc_t));
  for (int it = 0; it < IT; it++)
    for (int jt = 0; jt < JT; jt++)
      gemmini_mvout(&c[it * DIM][jt * DIM], ACC_BIT | FULL_BIT | C_ACC(it, jt));
}

static int check(const char * name, acc_t c[NM_I][NM_J]) {
  for (int y = 0; y < NM_I; y++)
    for (int x = 0; x < NM_J; x++)
      if (c[y][x] != gold[y][x]) {
        printf("%s mismatch at (%d,%d): got %d want %d\n", name, y, x, c[y][x], gold[y][x]);
        return 1;
      }
  return 0;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  printf("nm_sparse_matmul: %d:%d, i=%d k=%d j=%d\n", NM_N, NM_M, NM_I, NM_K, NM_J);
  gemmini_flush(0);

  for (int y = 0; y < NM_I; y++)
    for (int z = 0; z < NM_K; z++)
      a[y][z] = (rand() % 16) - 8;
  for (int z = 0; z < NM_K; z++)
    for (int x = 0; x < NM_J; x++)
      w[z][x] = (rand() % 16) - 8;

  nm_prune(NM_K, NM_J, &w[0][0], NM_J, NM_N, NM_M);
  nm_compress(NM_K, NM_J, &w[0][0], NM_J, &wc[0][0], &idx[0][0], NM_J, NM_N, NM_M);
  nm_ref_dense(NM_I, NM_K, NM_J, &a[0][0], NM_K, &w[0][0], NM_J, &gold[0][0], NM_J);

  // ---- dense run, plain ws on the pruned (zero filled) weights ----
  load_a();
  gemmini_config_ld(NM_J * sizeof(elem_t));
  for (int kt = 0; kt < KT; kt++)
    for (int jt = 0; jt < JT; jt++)
      gemmini_mvin(&w[kt * DIM][jt * DIM], W_SP(kt, jt));
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_fence();

  uint64_t d_start = read_cycles();
  for (int it = 0; it < IT; it++)
    for (int jt = 0; jt < JT; jt++)
      for (int kt = 0; kt < KT; kt++) {
        uint32_t c = ACC_BIT | (kt > 0 ? ACCUM_BIT : 0) | C_ACC(it, jt);
        gemmini_preload(W_SP(kt, jt), c);
        gemmini_compute_preloaded(A_SP(it, kt), GARBAGE_ADDR);
      }
  gemmini_fence();
  uint64_t d_end = read_cycles();

  store_c(c_dense);
  gemmini_fence();

  // ---- sparse run, squashed weights + idx ----
  gemmini_config_ld(NM_J * sizeof(elem_t));
  for (int t = 0; t < KCT; t++)
    for (int jt = 0; jt < JT; jt++)
      gemmini_mvin(&wc[t * DIM][jt * DIM], W_SP(t, jt));
  gemmini_nm_config_ex(1, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, 1);
  gemmini_fence();

  uint64_t s_start = read_cycles();
  for (int it = 0; it < IT; it++)
    for (int jt = 0; jt < JT; jt++)
      for (int t = 0; t < KCT; t++) {
        uint32_t c = ACC_BIT | (t > 0 ? ACCUM_BIT : 0) | C_ACC(it, jt);
        nm_load_meta(&idx[t * DIM][jt * DIM], NM_J, NM_M);
        gemmini_preload(W_SP(t, jt), c);
        // one squashed tile eats two dense a tiles
        gemmini_compute_preloaded(A_SP(it, 2 * t), A_SP(it, 2 * t + 1));
      }
  gemmini_fence();
  uint64_t s_end = read_cycles();

  store_c(c_sparse);
  gemmini_fence();

  // back to normal so nothing after us gets surprised
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_fence();

  int fails = check("dense", c_dense) + check("sparse", c_sparse);

  printf("dense  cycles: %llu (%d preload/compute pairs)\n", (unsigned long long)(d_end - d_start), IT * JT * KT);
  printf("sparse cycles: %llu (%d preload/compute pairs, %d meta cmds each)\n", (unsigned long long)(s_end - s_start), IT * JT * KCT, (DIM + 2 * (64 / (DIM * nm_idx_bits(NM_M))) - 1) / (2 * (64 / (DIM * nm_idx_bits(NM_M)))));

  if (fails) {
    printf("nm_sparse_matmul FAILED\n");
    exit(1);
  }
  printf("nm_sparse_matmul passed\n");
  exit(0);
}
