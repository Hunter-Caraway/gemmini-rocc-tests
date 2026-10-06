// perf breakdown for the n:m sparse mode vs dense ws on the same pruned weights
// changed: new test. no cpu reference here (too slow in rtl sim), it checks the sparse
// outputs against the dense gemmini outputs instead. nm_sparse_matmul covers the cpu check.
//
// runs:
//   dense         plain ws, preload/compute per dense weight tile
//   sparse        preload(q), meta(q+1), compute(q)  <- the intended order
//   sparse_nat    meta(q), preload(q), compute(q)    <- simple order, kills the overlap
//   sparse_nometa indexes sent once up front, results are junk, shows the datapath-only time

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/gemmini_counter.h"
#include "include/gemmini_nm.h"

#ifndef NM_I
#define NM_I (4*DIM)
#endif
#ifndef NM_K
#define NM_K (8*DIM)
#endif
#ifndef NM_J
#define NM_J (2*DIM)
#endif

#define IT (NM_I / DIM)
#define KT (NM_K / DIM)
#define JT (NM_J / DIM)
#define KC (NM_K / NM_M * NM_N)
#define KCT (KC / DIM)

static elem_t a[NM_I][NM_K] row_align(1);
static elem_t w[NM_K][NM_J] row_align(1);
static elem_t wc[KC][NM_J] row_align(1);
static nm_idx_t idx[KC][NM_J];
static uint64_t meta[KCT][JT][2 * DIM];
static acc_t c_dense[NM_I][NM_J] row_align_acc(1);
static acc_t c_sparse[NM_I][NM_J] row_align_acc(1);

// a: even k tiles bank 0, odd bank 1. dense weights bank 2, squashed weights bank 3
#define A_SP(it, kt) (((kt) % 2) * BANK_ROWS + ((it) * (KT / 2) + (kt) / 2) * DIM)
#define W_SP(kt, jt) (2 * BANK_ROWS + ((kt) * JT + (jt)) * DIM)
#define WC_SP(t, jt) (3 * BANK_ROWS + ((t) * JT + (jt)) * DIM)
#define ACC_BIT ((uint32_t)1 << (ADDR_LEN - 1))
#define ACCUM_BIT ((uint32_t)1 << (ADDR_LEN - 2))
#define FULL_BIT ((uint32_t)1 << (ADDR_LEN - 3))
#define C_ACC(it, jt) (((it) * JT + (jt)) * DIM)

static const int ctr_ids[8] = {
  EXE_ACTIVE_CYCLE, EXE_FLUSH_CYCLE, EXE_PRELOAD_HAZ_CYCLE, EXE_OVERLAP_HAZ_CYCLE,
  RESERVATION_STATION_FULL_CYCLES, EXE_CONTROL_Q_BLOCK_CYCLE, SCRATCHPAD_B_WAIT_CYCLE, MAIN_EX_CYCLES
};
static const char * ctr_names[8] = {
  "ex_active", "ex_flush", "preload_haz", "overlap_haz",
  "rs_full", "ctrl_q_block", "spad_b_wait", "main_ex"
};

static void ctr_start() {
  for (int i = 0; i < 8; i++) counter_configure(i, ctr_ids[i]);
  counter_reset();
  for (int i = 0; i < 8; i++) counter_configure(i, ctr_ids[i]);
}

static void report(const char * name, uint64_t start, uint64_t issued, uint64_t end, int pairs) {
  printf("%-14s total %6llu  cpu_issue_done %6llu  pairs %3d  cyc/pair %4llu |",
    name, (unsigned long long)(end - start), (unsigned long long)(issued - start), pairs,
    (unsigned long long)((end - start) / pairs));
  for (int i = 0; i < 8; i++) printf(" %s=%d", ctr_names[i], counter_read(i));
  printf("\n");
}

static void store_c(acc_t c[NM_I][NM_J]) {
  gemmini_config_st(NM_J * sizeof(acc_t));
  for (int it = 0; it < IT; it++)
    for (int jt = 0; jt < JT; jt++)
      gemmini_mvout(&c[it * DIM][jt * DIM], ACC_BIT | FULL_BIT | C_ACC(it, jt));
  gemmini_fence();
}

static int same(void) {
  for (int y = 0; y < NM_I; y++)
    for (int x = 0; x < NM_J; x++)
      if (c_dense[y][x] != c_sparse[y][x]) {
        printf("  mismatch at (%d,%d): dense %d sparse %d\n", y, x, c_dense[y][x], c_sparse[y][x]);
        return 0;
      }
  return 1;
}

// mode 0 = intended order, 1 = natural order, 2 = no meta in the loop
static void run_sparse(int mode, const char * name) {
  const int ntiles = IT * JT * KCT;
  gemmini_nm_config_ex(1, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, 1);
  if (mode == 2) nm_send_meta(meta[0][0], NM_M);
  gemmini_fence();
  ctr_start();
  uint64_t start = read_cycles();
  if (mode == 0) nm_send_meta(meta[0][0], NM_M);
  for (int q = 0; q < ntiles; q++) {
    const int it = q / (JT * KCT), jt = (q / KCT) % JT, t = q % KCT;
    uint32_t c = ACC_BIT | (t > 0 ? ACCUM_BIT : 0) | C_ACC(it, jt);
    if (mode == 1) nm_send_meta(meta[t][jt], NM_M);
    gemmini_preload(WC_SP(t, jt), c);
    if (mode == 0 && q + 1 < ntiles) {
      const int jn = ((q + 1) / KCT) % JT, tn = (q + 1) % KCT;
      nm_send_meta(meta[tn][jn], NM_M);
    }
    gemmini_compute_preloaded(A_SP(it, 2 * t), A_SP(it, 2 * t + 1));
  }
  uint64_t issued = read_cycles();
  gemmini_fence();
  uint64_t end = read_cycles();
  report(name, start, issued, end, ntiles);
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_fence();
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif
  printf("nm_sparse_perf: %d:%d, i=%d k=%d j=%d\n", NM_N, NM_M, NM_I, NM_K, NM_J);
  gemmini_flush(0);

  for (int y = 0; y < NM_I; y++)
    for (int z = 0; z < NM_K; z++)
      a[y][z] = (rand() % 16) - 8;
  for (int z = 0; z < NM_K; z++)
    for (int x = 0; x < NM_J; x++)
      w[z][x] = (rand() % 16) - 8;
  nm_prune(NM_K, NM_J, &w[0][0], NM_J, NM_N, NM_M);
  nm_compress(NM_K, NM_J, &w[0][0], NM_J, &wc[0][0], &idx[0][0], NM_J, NM_N, NM_M);
  for (int t = 0; t < KCT; t++)
    for (int jt = 0; jt < JT; jt++)
      nm_pack_meta(&idx[t * DIM][jt * DIM], NM_J, NM_M, meta[t][jt]);

  // everything lives in the scratchpad before any timing starts
  gemmini_config_ld(NM_K * sizeof(elem_t));
  for (int it = 0; it < IT; it++)
    for (int kt = 0; kt < KT; kt++)
      gemmini_mvin(&a[it * DIM][kt * DIM], A_SP(it, kt));
  gemmini_config_ld(NM_J * sizeof(elem_t));
  for (int kt = 0; kt < KT; kt++)
    for (int jt = 0; jt < JT; jt++)
      gemmini_mvin(&w[kt * DIM][jt * DIM], W_SP(kt, jt));
  for (int t = 0; t < KCT; t++)
    for (int jt = 0; jt < JT; jt++)
      gemmini_mvin(&wc[t * DIM][jt * DIM], WC_SP(t, jt));
  gemmini_fence();

  // dense
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_fence();
  ctr_start();
  uint64_t start = read_cycles();
  for (int it = 0; it < IT; it++)
    for (int jt = 0; jt < JT; jt++)
      for (int kt = 0; kt < KT; kt++) {
        uint32_t c = ACC_BIT | (kt > 0 ? ACCUM_BIT : 0) | C_ACC(it, jt);
        gemmini_preload(W_SP(kt, jt), c);
        gemmini_compute_preloaded(A_SP(it, kt), GARBAGE_ADDR);
      }
  uint64_t issued = read_cycles();
  gemmini_fence();
  uint64_t end = read_cycles();
  report("dense", start, issued, end, IT * JT * KT);
  store_c(c_dense);

  int fails = 0;
  run_sparse(0, "sparse");
  store_c(c_sparse);
  if (!same()) { printf("sparse result wrong\n"); fails++; }

  run_sparse(1, "sparse_nat");
  store_c(c_sparse);
  if (!same()) { printf("sparse_nat result wrong\n"); fails++; }

  run_sparse(2, "sparse_nometa");
  printf("(sparse_nometa results are junk on purpose)\n");

  if (fails) {
    printf("nm_sparse_perf FAILED\n");
    exit(1);
  }
  printf("nm_sparse_perf passed\n");
  exit(0);
}
