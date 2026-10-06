// perf breakdown for the n:m sparse mode vs dense ws on the same pruned weights
// changed: new test. no cpu reference here (too slow in rtl sim), it checks the sparse
// outputs against the dense gemmini outputs instead. nm_sparse_matmul covers the cpu check.
//
// runs (no reuse = fresh weight preload for every a tile, reuse = preload once per weight
// tile then compute_accumulated over all the a tiles, which is what tiled_matmul does):
//   dense           plain ws, no reuse
//   dense_bias      plain ws with a real bias tile, same b port the sparse a_hi uses
//   sparse          meta(0), then preload(q), meta(q+1), compute(q), no reuse
//   sparse_nat      meta(q), preload(q), compute(q), no reuse
//   dense_reuse     dense with weight reuse
//   sparse_reuse    sparse with weight reuse

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

// cpu_only = run the same issue loop but swap every gemmini instruction for an empty asm
// that still uses its operands, so the time is just what the core needs to build the
// command stream (no accelerator, no backpressure)
static int cpu_only = 0;
#define SINK(x, y) asm volatile ("" :: "r"((uint64_t)(x)), "r"((uint64_t)(y)))
#define PRELOAD(b, c) do { if (cpu_only) SINK(b, c); else { gemmini_preload(b, c); } } while (0)
#define COMPUTE_PRE(a, bd) do { if (cpu_only) SINK(a, bd); else { gemmini_compute_preloaded(a, bd); } } while (0)
#define COMPUTE_ACC(a, bd) do { if (cpu_only) SINK(a, bd); else { gemmini_compute_accumulated(a, bd); } } while (0)
static inline void send_meta(const uint64_t * packed) {
  if (!cpu_only) { nm_send_meta(packed, NM_M); return; }
  for (int i = 0; i < nm_meta_cmds(NM_M); i++) SINK(packed[2 * i], packed[2 * i + 1]);
}

static const int ctr_ids[8] = {
  EXE_ACTIVE_CYCLE, RESERVATION_STATION_FULL_CYCLES, EXE_CONTROL_Q_BLOCK_CYCLE, EXE_OVERLAP_HAZ_CYCLE,
  SCRATCHPAD_A_WAIT_CYCLE, SCRATCHPAD_B_WAIT_CYCLE, SCRATCHPAD_D_WAIT_CYCLE, MAIN_EX_CYCLES
};
static const char * ctr_names[8] = {
  "ex_active", "rs_full", "ctrl_q_block", "overlap_haz",
  "spad_a_wait", "spad_b_wait", "spad_d_wait", "main_ex"
};
static uint32_t ctr_vals[8];

static void ctr_start() {
  counter_reset();
  for (int i = 0; i < 8; i++) counter_configure(i, ctr_ids[i]);
}

// freeze the counters right when the work is done, some of them keep ticking on
// stale control signals while the array sits idle
static void ctr_stop() {
  counter_snapshot_take();
  for (int i = 0; i < 8; i++) ctr_vals[i] = counter_read(i);
  counter_snapshot_reset();
}

static void report(const char * name, uint64_t start, uint64_t issued, uint64_t end, int pairs) {
  printf("%-14s total %6llu  cpu_issue_done %6llu  pairs %3d  cyc/pair %4llu |",
    name, (unsigned long long)(end - start), (unsigned long long)(issued - start), pairs,
    (unsigned long long)((end - start) / pairs));
  for (int i = 0; i < 8; i++) printf(" %s=%u", ctr_names[i], ctr_vals[i]);
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

// bias = 1 streams a real bias tile through the b port (results not checked)
static void run_dense(int bias, const char * name) {
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_fence();
  ctr_start();
  uint64_t start = read_cycles();
  for (int it = 0; it < IT; it++)
    for (int jt = 0; jt < JT; jt++)
      for (int kt = 0; kt < KT; kt++) {
        uint32_t c = ACC_BIT | (kt > 0 ? ACCUM_BIT : 0) | C_ACC(it, jt);
        PRELOAD(W_SP(kt, jt), c);
        // the bias just reuses an a tile from the other bank, values dont matter here
        COMPUTE_PRE(A_SP(it, kt), bias ? A_SP(it, kt ^ 1) : GARBAGE_ADDR);
      }
  uint64_t issued = read_cycles();
  gemmini_fence();
  uint64_t end = read_cycles();
  ctr_stop();
  report(name, start, issued, end, IT * JT * KT);
}

// weight reuse: one real preload per weight tile, then preload(garbage, c) +
// compute_accumulated keeps the same weights for the rest of the a tiles
static void run_reuse(int sparse, const char * name) {
  const int ktiles = sparse ? KCT : KT;
  const int blocks = JT * ktiles;
  // gemmini macros are multi statement, keep the braces
  if (sparse) { gemmini_nm_config_ex(1, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, 1); }
  else { gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0); }
  gemmini_fence();
  ctr_start();
  uint64_t start = read_cycles();
  if (sparse) send_meta(meta[0][0]);
  for (int b = 0; b < blocks; b++) {
    const int jt = b / ktiles, kt = b % ktiles;
    for (int it = 0; it < IT; it++) {
      uint32_t c = ACC_BIT | (kt > 0 ? ACCUM_BIT : 0) | C_ACC(it, jt);
      if (it == 0) { PRELOAD(sparse ? WC_SP(kt, jt) : W_SP(kt, jt), c); }
      else { PRELOAD(GARBAGE_ADDR, c); }
      // next block's indexes go after the last preload of this block, before its compute
      if (sparse && it == IT - 1 && b + 1 < blocks)
        send_meta(meta[(b + 1) % ktiles][(b + 1) / ktiles]);
      const uint32_t a_lo = sparse ? A_SP(it, 2 * kt) : A_SP(it, kt);
      const uint32_t bd = sparse ? A_SP(it, 2 * kt + 1) : GARBAGE_ADDR;
      if (it == 0) { COMPUTE_PRE(a_lo, bd); }
      else { COMPUTE_ACC(a_lo, bd); }
    }
  }
  uint64_t issued = read_cycles();
  gemmini_fence();
  uint64_t end = read_cycles();
  ctr_stop();
  report(name, start, issued, end, blocks * IT);
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_fence();
}

// mode 0 = meta for the next tile sent between preload and compute, 1 = natural order
static void run_sparse(int mode, const char * name) {
  const int ntiles = IT * JT * KCT;
  gemmini_nm_config_ex(1, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, 1);
  gemmini_fence();
  ctr_start();
  uint64_t start = read_cycles();
  if (mode == 0) send_meta(meta[0][0]);
  for (int q = 0; q < ntiles; q++) {
    const int it = q / (JT * KCT), jt = (q / KCT) % JT, t = q % KCT;
    uint32_t c = ACC_BIT | (t > 0 ? ACCUM_BIT : 0) | C_ACC(it, jt);
    if (mode == 1) send_meta(meta[t][jt]);
    PRELOAD(WC_SP(t, jt), c);
    if (mode == 0 && q + 1 < ntiles) {
      const int jn = ((q + 1) / KCT) % JT, tn = (q + 1) % KCT;
      send_meta(meta[tn][jn]);
    }
    COMPUTE_PRE(A_SP(it, 2 * t), A_SP(it, 2 * t + 1));
  }
  uint64_t issued = read_cycles();
  gemmini_fence();
  uint64_t end = read_cycles();
  ctr_stop();
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

  run_dense(0, "dense");
  store_c(c_dense);
  run_dense(1, "dense_bias");
  int fails = 0;
  run_sparse(0, "sparse");
  store_c(c_sparse);
  if (!same()) { printf("sparse result wrong\n"); fails++; }

  run_sparse(1, "sparse_nat");
  store_c(c_sparse);
  if (!same()) { printf("sparse_nat result wrong\n"); fails++; }


  run_reuse(0, "dense_reuse");
  store_c(c_dense);
  run_reuse(1, "sparse_reuse");
  store_c(c_sparse);
  if (!same()) { printf("sparse_reuse result wrong\n"); fails++; }

  // same loops again with no accelerator, just the core building the command stream
  printf("cpu only (no gemmini instructions issued):\n");
  cpu_only = 1;
  run_dense(0, "cpu_dense");
  run_sparse(0, "cpu_sparse");
  run_sparse(1, "cpu_sparse_nat");
  run_reuse(0, "cpu_dense_reuse");
  run_reuse(1, "cpu_sparse_reuse");
  cpu_only = 0;

  if (fails) {
    printf("nm_sparse_perf FAILED\n");
    exit(1);
  }
  printf("nm_sparse_perf passed\n");
  exit(0);
}
