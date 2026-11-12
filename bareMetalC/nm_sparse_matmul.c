// n:m sparse matmul on gemmini vs dense matmul on the same pruned weights
// changed: new test. checks both against the cpu reference and prints cycles for each
// c (i x j) = a (i x k) * w (k x j), w pruned n:m down each column, full-range int8 data
// outputs and accumulators are poisoned before every run, so a run that writes nothing fails.
// the sparse run starts after a half-sent index tile and gemmini_nm_reset(), so a broken reset fails.
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

#if (NM_K % (2 * DIM)) != 0
#error "k has to be a multiple of 2*DIM"
#endif
#if (NM_I / DIM) * (NM_K / DIM / 2) * DIM > BANK_ROWS
#error "a doesnt fit in its two banks"
#endif
#if KT * JT * DIM > BANK_ROWS
#error "w doesnt fit in its bank"
#endif
#if IT * JT * DIM > ACC_ROWS
#error "c doesnt fit in the accumulator"
#endif

static elem_t a[NM_I][NM_K] row_align(1);
static elem_t w[NM_K][NM_J] row_align(1);
static elem_t wc[KC][NM_J] row_align(1);
static nm_idx_t idx[KC][NM_J];
static uint64_t meta[KCT][JT][2 * DIM]; // packed indexes per weight tile
static acc_t gold[NM_I][NM_J];
static acc_t c_dense[NM_I][NM_J] row_align_acc(1);
static acc_t c_sparse[NM_I][NM_J] row_align_acc(1);
static acc_t poison[DIM][DIM] row_align_acc(1);
#define POISON ((acc_t)0x5EADBEEF)

// even a tiles in bank 0, odd in bank 1, weights in bank 2, so a_lo / a_hi / weights never share a bank
#define A_SP(it, kt) (((kt) % 2) * BANK_ROWS + ((it) * (KT / 2) + (kt) / 2) * DIM)
#define W_SP(kt, jt) (2 * BANK_ROWS + ((kt) * JT + (jt)) * DIM)
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

// fill the output and every accumulator tile the runs use with POISON
static void poison_c(acc_t c[NM_I][NM_J]) {
  for (int y = 0; y < NM_I; y++)
    for (int x = 0; x < NM_J; x++)
      c[y][x] = POISON;
  gemmini_config_ld(DIM * sizeof(acc_t));
  for (int it = 0; it < IT; it++)
    for (int jt = 0; jt < JT; jt++)
      gemmini_mvin(poison, ACC_BIT | C_ACC(it, jt));
  gemmini_fence();
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
      a[y][z] = (rand() % 256) - 128;
  for (int z = 0; z < NM_K; z++)
    for (int x = 0; x < NM_J; x++)
      w[z][x] = (rand() % 256) - 128;

  nm_prune(NM_K, NM_J, &w[0][0], NM_J, NM_N, NM_M);
  nm_compress(NM_K, NM_J, &w[0][0], NM_J, &wc[0][0], &idx[0][0], NM_J, NM_N, NM_M);
  nm_ref_dense(NM_I, NM_K, NM_J, &a[0][0], NM_K, &w[0][0], NM_J, &gold[0][0], NM_J);
  for (int y = 0; y < DIM; y++)
    for (int x = 0; x < DIM; x++)
      poison[y][x] = POISON;
  // index packing is weight prep, same as compressing, so it happens outside the timed part
  for (int t = 0; t < KCT; t++)
    for (int jt = 0; jt < JT; jt++)
      nm_pack_meta(&idx[t * DIM][jt * DIM], NM_J, DIM, DIM, NM_M, meta[t][jt]);

  // ---- dense run, plain ws on the pruned (zero filled) weights ----
  load_a();
  gemmini_config_ld(NM_J * sizeof(elem_t));
  for (int kt = 0; kt < KT; kt++)
    for (int jt = 0; jt < JT; jt++)
      gemmini_mvin(&w[kt * DIM][jt * DIM], W_SP(kt, jt));
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  poison_c(c_dense);

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
  // leave half an index tile behind, then reset: the run below only passes if the reset dropped it
  for (int i = 0; i < (nm_meta_cmds(NM_M) + 1) / 2; i++)
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ~(uint64_t)0, ~(uint64_t)0, k_NM_META);
  gemmini_nm_reset();
  poison_c(c_sparse);

  // meta(0), then for each tile preload(q), meta(q+1), compute(q): at most 2 tiles queued
  const int ntiles = IT * JT * KCT;
  uint64_t s_start = read_cycles();
  nm_send_meta(meta[0][0], NM_M);
  for (int q = 0; q < ntiles; q++) {
    const int it = q / (JT * KCT), jt = (q / KCT) % JT, t = q % KCT;
    uint32_t c = ACC_BIT | (t > 0 ? ACCUM_BIT : 0) | C_ACC(it, jt);
    gemmini_preload(W_SP(t, jt), c);
    if (q + 1 < ntiles) {
      const int jn = ((q + 1) / KCT) % JT, tn = (q + 1) % KCT;
      nm_send_meta(meta[tn][jn], NM_M);
    }
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
  printf("sparse cycles: %llu (%d preload/compute pairs, %d meta cmds each)\n", (unsigned long long)(s_end - s_start), IT * JT * KCT, nm_meta_cmds(NM_M));

  if (fails) {
    printf("nm_sparse_matmul FAILED\n");
    exit(1);
  }
  printf("nm_sparse_matmul passed\n");
  exit(0);
}
