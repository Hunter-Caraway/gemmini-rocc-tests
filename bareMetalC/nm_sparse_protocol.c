// changed: n:m sparse protocol test. Small directed cases for the rules in gemmini_nm.h that the bigger tests
// never exercise: a_hi rows are 1 apart whatever a_stride is, weights are latched at the preload, a GARBAGE
// preload does not pop an index tile, the clear command, 4 tiles queued ahead, dense after sparse.
//
// Default build: every legal case, each checked against a CPU reference (exit 0 = all right).
// -DNEG=<n>: one illegal use that Spike must abort on and the RTL must assert on (exit 0 = NOT caught):
//   4  a 5th index tile between a preload and its compute (FIFO depth 4; the RTL deadlocks, then asserts)
//   6  sparse compute with a GARBAGE A operand (Spike only; the RTL has no check)
//   7  sparse compute on weights preloaded in dense mode
//   9  sparse mode with the OS dataflow
//   10 LOOP_WS while sparse
//   14 GARBAGE preload in sparse mode onto a buffer last preloaded in dense mode
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "include/gemmini.h"
#include "include/gemmini_nm.h"

#define A_LO 0
#define A_HI (2 * DIM)
#define W1 (4 * DIM)
#define W2 (5 * DIM)
#define C_SP (6 * DIM)
#define A2_LO (7 * DIM)
#define A2_HI (9 * DIM)

static elem_t alo[2 * DIM][DIM], ahi[DIM][DIM], alo2[2 * DIM][DIM], ahi2[DIM][DIM];
static elem_t w1[DIM][DIM], w2[DIM][DIM], out[DIM][DIM];
static nm_idx_t idx[DIM][DIM], idx_inv[DIM][DIM];
static unsigned rs = 12345;
static int rnd(int lo, int hi) { rs = rs * 1103515245u + 12345u; return lo + (int)((rs >> 16) % (unsigned)(hi - lo + 1)); }

static void fill(void) {
  for (int i = 0; i < 2 * DIM; i++) for (int j = 0; j < DIM; j++) { alo[i][j] = rnd(-2, 2); alo2[i][j] = rnd(-2, 2); }
  for (int i = 0; i < DIM; i++) for (int j = 0; j < DIM; j++) {
    ahi[i][j] = rnd(-2, 2); ahi2[i][j] = rnd(-2, 2); w1[i][j] = rnd(-2, 2); w2[i][j] = rnd(-2, 2);
    idx[i][j] = rnd(0, NM_M - 1); idx_inv[i][j] = NM_M - 1 - idx[i][j];
  }
}

// out == sparse product of (lo rows s apart, hi rows 1 apart) and w with indexes idx
static int check(elem_t lo[][DIM], elem_t hi[][DIM], elem_t w[][DIM], int s, const char * what) {
  int bad = 0;
  for (int i = 0; i < DIM; i++) for (int j = 0; j < DIM; j++) {
    int sum = 0;
    for (int r = 0; r < DIM; r++) {
      int k = (r / NM_N) * NM_M + idx[r][j];
      sum += w[r][j] * (k < DIM ? lo[s * i][k] : hi[i][k - DIM]);
    }
    if (out[i][j] != sum) { if (!bad) printf("%s: mismatch at (%d,%d): got %d want %d\n", what, i, j, out[i][j], sum); bad++; }
  }
  printf("%-22s %s (%d/%d wrong)\n", what, bad ? "FAIL" : "ok", bad, DIM * DIM);
  return bad != 0;
}

static void poison(void) {
  for (int i = 0; i < DIM; i++) for (int j = 0; j < DIM; j++) out[i][j] = 0x55;
  gemmini_mvin(out, C_SP); gemmini_fence();
}
static void mvout_c(void) { gemmini_mvout(out, C_SP); gemmini_fence(); }
static void sparse(int on, int s) { gemmini_nm_config_ex(on, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, s); }
static void meta(nm_idx_t m[][DIM]) { nm_load_meta(&m[0][0], DIM, DIM, DIM, NM_M); }
// start of every case: fence, empty index FIFO, C poisoned, sparse on with A stride s, first preload real
static void begin(int s) { gemmini_nm_reset(); poison(); sparse(1, s); }

int main(void) {
  fill();
  gemmini_flush(0);
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));
  gemmini_mvin(alo, A_LO); gemmini_mvin(alo[DIM], A_LO + DIM); gemmini_mvin(ahi, A_HI);
  gemmini_mvin(alo2, A2_LO); gemmini_mvin(alo2[DIM], A2_LO + DIM); gemmini_mvin(ahi2, A2_HI);
  gemmini_mvin(w1, W1); gemmini_mvin(w2, W2);
  gemmini_fence();

#ifndef NEG
  int fails = 0;

  begin(1); meta(idx);
  gemmini_preload(W1, C_SP); gemmini_compute_preloaded(A_LO, A_HI); mvout_c();
  fails += check(alo, ahi, w1, 1, "basic");

  // a_lo rows 2 apart, a_hi rows still 1 apart
  begin(2); meta(idx);
  gemmini_preload(W1, C_SP); gemmini_compute_preloaded(A_LO, A_HI); mvout_c();
  fails += check(alo, ahi, w1, 2, "a_stride 2");

  // weights latched at the preload: a mvin over W1 between preload and compute must not change the result
  begin(1); meta(idx);
  gemmini_preload(W1, C_SP); gemmini_mvin(w2, W1); gemmini_compute_preloaded(A_LO, A_HI); mvout_c();
  gemmini_mvin(w1, W1); gemmini_fence();
  fails += check(alo, ahi, w1, 1, "weights at preload");

  // a whole junk tile plus half a junk tile, then the clear: only the real tile after it is used
  begin(1); meta(idx_inv);
  { uint64_t junk[2 * DIM]; nm_pack_meta(&idx_inv[0][0], DIM, DIM, DIM, NM_M, junk);
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, junk[0], junk[1], k_NM_META); }
  gemmini_nm_reset(); meta(idx);
  gemmini_preload(W1, C_SP); gemmini_compute_preloaded(A_LO, A_HI); mvout_c();
  fails += check(alo, ahi, w1, 1, "clear");

  // 4 tiles queued ahead (the limit), the 5th sent after the first compute
  begin(1);
  for (int t = 0; t < 4; t++) meta(idx);
  gemmini_preload(W1, C_SP); gemmini_compute_preloaded(A_LO, A_HI);
  meta(idx);
  for (int t = 0; t < 4; t++) { gemmini_preload(W1, C_SP); gemmini_compute_preloaded(A_LO, A_HI); }
  mvout_c();
  fails += check(alo, ahi, w1, 1, "4 tiles ahead");

  // compute_accumulated with a GARBAGE preload keeps the weights and indexes (C overwritten, not accumulated)
  begin(1); meta(idx);
  gemmini_preload(W1, C_SP); gemmini_compute_preloaded(A_LO, A_HI);
  gemmini_preload(GARBAGE_ADDR, C_SP); gemmini_compute_accumulated(A2_LO, A2_HI); mvout_c();
  fails += check(alo2, ahi2, w1, 1, "garbage keeps weights");

  // a GARBAGE preload does not pop: the 2nd real preload gets the 2nd index tile
  begin(1); meta(idx_inv); meta(idx);
  gemmini_preload(W2, C_SP); gemmini_compute_preloaded(A_LO, A_HI);
  gemmini_preload(GARBAGE_ADDR, C_SP); gemmini_compute_accumulated(A_LO, A_HI);
  gemmini_preload(W1, C_SP); gemmini_compute_preloaded(A2_LO, A2_HI); mvout_c();
  fails += check(alo2, ahi2, w1, 1, "garbage does not pop");

  // two tiles queued, a_stride 2, each tile with its own indexes
  begin(2); meta(idx_inv); meta(idx);
  gemmini_preload(W2, C_SP); gemmini_compute_preloaded(A_LO, A_HI);
  gemmini_preload(W1, C_SP); gemmini_compute_preloaded(A2_LO, A2_HI); mvout_c();
  fails += check(alo2, ahi2, w1, 2, "two tiles, stride 2");

  // dense preload carries index 0, so a dense run works right after sparse runs
  begin(1); meta(idx);
  gemmini_preload(W1, C_SP); gemmini_compute_preloaded(A_LO, A_HI);
  sparse(0, 1); poison();
  gemmini_preload(W1, C_SP); gemmini_compute_preloaded(A_LO, GARBAGE_ADDR); mvout_c();
  { int bad = 0;
    for (int i = 0; i < DIM; i++) for (int j = 0; j < DIM; j++) {
      int sum = 0; for (int k = 0; k < DIM; k++) sum += alo[i][k] * w1[k][j]; bad += out[i][j] != sum; }
    printf("%-22s %s (%d/%d wrong)\n", "dense after sparse", bad ? "FAIL" : "ok", bad, DIM * DIM); fails += bad != 0; }

  if (fails) { printf("nm_sparse_protocol: %d case(s) FAILED\n", fails); exit(1); }
  printf("nm_sparse_protocol passed\n");
  exit(0);

#else // NEG: an illegal use; reaching the end means the check did not fire
  begin(1);
#if NEG == 4
  for (int t = 0; t < 4; t++) meta(idx);
  gemmini_preload(W1, C_SP);
  meta(idx);
  gemmini_compute_preloaded(A_LO, A_HI);
#elif NEG == 6
  meta(idx);
  gemmini_preload(W1, C_SP); gemmini_compute_preloaded(GARBAGE_ADDR, A_HI);
#elif NEG == 7
  sparse(0, 1); gemmini_preload(W1, C_SP); sparse(1, 1); gemmini_compute_preloaded(A_LO, A_HI);
#elif NEG == 9
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)1 << 10) | (OUTPUT_STATIONARY << 2) | CONFIG_EX, 0, k_CONFIG);
#elif NEG == 10
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, 0, k_LOOP_WS);
#elif NEG == 14
  sparse(0, 1); gemmini_preload(W1, C_SP); gemmini_compute_preloaded(A_LO, GARBAGE_ADDR);
  sparse(1, 1); gemmini_preload(GARBAGE_ADDR, C_SP); gemmini_compute_accumulated(A_LO, A_HI);
#else
#error "unknown NEG case"
#endif
  mvout_c();
  printf("nm_sparse_protocol NEG=%d: illegal use NOT caught\n", NEG);
  exit(0);
#endif
}
