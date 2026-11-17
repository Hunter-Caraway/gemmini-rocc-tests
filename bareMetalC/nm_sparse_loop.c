// changed: new test (sparse-loop-fix). n:m sparse mode inside the LOOP_WS hardware loop: funct 27 LOAD_IDX, funct 28
// LOOP_WS_CONFIG_NM, LOOP_WS rs1[3], tiled_matmul_nm_loop().
// spec: gemmini-sparsity experiments/streaming/worklogs/fix-b-spec.md (2.5 encodings and asserts, 6.3 this test plan)
//
// Default build (exit 0 = all right). every case: full-range int8 data, C and the whole accumulator poisoned first,
// every value of C compared to a CPU reference (bias + A * pruned W).
//  - single loops with T = 1, 2, 3, 4, 5 and 10 index tiles (the first LOAD of 4, tails of 1 and 2), max_i 1 and > 1,
//    max_j 1, > 1 and 5 (more than one block mvin), partial last row tile; no bias, repeating bias, per-row bias
//  - back-to-back loops (both scratchpad halves, accumulation over K blocks, partial row and column tiles, bias)
//  - K not a multiple of 2*DIM through nm_loop_pad_k / nm_zero_pad
//  - N not a multiple of DIM (pad_J != 0: the last column tile's preload has fewer than DIM weight columns), in
//    single loops and in the last column block of several calls
//  - nm_loop_tiles picking the tiles
//  - sparse loop, then Berkeley's dense loop (tiled_matmul_auto), then two per-tile sparse runs whose index rows the
//    CPU moves in with LOAD_IDX (all three config_ld strides set to something else), then a sparse loop again
//  - bank-split-fix (bank-split-fix-spec.md 2.1, 4): loops whose E + B regions (even A K tiles + weights) fill or pass
//    the first bank of the loop's half: B ends exactly at the bank boundary (O starts on it), B straddles the boundary
//    (O above B), and two straddling loops back to back (both halves). Smaller cases leave a gap below O.
//
// -DNEG=<n>: one illegal use, n / 10 = the assert N<n/10> of spec 2.5.6 that Spike must abort on and the RTL must
// assert on with that message (exit 0 = NOT caught):
//   10 N1 LOOP_WS without rs1[3] in sparse mode       11 N1 LOOP_WS with rs1[3] in dense mode
//   20 N2 LOOP_CONV_WS in sparse mode                 30 N3 rs1[3] with no funct 28 before
//   40 N4 funct 28, then LOOP_WS without rs1[3] (dense mode)
//   50 N5 funct 28 rs1 == 0    51 N5 rs1 not 16 byte aligned    52 N5 rs2 != 0
//   60 N6 odd K tile count     61 N6 pad_K != 0
//   70 N7 A_transpose   71 N7 B_transpose   72 N7 is_resadd   73 N7 low_D   74 N7 spad-only (rs2[9])
//   80 N8 A_stride == 2
//   90 N9 A + B + index rows over the scratchpad half     91 N9 more than one bank of index rows
//   92 N9 (bank-split-fix BS1) even A K tiles alone over one bank (A over the half)
//   100 N10 a register index tile pending when the loop starts
//   110 N11 funct 27 on a dense build. dense Spike lib (libgemmini NM_DENSE_BUILD=1) only: legal on an n:m build,
//       and the dense RTL has no funct 27 / 28 hardware and no assert (spec changelog 2026-11-13, D1)
//   120 N12 cols == 8    121 N12 rows == 0    122 N12 rows == 17
//   130 N13 accumulator address    131 N13 GARBAGE_ADDR    132 N13 row 16384 (past the scratchpad)
//   133 N13 starts inside the scratchpad, 1..16 rows, the last row one past its end (ending on the last row is legal:
//       "seq: per-tile, LOAD_IDX groups" does it)
//   140 N14 funct 28 on a dense build (dense Spike lib only, as 110)
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "include/gemmini_testutils.h"
#include "include/gemmini_nm.h"

#define MAXM (3 * DIM)
#define MAXN (6 * DIM)
#define MAXK (8 * DIM)
#define MAXKC (MAXK / 2)
#define MAXT ((MAXKC / DIM) * (MAXN / DIM))
#define R NM_SPAD_ROWS_PER_TILE
#define TB NM_META_SPAD_BYTES
#define ACC_BIT ((uint32_t)1 << (ADDR_LEN - 1))
#define ACCUM_BIT ((uint32_t)1 << (ADDR_LEN - 2))
#define FULL_BIT ((uint32_t)1 << (ADDR_LEN - 3))
#define POISON ((acc_t)0x5A5A5A5A)

#if BANK_NUM != 4
#error "layout does not fit this gemmini config"
#endif

// every row stride is the buffer width, so strides differ from the matrix widths in most cases
static elem_t a[MAXM][MAXK] row_align(1);
static elem_t w[MAXK][MAXN] row_align(1);
static elem_t wc[MAXKC][MAXN] row_align(1);
static nm_idx_t idx[MAXKC][MAXN];
static uint8_t meta_tj[MAXT][TB] row_align(1);   // scratchpad format, tile (t, jt) at t * NT + jt
static uint8_t meta_loop[MAXT][TB] row_align(1); // the same tiles in nm_loop_order_idx() order
static acc_t bias[MAXM][MAXN] row_align_acc(1);
static acc_t gold[MAXM][MAXN];
static acc_t c_out[MAXM][MAXN] row_align_acc(1);
static acc_t poison_tile[DIM][DIM] row_align_acc(1);
// K padding case: the unpadded operands
static elem_t a_raw[MAXM][MAXK];
static elem_t w_raw[MAXK][MAXN];

enum { NO_BIAS, REPEATING_BIAS, ROW_BIAS };

static unsigned rs = 20261112;
static int rnd(int lo, int hi) { rs = rs * 1103515245u + 12345u; return lo + (int)((rs >> 8) % (unsigned)(hi - lo + 1)); }

// ---- data ----
// gold = bias + a_src (M x K) * w_src (K x N, pruned here); a / w = the operands zero padded to Kp; wc, idx, meta_tj
static void make_case(size_t M, size_t N, size_t K, size_t Kp, int bias_mode) {
  NM_REQUIRE(M <= MAXM && N <= MAXN && Kp <= MAXK && K <= Kp && Kp % (2 * DIM) == 0);
  for (size_t y = 0; y < M; y++) for (size_t z = 0; z < K; z++) a_raw[y][z] = rnd(-128, 127);
  for (size_t z = 0; z < K; z++) for (size_t x = 0; x < N; x++) w_raw[z][x] = rnd(-128, 127);
  nm_prune(K, N, &w_raw[0][0], MAXN, NM_N, NM_M);
  NM_REQUIRE(nm_check(K, N, &w_raw[0][0], MAXN, NM_N, NM_M));
  for (size_t y = 0; y < M; y++) for (size_t x = 0; x < N; x++) bias[y][x] = bias_mode == NO_BIAS ? 0 : rnd(-1000000, 1000000);
  nm_ref_dense(M, K, N, &a_raw[0][0], MAXK, &w_raw[0][0], MAXN, &gold[0][0], MAXN);
  for (size_t y = 0; y < M; y++) for (size_t x = 0; x < N; x++) gold[y][x] += bias[bias_mode == REPEATING_BIAS ? 0 : y][x];
  nm_zero_pad(M, K, &a_raw[0][0], MAXK, M, Kp, &a[0][0], MAXK);
  nm_zero_pad(K, N, &w_raw[0][0], MAXN, Kp, N, &w[0][0], MAXN);
  nm_compress(Kp, N, &w[0][0], MAXN, &wc[0][0], &idx[0][0], MAXN, NM_N, NM_M);
  const size_t NT = (N + DIM - 1) / DIM, KBT = Kp / (2 * DIM);
  for (size_t t = 0; t < KBT; t++)
    for (size_t jt = 0; jt < NT; jt++)
      nm_pack_meta_spad(&idx[t * DIM][jt * DIM], MAXN, DIM, N - jt * DIM < DIM ? N - jt * DIM : DIM, NM_M, meta_tj[t * NT + jt]);
}

// ---- gemmini helpers ----
// C = POISON, and every accumulator row = POISON (a loop that writes nothing, or accumulates on stale rows, fails)
static void poison(void) {
  for (int y = 0; y < MAXM; y++) for (int x = 0; x < MAXN; x++) c_out[y][x] = POISON;
  gemmini_config_ld(DIM * sizeof(acc_t));
  for (uint32_t r = 0; r < ACC_ROWS; r += DIM) gemmini_mvin(&poison_tile[0][0], ACC_BIT | r);
  gemmini_fence();
}
static int check(const char * name, size_t M, size_t N) {
  int bad = 0;
  for (size_t y = 0; y < M; y++) for (size_t x = 0; x < N; x++)
    if (c_out[y][x] != gold[y][x]) { if (!bad) printf("%s: mismatch at (%d,%d): got %d want %d\n", name, (int)y, (int)x, c_out[y][x], gold[y][x]); bad++; }
  // nothing may be written outside the M x N result
  for (size_t y = 0; y < MAXM; y++) for (size_t x = 0; x < MAXN; x++)
    if ((y >= M || x >= N) && c_out[y][x] != POISON) { if (!bad) printf("%s: wrote outside C at (%d,%d)\n", name, (int)y, (int)x); bad++; }
  printf("%-34s %s (%d/%d wrong)\n", name, bad ? "FAIL" : "ok", bad, (int)(M * N));
  return bad != 0;
}

// one sparse-loop case; tile_I == 0: nm_loop_tiles picks the tiles
static int loop_case(const char * name, size_t M, size_t N, size_t K, size_t tile_I, size_t tile_J, size_t tile_K, int bias_mode) {
  const size_t Kp = nm_loop_pad_k(K);
  make_case(M, N, K, Kp, bias_mode);
  if (tile_I == 0) nm_loop_tiles(M, N, Kp, &tile_I, &tile_J, &tile_K);
  nm_loop_order_idx(N, Kp, tile_J, tile_K, meta_tj[0], (N + DIM - 1) / DIM, meta_loop[0]);
  poison();
  tiled_matmul_nm_loop(M, N, Kp, &a[0][0], MAXK, &wc[0][0], MAXN, bias_mode == NO_BIAS ? NULL : &bias[0][0], MAXN,
                       bias_mode == REPEATING_BIAS, meta_loop[0], &c_out[0][0], MAXN, tile_I, tile_J, tile_K);
  return check(name, M, N);
}
// the current case (no bias, K already padded) again, data untouched
static int loop_again(const char * name, size_t M, size_t N, size_t K, size_t tile_I, size_t tile_J, size_t tile_K) {
  nm_loop_order_idx(N, K, tile_J, tile_K, meta_tj[0], (N + DIM - 1) / DIM, meta_loop[0]);
  poison();
  tiled_matmul_nm_loop(M, N, K, &a[0][0], MAXK, &wc[0][0], MAXN, NULL, MAXN, false, meta_loop[0], &c_out[0][0], MAXN,
                       tile_I, tile_J, tile_K);
  return check(name, M, N);
}

// per-tile sparse run of the current case (whole tiles only) with the index rows moved in by LOAD_IDX from the CPU.
// A in Berkeley's loop layout from row 0 (a_hi = a_lo + DIM), compressed W in bank 2, index tiles ending at idx_end.
// one_by_one: a LOAD_IDX and a load per tile; else LOAD_IDX of whole groups and loads of up to 4 tiles
static int per_tile_case(const char * name, size_t M, size_t N, size_t K, uint32_t idx_end, int one_by_one) {
  const size_t IT = M / DIM, JT = N / DIM, KT = K / DIM, KBT = KT / 2, T = KBT * JT;
  const uint32_t idx_row = idx_end - (uint32_t)(T * R);
  NM_REQUIRE(M % DIM == 0 && N % DIM == 0 && IT * KT * DIM <= BANK_ROWS && T * DIM <= BANK_ROWS && IT * JT * DIM <= ACC_ROWS);
#define PT_A(it, kt) ((uint32_t)(((it) * KT + (kt)) * DIM))
#define PT_W(t, jt) ((uint32_t)(2 * BANK_ROWS + ((t) * JT + (jt)) * DIM))
#define PT_C(it, jt) ((uint32_t)(((it) * JT + (jt)) * DIM))
  gemmini_nm_reset();
  poison();
  gemmini_nm_config_ex(1, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, 1);
  gemmini_extended3_config_ld(MAXK * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(MAXN * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
  gemmini_extended3_config_ld(3 * DIM * sizeof(acc_t), MVIN_SCALE_IDENTITY, false, 2); // no load uses it: LOAD_IDX must not either
  for (size_t it = 0; it < IT; it++) for (size_t kt = 0; kt < KT; kt++) gemmini_extended_mvin(&a[it * DIM][kt * DIM], PT_A(it, kt), DIM, DIM);
  for (size_t t = 0; t < KBT; t++) for (size_t jt = 0; jt < JT; jt++) gemmini_extended_mvin2(&wc[t * DIM][jt * DIM], PT_W(t, jt), DIM, DIM);
  if (!one_by_one) nm_load_idx_meta_spad(meta_tj[0], idx_row, T);
  for (size_t u = 0; u < T; u++) {
    const size_t t = u / JT, jt = u % JT;
    if (one_by_one) {
      gemmini_nm_load_idx(meta_tj[u], idx_row + (uint32_t)(u * R), R);
      gemmini_nm_load_spad(idx_row + (uint32_t)(u * R), 1);
    } else if (u % NM_META_FIFO_DEPTH == 0) {
      gemmini_nm_load_spad(idx_row + (uint32_t)(u * R), T - u < NM_META_FIFO_DEPTH ? T - u : NM_META_FIFO_DEPTH);
    }
    for (size_t it = 0; it < IT; it++) {
      const uint32_t c = ACC_BIT | (t > 0 ? ACCUM_BIT : 0) | PT_C(it, jt);
      if (it == 0) { gemmini_preload(PT_W(t, jt), c); gemmini_compute_preloaded(PT_A(it, 2 * t), PT_A(it, 2 * t + 1)); }
      else { gemmini_preload(GARBAGE_ADDR, c); gemmini_compute_accumulated(PT_A(it, 2 * t), PT_A(it, 2 * t + 1)); }
    }
  }
  gemmini_config_st(MAXN * sizeof(acc_t));
  for (size_t it = 0; it < IT; it++) for (size_t jt = 0; jt < JT; jt++)
    gemmini_mvout(&c_out[it * DIM][jt * DIM], ACC_BIT | FULL_BIT | PT_C(it, jt));
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_fence();
#undef PT_A
#undef PT_W
#undef PT_C
  return check(name, M, N);
}

#ifndef NEG
// ---- bank-split-fix layout cases: own buffers, K up to BK columns, one row tile, up to two column tiles ----
#define BM DIM
#define BN (2 * DIM)
#define BK (384 * DIM)
#define BT ((BK / (2 * DIM)) * (BN / DIM))
static elem_t ba[BM][BK] row_align(1);
static elem_t bw[BK][BN] row_align(1);
static elem_t bwc[BK / 2][BN] row_align(1);
static nm_idx_t bidx[BK / 2][BN];
static uint8_t bmeta_tj[BT][TB] row_align(1);
static uint8_t bmeta_loop[BT][TB] row_align(1);
static acc_t bgold[BM][BN];
static acc_t bc_out[BM][BN] row_align_acc(1);

// sparse loops of 1 x NT x tile_K tiles over an M x N x K problem (K % (2*DIM) == 0, no bias)
static int bank_case(const char * name, size_t M, size_t N, size_t K, size_t tile_K) {
  NM_REQUIRE(M <= BM && N <= BN && K <= BK && K % (2 * DIM) == 0 && tile_K % 2 == 0);
  const size_t NT = (N + DIM - 1) / DIM, KBT = K / (2 * DIM);
  for (size_t y = 0; y < M; y++) for (size_t z = 0; z < K; z++) ba[y][z] = rnd(-128, 127);
  for (size_t z = 0; z < K; z++) for (size_t x = 0; x < N; x++) bw[z][x] = rnd(-128, 127);
  nm_prune(K, N, &bw[0][0], BN, NM_N, NM_M);
  NM_REQUIRE(nm_check(K, N, &bw[0][0], BN, NM_N, NM_M));
  nm_ref_dense(M, K, N, &ba[0][0], BK, &bw[0][0], BN, &bgold[0][0], BN);
  nm_compress(K, N, &bw[0][0], BN, &bwc[0][0], &bidx[0][0], BN, NM_N, NM_M);
  for (size_t t = 0; t < KBT; t++)
    for (size_t jt = 0; jt < NT; jt++)
      nm_pack_meta_spad(&bidx[t * DIM][jt * DIM], BN, DIM, N - jt * DIM < DIM ? N - jt * DIM : DIM, NM_M, bmeta_tj[t * NT + jt]);
  nm_loop_order_idx(N, K, NT, tile_K, bmeta_tj[0], NT, bmeta_loop[0]);
  poison();
  for (int y = 0; y < BM; y++) for (int x = 0; x < BN; x++) bc_out[y][x] = POISON;
  tiled_matmul_nm_loop(M, N, K, &ba[0][0], BK, &bwc[0][0], BN, NULL, BN, false, bmeta_loop[0], &bc_out[0][0], BN, 1, NT, tile_K);
  int bad = 0;
  for (size_t y = 0; y < BM; y++) for (size_t x = 0; x < BN; x++) {
    const bool in = y < M && x < N;
    if (in ? bc_out[y][x] != bgold[y][x] : bc_out[y][x] != POISON) {
      if (!bad) printf("%s: %s at (%d,%d): got %d\n", name, in ? "mismatch" : "wrote outside C", (int)y, (int)x, bc_out[y][x]);
      bad++;
    }
  }
  printf("%-34s %s (%d/%d wrong)\n", name, bad ? "FAIL" : "ok", bad, (int)(M * N));
  return bad != 0;
}
#endif

#ifdef NEG
// the loop config commands one by one, so a NEG case can leave out or bend any of them. the operands are the valid
// 1 x 1 x 2 tile case made in main(); ws_rs1 0xA = rs1[3] (sparse loop) + full_C
static void raw_loop(uint64_t I, uint64_t J, uint64_t K, uint64_t pad_K, int cfg_nm, uint64_t cfg_rs1, uint64_t cfg_rs2,
                     uint64_t ws_rs1, uint64_t ws_rs2) {
  gemmini_extended_config_st(MAXN * sizeof(acc_t), NO_ACTIVATION, ACC_SCALE_IDENTITY);
  gemmini_extended3_config_ld(MAXK * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(MAXN * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
  gemmini_extended3_config_ld(MAXN * sizeof(acc_t), MVIN_SCALE_IDENTITY, false, 2);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, pad_K << 32, (K << 32) | (J << 16) | I, k_LOOP_WS_CONFIG_BOUNDS);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, &a[0][0], &wc[0][0], k_LOOP_WS_CONFIG_ADDRS_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, &c_out[0][0], k_LOOP_WS_CONFIG_ADDRS_DC);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, MAXK, MAXN, k_LOOP_WS_CONFIG_STRIDES_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, MAXN, MAXN, k_LOOP_WS_CONFIG_STRIDES_DC);
  if (cfg_nm) { ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, cfg_rs1, cfg_rs2, k_LOOP_WS_CONFIG_NM); }
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ws_rs1, ws_rs2, k_LOOP_WS);
}
#define IDX0 ((uint64_t)(uintptr_t)meta_loop[0])
#define WS_NM 0xAu
#define WS_DENSE 0x2u
static void sparse_on(int a_stride) { gemmini_nm_reset(); gemmini_nm_config_ex(1, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, a_stride); }
static void dense_on(void) { gemmini_nm_reset(); gemmini_nm_config_ex(0, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, 1); }
#endif

int main(void) {
  for (int y = 0; y < DIM; y++) for (int x = 0; x < DIM; x++) poison_tile[y][x] = POISON;
  gemmini_flush(0);

#ifndef NEG
  int fails = 0;
  printf("nm_sparse_loop: %d:%d, R = %d rows per index tile, %d tiles per LOAD_IDX\n", NM_N, NM_M, (int)R, (int)NM_LOAD_IDX_TILES);

  // single loops:        name                              M        N        K        tiles      bias
  fails += loop_case("T=1 i1 j1",                         DIM,     DIM,     2 * DIM, 1, 1, 2, NO_BIAS);
  fails += loop_case("T=2 i2 j2 repeating bias",          2 * DIM, 2 * DIM, 2 * DIM, 2, 2, 2, REPEATING_BIAS);
  fails += loop_case("T=3 i1 j3 row bias",                DIM,     3 * DIM, 2 * DIM, 1, 3, 2, ROW_BIAS);
  fails += loop_case("T=4 i2 j2",                         2 * DIM, 2 * DIM, 4 * DIM, 2, 2, 4, NO_BIAS);
  fails += loop_case("T=5 i2 j5",                         2 * DIM, 5 * DIM, 2 * DIM, 2, 5, 2, NO_BIAS);
  fails += loop_case("T=10 i3 j5 partial i, row bias",    2 * DIM + 8, 5 * DIM, 4 * DIM, 3, 5, 4, ROW_BIAS);
  // 2 x 2 x 2 = 8 loops back to back: partial last row and column tiles, accumulation over the K blocks
  fails += loop_case("8 loops, partial i and j, rep. bias", 2 * DIM + 4, 3 * DIM + 8, 8 * DIM, 2, 2, 4, REPEATING_BIAS);
  fails += loop_case("8 loops, no bias",                  3 * DIM, 4 * DIM, 8 * DIM, 2, 2, 4, NO_BIAS);
  // 2 x 2 loops with 3 and 1 index tiles (K blocks of 2 tiles; the last column block is 1 tile wide)
  fails += loop_case("4 loops, T=3 and T=1, row bias",    DIM, 4 * DIM, 4 * DIM, 1, 3, 2, ROW_BIAS);
  // K = 2.5 tiles, zero padded to 4
  fails += loop_case("K=40 padded to 64",                 DIM, 2 * DIM, 2 * DIM + 8, 1, 2, 4, NO_BIAS);
  fails += loop_case("nm_loop_tiles, row bias",           3 * DIM, 6 * DIM, 6 * DIM, 0, 0, 0, ROW_BIAS);
  // pad_J: N % DIM != 0. one loop each, the partial column tile alone, last of 2 and last of 5
  fails += loop_case("pad_J: N=5, T=1",                   DIM,     5,       2 * DIM, 1, 1, 2, NO_BIAS);
  fails += loop_case("pad_J: N=DIM+1, T=4, rep. bias",    2 * DIM, DIM + 1, 4 * DIM, 2, 2, 4, REPEATING_BIAS);
  fails += loop_case("pad_J: N=4*DIM+13, partial i, row bias", DIM + 3, 4 * DIM + 13, 2 * DIM, 2, 5, 2, ROW_BIAS);
  // 2 x 2 loops, the last column block is 2 tiles wide with 1 column missing
  fails += loop_case("pad_J: N=5*DIM-1, 4 loops",         2 * DIM, 5 * DIM - 1, 4 * DIM, 2, 3, 2, NO_BIAS);
  fails += loop_case("pad_J: N=5*DIM+7, nm_loop_tiles, K=40", 2 * DIM + 9, 5 * DIM + 7, 2 * DIM + 8, 0, 0, 0, ROW_BIAS);

  // mode changes: sparse loop, Berkeley's dense loop, per-tile sparse with CPU LOAD_IDX, sparse loop again
  {
    const size_t M = 2 * DIM, N = 2 * DIM, K = 4 * DIM;
    fails += loop_case("seq: sparse loop", M, N, K, 2, 2, 4, NO_BIAS);
    poison();
    tiled_matmul_auto(M, N, K, &a[0][0], &w[0][0], NULL, &c_out[0][0], MAXK, MAXN, MAXN, MAXN,
                      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
                      false, false, false, true, false, 0, WS);
    fails += check("seq: dense loop (pruned W)", M, N);
    fails += per_tile_case("seq: per-tile, LOAD_IDX groups", M, N, K, 4 * BANK_ROWS, 0);
    fails += loop_again("seq: sparse loop again", M, N, K, 2, 2, 4);
    fails += per_tile_case("seq: per-tile, LOAD_IDX by tile", M, N, K, BANK_ROWS + (uint32_t)(MAXT * R), 1);
    fails += loop_again("seq: sparse loop, 2 calls", M, N, K, 2, 2, 2);
  }

  // bank-split-fix layouts (rows from the loop's A start, S = BANK_ROWS = 4096; ea = E rows, bsz = B rows):
  // the layout depends on tile counts only, so few rows of C keep the CPU reference (and RTL run time) small
  //   kb 128, J 1: ea 2048, bsz 2048, B ends at S, O = [S, S + ea)
  fails += bank_case("bank: B ends at the bank boundary", 2, DIM, 256 * DIM, 256);
  //   kb 96, J 2: ea 1536, bsz 3072, B straddles S, O = [4608, 6144); partial column tile
  fails += bank_case("bank: B straddles, pad_I and pad_J", 3, 2 * DIM - 5, 192 * DIM, 192);
  //   two such loops, the second in the other half, accumulating on the first
  fails += bank_case("bank: B straddles, 2 loops", 1, 2 * DIM, 384 * DIM, 192);

  if (fails) { printf("nm_sparse_loop: %d case(s) FAILED\n", fails); exit(1); }
  printf("nm_sparse_loop passed\n");
  exit(0);

#else // NEG: an illegal use; reaching the end means the check did not fire
  make_case(DIM, DIM, 2 * DIM, 2 * DIM, NO_BIAS);
  nm_loop_order_idx(DIM, 2 * DIM, 1, 2, meta_tj[0], 1, meta_loop[0]);
#if NEG == 10
  sparse_on(1); raw_loop(1, 1, 2, 0, 0, 0, 0, WS_DENSE, 0);
#elif NEG == 11
  dense_on(); raw_loop(1, 1, 2, 0, 1, IDX0, 0, WS_NM, 0);
#elif NEG == 20
  sparse_on(1); ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, 0, k_LOOP_CONV_WS);
#elif NEG == 30
  sparse_on(1); raw_loop(1, 1, 2, 0, 0, 0, 0, WS_NM, 0);
#elif NEG == 40
  dense_on(); raw_loop(1, 1, 2, 0, 1, IDX0, 0, WS_DENSE, 0);
#elif NEG == 50
  sparse_on(1); ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, 0, k_LOOP_WS_CONFIG_NM);
#elif NEG == 51
  sparse_on(1); ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, IDX0 + 8, 0, k_LOOP_WS_CONFIG_NM);
#elif NEG == 52
  sparse_on(1); ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, IDX0, 1, k_LOOP_WS_CONFIG_NM);
#elif NEG == 60
  sparse_on(1); raw_loop(1, 1, 3, 0, 1, IDX0, 0, WS_NM, 0);
#elif NEG == 61
  sparse_on(1); raw_loop(1, 1, 2, 8, 1, IDX0, 0, WS_NM, 0);
#elif NEG == 70
  sparse_on(1); raw_loop(1, 1, 2, 0, 1, IDX0, 0, WS_NM, 1);
#elif NEG == 71
  sparse_on(1); raw_loop(1, 1, 2, 0, 1, IDX0, 0, WS_NM, 2);
#elif NEG == 72
  sparse_on(1); raw_loop(1, 1, 2, 0, 1, IDX0, 0, WS_NM, 4);
#elif NEG == 73
  sparse_on(1); raw_loop(1, 1, 2, 0, 1, IDX0, 0, WS_NM | 4, 0);
#elif NEG == 74
  sparse_on(1); raw_loop(1, 1, 2, 0, 1, IDX0, 0, WS_NM, 0x200);
#elif NEG == 80
  sparse_on(2); raw_loop(1, 1, 2, 0, 1, IDX0, 0, WS_NM, 0);
#elif NEG == 90
  // A 8*32*DIM + B 16*16*DIM = 8192 rows, plus the index rows
  sparse_on(1); raw_loop(8, 16, 32, 0, 1, IDX0, 0, WS_NM, 0);
#elif NEG == 91
  // 32 * 40 index tiles: over one bank of index rows for 2:4 and 4:8 (B alone is over the half too)
  sparse_on(1); raw_loop(1, 40, 64, 0, 1, IDX0, 0, WS_NM, 0);
#elif NEG == 92
  // E = 17 * 16 * DIM = 4352 rows, over one bank: A = 8704 rows is over the half, so N9 fires (bank-split-fix-spec.md 3.2 BS1)
  sparse_on(1); raw_loop(17, 1, 32, 0, 1, IDX0, 0, WS_NM, 0);
#elif NEG == 100
  sparse_on(1);
  { uint64_t reg[2 * DIM]; nm_pack_meta(&idx[0][0], MAXN, DIM, DIM, NM_M, reg); nm_send_meta(reg, NM_M); }
  raw_loop(1, 1, 2, 0, 1, IDX0, 0, WS_NM, 0);
#elif NEG == 110
  // no sparse-mode command before it: a dense build must stop here, not earlier
  gemmini_nm_load_idx(meta_loop[0], 0, R);
#elif NEG == 120
  sparse_on(1); gemmini_nm_extended_load_idx(meta_loop[0], 0, 8, R);
#elif NEG == 121
  sparse_on(1); gemmini_nm_extended_load_idx(meta_loop[0], 0, NM_LOAD_IDX_COLS, 0);
#elif NEG == 122
  sparse_on(1); gemmini_nm_extended_load_idx(meta_loop[0], 0, NM_LOAD_IDX_COLS, 17);
#elif NEG == 130
  sparse_on(1); gemmini_nm_load_idx(meta_loop[0], ACC_BIT | 0, R);
#elif NEG == 131
  sparse_on(1); gemmini_nm_load_idx(meta_loop[0], GARBAGE_ADDR, R);
#elif NEG == 132
  sparse_on(1); gemmini_nm_load_idx(meta_loop[0], (uint32_t)(BANK_NUM * BANK_ROWS), R);
#elif NEG == 133
  sparse_on(1); gemmini_nm_load_idx(meta_loop[0], (uint32_t)(BANK_NUM * BANK_ROWS - R + 1), R);
#elif NEG == 140
  // as 110
  gemmini_loop_ws_config_nm(meta_loop[0]);
#else
#error "unknown NEG case"
#endif
  gemmini_fence();
  printf("nm_sparse_loop NEG=%d: illegal use NOT caught\n", NEG);
  exit(0);
#endif
}
