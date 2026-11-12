// changed: new test (Fix A). n:m index tiles loaded from the scratchpad: funct 26 rd == 2, gemmini_nm_load_spad().
// spec: gemini-sparsity experiments/streaming/worklogs/fix-a-spec.md
//
// Default build (exit 0 = all right):
//  - the tight packer against the spec's worked example (section 9; 2:4 also == nm_pack_meta), unpack round trip,
//    edge-tile packing
//  - a 2-row-tile x 2-k-tile x 4-col-tile sparse matmul (8 index tiles, weight reuse through GARBAGE preloads),
//    full-range int8 data, outputs and accumulators poisoned before every run, compared to a CPU reference, with
//    the index tiles delivered as: batches of 4 loaded early (right after the compute of the previous batch's last
//    tile); loads of 1, 3, 2, 2 tiles; one register tile then loads; loads issued in dense mode; tiles ending at
//    the last row of bank 0 and of the last bank; index rows overwritten by a mvin right after their load (the
//    load must see the old rows); and a register-path run after gemmini_nm_reset() (a load is no longer pending)
//  - the spec's command sequence (section 9) with its index pattern at row 12288
//
// -DNEG=<n>: one illegal use that Spike must abort on and the RTL must assert on (exit 0 = NOT caught):
//   21 funct 26 rd == 3 (reserved)          22 funct 26 rd == 31 (reserved)
//   23 load of 0 tiles                       24 load of 5 tiles
//   25 load with rs1[63:32] != 0             26 load with rs2[63:8] != 0
//   27 load from an accumulator address      28 load from GARBAGE_ADDR
//   29 load from row 16384 (past the spad)   30 load across a bank boundary
//   31 load running off the last row         32 load while a register tile is half sent
//   33 load over the fifo budget (4 queued + 1)
//   34 load over the fifo budget (3 queued + 1 popped by a preload whose compute is not issued yet + 1)
//   35 register rows (rd == 0) after a load, no gemmini_nm_reset() in between
//   36 clear between a sparse preload and its compute (index tile from a load)
//   37 sparse preload with no index tile left (the load brought one tile, two preloads)
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "include/gemmini_testutils.h"
#include "include/gemmini_nm.h"

#define NM_I (2 * DIM)
#define NM_K (4 * DIM)
#define NM_J (4 * DIM)
#define IT (NM_I / DIM)
#define KT (NM_K / DIM)
#define JT (NM_J / DIM)
#define KC (NM_K / NM_M * NM_N)
#define KCT (KC / DIM)
#define NTILE (KCT * JT) // index tiles, u = jt * KCT + t in preload order
#define R NM_SPAD_ROWS_PER_TILE
#define TB NM_META_SPAD_BYTES

// spad: A tile (it, kt) in bank kt % 2 (a_lo / a_hi in different banks); compressed W in bank 2 from row 1024;
// index tiles: bank 3 (from its first row, 12288 at 4 x 4096), bank 2 from its first row, the ends of banks 0 and 3
#define A_SP(it, kt) ((uint32_t)(((kt) % 2) * BANK_ROWS + ((it) * (KT / 2) + (kt) / 2) * DIM))
#define WC_SP(t, jt) ((uint32_t)(2 * BANK_ROWS + 1024 + ((t) * JT + (jt)) * DIM))
#define IDX_B3 ((uint32_t)(3 * BANK_ROWS))
#define IDX_B2 ((uint32_t)(2 * BANK_ROWS))
#define IDX_END0 ((uint32_t)(BANK_ROWS - 4 * R))     // 4 tiles ending at the last row of bank 0
#define IDX_END3 ((uint32_t)(4 * BANK_ROWS - 4 * R)) // 4 tiles ending at the last row of the scratchpad
#define ACC_BIT ((uint32_t)1 << (ADDR_LEN - 1))
#define ACCUM_BIT ((uint32_t)1 << (ADDR_LEN - 2))
#define FULL_BIT ((uint32_t)1 << (ADDR_LEN - 3))
#define C_ACC(it, jt) ((uint32_t)(((it) * JT + (jt)) * DIM))
#define POISON ((acc_t)0x5A5A5A5A)

#if BANK_NUM != 4 || IT * JT * DIM > ACC_ROWS
#error "layout does not fit this gemmini config"
#endif
_Static_assert(A_SP(IT - 1, KT - 1) + DIM <= 2 * BANK_ROWS && WC_SP(KCT - 1, JT - 1) + DIM <= 3 * BANK_ROWS &&
               IDX_B3 + 68 * R <= 4 * BANK_ROWS && IDX_B2 + NTILE * R <= WC_SP(0, 0), "layout overlaps");

// raw funct 26 with any rd field (xd = 0): for the reserved rd values and the bad operands
#define NM_RAW(rd, a, b) asm volatile (".insn r " STR(CAT(CUSTOM_, XCUSTOM_ACC)) ", 0x3, " STR(k_NM_META) ", " #rd ", %0, %1" \
                                       :: "r"((uint64_t)(a)), "r"((uint64_t)(b)) : "memory")

static elem_t a[NM_I][NM_K] row_align(1);
static elem_t w[NM_K][NM_J];
static elem_t wc[KC][NM_J] row_align(1);
static nm_idx_t idx[KC][NM_J];
static nm_idx_t tidx[NTILE][DIM][DIM];           // index tile u, cut out of idx
static uint64_t meta_reg[NTILE][2 * DIM];         // register format (nm_pack_meta)
static uint8_t meta_sp[NTILE][TB] row_align(1);   // scratchpad format (nm_pack_meta_spad), u order, back to back
static uint8_t junk_sp[NTILE][TB] row_align(1);   // other index tiles (the "overwritten after the load" case)
static acc_t gold[NM_I][NM_J];
static acc_t c_out[NM_I][NM_J] row_align_acc(1);
static acc_t poison_tile[DIM][DIM] row_align_acc(1);

// bare metal has no memcmp
static int differ(const void * x, const void * y, size_t n) {
  const uint8_t * p = (const uint8_t *)x, * q = (const uint8_t *)y;
  for (size_t i = 0; i < n; i++) if (p[i] != q[i]) return 1;
  return 0;
}

static unsigned rs = 20261111;
static int rnd(int lo, int hi) { rs = rs * 1103515245u + 12345u; return lo + (int)((rs >> 8) % (unsigned)(hi - lo + 1)); }

// ---- data, packing ----
static void make_data(void) {
  for (int y = 0; y < NM_I; y++) for (int z = 0; z < NM_K; z++) a[y][z] = rnd(-128, 127);
  for (int z = 0; z < NM_K; z++) for (int x = 0; x < NM_J; x++) w[z][x] = rnd(-128, 127);
  nm_prune(NM_K, NM_J, &w[0][0], NM_J, NM_N, NM_M);
  NM_REQUIRE(nm_check(NM_K, NM_J, &w[0][0], NM_J, NM_N, NM_M));
  nm_compress(NM_K, NM_J, &w[0][0], NM_J, &wc[0][0], &idx[0][0], NM_J, NM_N, NM_M);
  nm_ref_dense(NM_I, NM_K, NM_J, &a[0][0], NM_K, &w[0][0], NM_J, &gold[0][0], NM_J);
  static acc_t gold2[NM_I][NM_J];
  nm_ref_sparse(NM_I, NM_K, NM_J, &a[0][0], NM_K, &wc[0][0], &idx[0][0], NM_J, &gold2[0][0], NM_J, NM_N, NM_M);
  NM_REQUIRE(!differ(gold, gold2, sizeof(gold)));
  for (int u = 0; u < NTILE; u++) {
    const int jt = u / KCT, t = u % KCT;
    for (int r = 0; r < DIM; r++) for (int c = 0; c < DIM; c++) tidx[u][r][c] = idx[t * DIM + r][jt * DIM + c];
    nm_pack_meta(&tidx[u][0][0], DIM, DIM, DIM, NM_M, meta_reg[u]);
    nm_pack_meta_spad(&tidx[u][0][0], DIM, DIM, DIM, NM_M, meta_sp[u]);
    nm_idx_t inv[DIM][DIM];
    for (int r = 0; r < DIM; r++) for (int c = 0; c < DIM; c++) inv[r][c] = NM_M - 1 - tidx[u][r][c];
    nm_pack_meta_spad(&inv[0][0], DIM, DIM, DIM, NM_M, junk_sp[u]);
  }
  for (int y = 0; y < DIM; y++) for (int x = 0; x < DIM; x++) poison_tile[y][x] = POISON;
}

// the spec's worked example (fix-a-spec.md section 9): idx[r][c] = (3r + c*c + (r*c)/4) mod m
#if NM_M == 4
static const uint8_t spec_bytes[] = {
  0x44,0x44,0x44,0x44,0x33,0x44,0x99,0xee,0x3e,0x94,0x3e,0x94,0x29,0xd4,0x83,0x7e,
  0x28,0x28,0x28,0x28,0xd7,0x28,0x7d,0x82,0xd2,0x78,0xd2,0x78,0xcd,0xb8,0x67,0x12,
  0xcc,0xcc,0xcc,0xcc,0xbb,0xcc,0x11,0x66,0xb6,0x1c,0xb6,0x1c,0xa1,0x5c,0x0b,0xf6,
  0xa0,0xa0,0xa0,0xa0,0x5f,0xa0,0xf5,0x0a,0x5a,0xf0,0x5a,0xf0,0x45,0x30,0xef,0x9a,
};
#elif NM_M == 8
static const uint8_t spec_bytes[] = {
  0x08,0x83,0x30,0x08,0x83,0x30,0xe3,0xc9,0xa2,0x75,0xec,0xeb,0xfe,0x80,0x54,0xda,
  0xc9,0xc6,0x91,0xc9,0xe6,0x07,0xa5,0xbd,0xb4,0x00,0x99,0xb4,0x00,0x99,0x4f,0x47,
  0x0b,0xd9,0x6b,0x50,0x62,0x0e,0xbd,0x46,0x47,0x2f,0x3d,0x47,0x4f,0xab,0x22,0x06,
  0x18,0x8e,0xe1,0x18,0x8e,0xe1,0xf3,0xc4,0x73,0x45,0xe9,0xb8,0xce,0x8d,0x05,0xea,
  0xc4,0x97,0xa1,0xc4,0xb7,0x17,0xa0,0x6e,0x84,0x0d,0x4a,0x84,0x0d,0x4a,0x5f,0x42,
  0xd8,0xe9,0x66,0x01,0x72,0x0b,0x6e,0x56,0x42,0xfc,0x0d,0x42,0x1c,0xbb,0x2f,0xd7,
};
#endif
static nm_idx_t spec_idx[DIM][DIM];
static uint8_t spec_sp[TB] row_align(1);

static int check_packer(void) {
  int fails = 0;
  for (int r = 0; r < DIM; r++) for (int c = 0; c < DIM; c++) spec_idx[r][c] = (3 * r + c * c + (r * c) / 4) % NM_M;
  nm_pack_meta_spad(&spec_idx[0][0], DIM, DIM, DIM, NM_M, spec_sp);
#if DIM == 16 && (NM_M == 4 || NM_M == 8)
  NM_REQUIRE(sizeof(spec_bytes) == TB && R == (NM_M == 4 ? 4 : 6) && nm_spad_rows_per_tile(NM_M) == R);
  if (differ(spec_sp, spec_bytes, TB)) { printf("packer: spec example bytes differ\n"); fails++; }
#endif
  uint64_t reg[2 * DIM];
  nm_pack_meta(&spec_idx[0][0], DIM, DIM, DIM, NM_M, reg);
  if (NM_M == 4 && differ(reg, spec_sp, TB)) { printf("packer: 2:4 spad bytes != nm_pack_meta\n"); fails++; }
  static uint8_t via_reg[TB + 1] row_align(1); // +1: the unaligned (byte by byte) path too
  nm_meta_regs_to_spad(reg, NM_M, via_reg);
  if (differ(via_reg, spec_sp, TB)) { printf("packer: nm_meta_regs_to_spad differs\n"); fails++; }
  nm_meta_regs_to_spad(reg, NM_M, via_reg + 1);
  if (differ(via_reg + 1, spec_sp, TB)) { printf("packer: nm_meta_regs_to_spad (unaligned) differs\n"); fails++; }
  for (int u = 0; u < NTILE; u++) {
    nm_meta_regs_to_spad(meta_reg[u], NM_M, via_reg);
    if (differ(via_reg, meta_sp[u], TB)) { printf("packer: nm_meta_regs_to_spad differs on tile %d\n", u); fails++; }
  }
  nm_idx_t back[DIM][DIM];
  nm_unpack_meta(reg, NM_M, &back[0][0], DIM);
  if (differ(back, spec_idx, sizeof(back))) { printf("packer: unpack round trip differs\n"); fails++; }
  // edge tile: top-left 5 x 11 of a 21-wide array == packing a zero-padded copy
  static nm_idx_t big[21][21], pad[DIM][DIM];
  for (int r = 0; r < 21; r++) for (int c = 0; c < 21; c++) big[r][c] = rnd(0, NM_M - 1);
  for (int r = 0; r < DIM; r++) for (int c = 0; c < DIM; c++) pad[r][c] = (r < 5 && c < 11) ? big[r][c] : 0;
  uint8_t e1[TB], e2[TB];
  for (int i = 0; i < (int)TB; i++) e1[i] = 0xA5; // the packer zeroes its output itself
  nm_pack_meta_spad(&big[0][0], 21, 5, 11, NM_M, e1);
  nm_pack_meta_spad(&pad[0][0], DIM, DIM, DIM, NM_M, e2);
  if (differ(e1, e2, TB)) { printf("packer: edge tile differs\n"); fails++; }
  printf("%-26s %s\n", "packer (spec section 9)", fails ? "FAIL" : "ok");
  return fails;
}

// ---- gemmini helpers ----
static void poison(void) {
  for (int y = 0; y < NM_I; y++) for (int x = 0; x < NM_J; x++) c_out[y][x] = POISON;
  gemmini_config_ld(DIM * sizeof(acc_t));
  for (int it = 0; it < IT; it++) for (int jt = 0; jt < JT; jt++) gemmini_mvin(&poison_tile[0][0], ACC_BIT | C_ACC(it, jt));
  gemmini_fence();
}
static int check(const char * name) {
  gemmini_fence();
  gemmini_config_st(NM_J * sizeof(acc_t));
  for (int it = 0; it < IT; it++) for (int jt = 0; jt < JT; jt++)
    gemmini_mvout(&c_out[it * DIM][jt * DIM], ACC_BIT | FULL_BIT | C_ACC(it, jt));
  gemmini_fence();
  int bad = 0;
  for (int y = 0; y < NM_I; y++) for (int x = 0; x < NM_J; x++)
    if (c_out[y][x] != gold[y][x]) { if (!bad) printf("%s: mismatch at (%d,%d): got %d want %d\n", name, y, x, c_out[y][x], gold[y][x]); bad++; }
  printf("%-26s %s (%d/%d wrong)\n", name, bad ? "FAIL" : "ok", bad, NM_I * NM_J);
  return bad != 0;
}
// index tiles u0 .. u0+n-1 (scratchpad format, from src) to rows row.. (mvin3, 16 byte rows)
static void mvin_idx(const uint8_t * src, uint32_t row, int n) { nm_mvin_meta_spad(src, row, n, 2); }
static void sparse_on(void) { gemmini_nm_config_ex(1, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, 1); }
static void dense_ws(void) { gemmini_nm_config_ex(0, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, 1); }
// tile u: real preload + compute_preloaded for row tile 0, then GARBAGE preload + compute_accumulated for the
// other row tiles (weights and indexes reused); ld_row / ld_n != 0: a load issued right after the first compute
static void tile(int u, uint32_t ld_row, int ld_n) {
  const int jt = u / KCT, t = u % KCT;
  for (int it = 0; it < IT; it++) {
    const uint32_t c = ACC_BIT | (t > 0 ? ACCUM_BIT : 0) | C_ACC(it, jt);
    if (it == 0) {
      gemmini_preload(WC_SP(t, jt), c);
      gemmini_compute_preloaded(A_SP(it, 2 * t), A_SP(it, 2 * t + 1));
      if (ld_n) gemmini_nm_load_spad(ld_row, ld_n);
    } else {
      gemmini_preload(GARBAGE_ADDR, c);
      gemmini_compute_accumulated(A_SP(it, 2 * t), A_SP(it, 2 * t + 1));
    }
  }
}
// start of every case: fence + clear, outputs poisoned, sparse on
static void begin(void) { gemmini_nm_reset(); poison(); sparse_on(); }
static void end(void) { gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0); }

static void setup_spad(void) {
  gemmini_flush(0);
  gemmini_config_ld(NM_K * sizeof(elem_t));
  for (int it = 0; it < IT; it++) for (int kt = 0; kt < KT; kt++) gemmini_mvin(&a[it * DIM][kt * DIM], A_SP(it, kt));
  gemmini_extended3_config_ld(NM_J * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
  for (int t = 0; t < KCT; t++) for (int jt = 0; jt < JT; jt++) gemmini_extended_mvin2(&wc[t * DIM][jt * DIM], WC_SP(t, jt), DIM, DIM);
  gemmini_extended3_config_ld(DIM * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 2); // index rows: 16 bytes apart
  mvin_idx(meta_sp[0], IDX_B3, NTILE);
  mvin_idx(meta_sp[0], IDX_B2, NTILE);
  mvin_idx(meta_sp[0], IDX_END0, 4);
  mvin_idx(meta_sp[4], IDX_END3, 4);
  gemmini_fence();
}

int main(void) {
  make_data();
  printf("nm_sparse_spad: %d:%d, R = %d rows per index tile, %d tiles\n", NM_N, NM_M, (int)R, NTILE);
  setup_spad();

#ifndef NEG
  int fails = check_packer();

  // batches of 4, each loaded right after the compute of the previous batch's last tile (the first one up front)
  begin();
  gemmini_nm_load_spad(IDX_B3, 4);
  for (int u = 0; u < NTILE; u++) tile(u, IDX_B3 + (u + 1) * R, (u % 4 == 3 && u + 1 < NTILE) ? 4 : 0);
  end(); fails += check("load 4, early");

  // loads of 1, 3, 2, 2 tiles, each right before its batch (bank 2 copy)
  begin();
  { const int nb[] = {1, 3, 2, 2}; int u = 0;
    for (int b = 0; b < 4; b++) { gemmini_nm_load_spad(IDX_B2 + u * R, nb[b]); for (int k = 0; k < nb[b]; k++, u++) tile(u, 0, 0); } }
  end(); fails += check("loads of 1,3,2,2");

  // one register tile, then loads (legal: no half tile, no load before the register rows)
  begin();
  nm_send_meta(meta_reg[0], NM_M);
  gemmini_nm_load_spad(IDX_B3 + 1 * R, 3);
  for (int u = 0; u < NTILE; u++) tile(u, IDX_B3 + (u + 1) * R, (u % 4 == 3 && u + 1 < NTILE) ? 4 : 0);
  end(); fails += check("register tile, then loads");

  // loads issued in dense mode wait in the fifo for the sparse preloads
  gemmini_nm_reset(); poison(); dense_ws();
  gemmini_nm_load_spad(IDX_B3, 4);
  sparse_on();
  for (int u = 0; u < NTILE; u++) tile(u, IDX_B3 + (u + 1) * R, (u % 4 == 3 && u + 1 < NTILE) ? 4 : 0);
  end(); fails += check("loads in dense mode");

  // tiles ending at the last row of bank 0 and at the last row of the scratchpad
  begin();
  gemmini_nm_load_spad(IDX_END0, 4);
  for (int u = 0; u < NTILE; u++) tile(u, IDX_END3, u == 3 ? 4 : 0);
  end(); fails += check("bank edges");

  // index rows overwritten by a mvin right after their load: the load already has the old rows
  begin();
  mvin_idx(meta_sp[0], IDX_B3 + 64 * R, 4); // rows no other case uses
  gemmini_nm_load_spad(IDX_B3 + 64 * R, 4);
  mvin_idx(junk_sp[0], IDX_B3 + 64 * R, 4);
  for (int u = 0; u < 4; u++) tile(u, 0, 0);
  mvin_idx(meta_sp[4], IDX_B3 + 64 * R, 4);
  gemmini_nm_load_spad(IDX_B3 + 64 * R, 4);
  mvin_idx(junk_sp[4], IDX_B3 + 64 * R, 4);
  for (int u = 4; u < NTILE; u++) tile(u, 0, 0);
  end(); fails += check("rows overwritten after load");

  // after a load, gemmini_nm_reset() makes the register path legal again
  begin();
  gemmini_nm_load_spad(IDX_B3, 2);
  gemmini_nm_reset();
  for (int u = 0; u < NTILE; u++) { nm_send_meta(meta_reg[u], NM_M); tile(u, 0, 0); }
  end(); fails += check("reset, then register rows");

  // the spec's command sequence (section 9): its index pattern at row 12288, W at row 0, one compute
  {
    static elem_t ws[DIM][DIM] row_align(1), al[DIM][DIM] row_align(1), ah[DIM][DIM] row_align(1);
    for (int r = 0; r < DIM; r++) for (int c = 0; c < DIM; c++) { ws[r][c] = rnd(-128, 127); al[r][c] = rnd(-128, 127); ah[r][c] = rnd(-128, 127); }
    static acc_t outs[DIM][DIM] row_align_acc(1);
    const uint32_t c_acc = ACC_BIT | C_ACC(0, 0), alo = A_SP(1, 0) + 2 * DIM, ahi = alo + BANK_ROWS;
    poison();
    gemmini_nm_reset();
    gemmini_nm_config_ex(1, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, 1);
    gemmini_config_ld(16);
    gemmini_extended_mvin(&ws[0][0], 0, 16, 16);
    gemmini_extended_mvin(spec_sp, 12288, 16, R);
    gemmini_extended_mvin(&al[0][0], alo, 16, 16);
    gemmini_extended_mvin(&ah[0][0], ahi, 16, 16);
    gemmini_nm_load_spad(12288, 1);
    gemmini_preload(0, c_acc);
    gemmini_compute_preloaded(alo, ahi);
    gemmini_config_st(DIM * sizeof(acc_t));
    gemmini_mvout(&outs[0][0], c_acc | FULL_BIT);
    gemmini_fence();
    end();
    int bad = 0;
    for (int i = 0; i < DIM; i++) for (int j = 0; j < DIM; j++) {
      int64_t sum = 0;
      for (int r = 0; r < DIM; r++) {
        const int k = (r / NM_N) * NM_M + spec_idx[r][j];
        sum += (int64_t)ws[r][j] * (k < DIM ? al[i][k] : ah[i][k - DIM]);
      }
      bad += outs[i][j] != sum;
    }
    printf("%-26s %s (%d/%d wrong)\n", "spec section 9 sequence", bad ? "FAIL" : "ok", bad, DIM * DIM);
    fails += bad != 0;
  }

  if (fails) { printf("nm_sparse_spad: %d case(s) FAILED\n", fails); exit(1); }
  printf("nm_sparse_spad passed\n");
  exit(0);

#else // NEG: an illegal use; reaching the end means the check did not fire
  begin();
#if NEG == 21
  NM_RAW(x3, 0, 0);
#elif NEG == 22
  NM_RAW(x31, IDX_B3, 1);
#elif NEG == 23
  NM_RAW(x2, IDX_B3, 0);
#elif NEG == 24
  NM_RAW(x2, IDX_B3, 5);
#elif NEG == 25
  NM_RAW(x2, ((uint64_t)1 << 32) | IDX_B3, 1);
#elif NEG == 26
  NM_RAW(x2, IDX_B3, 0x101);
#elif NEG == 27
  NM_RAW(x2, ACC_BIT | 0, 1);
#elif NEG == 28
  NM_RAW(x2, GARBAGE_ADDR, 1);
#elif NEG == 29
  NM_RAW(x2, BANK_NUM * BANK_ROWS, 1);
#elif NEG == 30
  NM_RAW(x2, 3 * BANK_ROWS - 4 * R + 1, 4);
#elif NEG == 31
  NM_RAW(x2, BANK_NUM * BANK_ROWS - R + 1, 1);
#elif NEG == 32
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, meta_reg[0][0], meta_reg[0][1], k_NM_META);
  gemmini_nm_load_spad(IDX_B3, 1);
#elif NEG == 33
  gemmini_nm_load_spad(IDX_B3, 4);
  gemmini_nm_load_spad(IDX_B3 + 4 * R, 1);
#elif NEG == 34
  gemmini_nm_load_spad(IDX_B3, 4);
  gemmini_preload(WC_SP(0, 0), ACC_BIT | C_ACC(0, 0));
  gemmini_nm_load_spad(IDX_B3 + 4 * R, 1);
  gemmini_compute_preloaded(A_SP(0, 0), A_SP(0, 1));
#elif NEG == 35
  gemmini_nm_load_spad(IDX_B3, 1);
  nm_send_meta(meta_reg[1], NM_M);
#elif NEG == 36
  gemmini_nm_load_spad(IDX_B3, 1);
  gemmini_preload(WC_SP(0, 0), ACC_BIT | C_ACC(0, 0));
  gemmini_nm_reset(); // its fence does not make the clear legal: the preload still waits for its compute
  gemmini_compute_preloaded(A_SP(0, 0), A_SP(0, 1));
#elif NEG == 37
  gemmini_nm_load_spad(IDX_B3, 1);
  tile(0, 0, 0);
  tile(1, 0, 0);
#else
#error "unknown NEG case"
#endif
  gemmini_fence();
  printf("nm_sparse_spad NEG=%d: illegal use NOT caught\n", NEG);
  exit(0);
#endif
}
