// n:m structured sparsity helpers for gemmini
// changed: new file, software side of the sparse-nm work
//
// weights w are k x j (same as gemmini's B in weight stationary).
// pruning runs down each column: every group of m rows keeps n values.
// compressed form is (nm_groups(k,m)*n) x j values + same-shape index array,
// idx says which row inside the group (0..m-1) the value came from.
// kept values stay in increasing row order inside each group.
//
// the pattern: -DNM_N/-DNM_M if given, else GEMMINI_HW_NM_N/M from gemmini_params.h
// (written by a sparse config's elaboration). neither, or a mismatch, is an #error.

#ifndef GEMMINI_NM_H
#define GEMMINI_NM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "include/gemmini_params.h"

#if defined(NM_N) != defined(NM_M)
#error "define both NM_N and NM_M (e.g. -DNM_N=2 -DNM_M=4), or neither"
#endif
#if defined(GEMMINI_HW_NM_N) != defined(GEMMINI_HW_NM_M)
#error "gemmini_params.h defines only one of GEMMINI_HW_NM_N / GEMMINI_HW_NM_M"
#endif
#if defined(NM_N)
#if defined(GEMMINI_HW_NM_N) && (NM_N != GEMMINI_HW_NM_N || NM_M != GEMMINI_HW_NM_M)
#error "-DNM_N/-DNM_M disagree with GEMMINI_HW_NM_N/GEMMINI_HW_NM_M in gemmini_params.h"
#endif
#elif defined(GEMMINI_HW_NM_N)
#define NM_N GEMMINI_HW_NM_N
#define NM_M GEMMINI_HW_NM_M
#else
#error "no n:m pattern: pass -DNM_N=.. -DNM_M=.. or use a sparse config's gemmini_params.h"
#endif

// biggest group the helpers support (they keep a per-group bitmap on the stack)
#define NM_MAX_M 16

#define NM_LOG2_M ((NM_M) <= 2 ? 1 : (NM_M) <= 4 ? 2 : (NM_M) <= 8 ? 3 : 4)
_Static_assert((NM_N) >= 1 && (NM_M) == 2 * (NM_N), "sparse mode needs m == 2*n");
_Static_assert(((NM_M) & ((NM_M) - 1)) == 0, "m has to be a power of 2");
_Static_assert((NM_M) <= NM_MAX_M, "m is larger than NM_MAX_M");
_Static_assert(DIM % (NM_N) == 0, "DIM has to be a multiple of n");
_Static_assert(DIM * NM_LOG2_M <= 64, "one index row (DIM * log2(m) bits) has to fit a 64 bit register");

typedef uint8_t nm_idx_t;

// assert that also works bare metal (no __assert_func there)
#define NM_REQUIRE(cond) do { if (!(cond)) { printf("gemmini_nm.h:%d: %s failed\n", __LINE__, #cond); exit(1); } } while (0)

static inline int nm_abs(int x) { return x < 0 ? -x : x; }

// m-groups that k rows need; a short last group counts as a whole one (its missing rows are zeros)
static inline size_t nm_groups(size_t k, int m) { return (k + m - 1) / m; }

// prune w in place so each m-group in each column keeps its n biggest (by magnitude), ties to the lower row
static void nm_prune(size_t k, size_t j, elem_t * w, size_t stride, int n, int m) {
  NM_REQUIRE(m >= 1 && m <= NM_MAX_M && n >= 0 && n <= m);
  for (size_t c = 0; c < j; c++) {
    for (size_t g = 0; g < k; g += m) {
      bool keep[NM_MAX_M] = {0};
      for (int pick = 0; pick < n; pick++) {
        int best = -1;
        int best_mag = -1;
        for (int r = 0; r < m && g + r < k; r++) {
          if (keep[r]) continue;
          int mag = nm_abs(w[(g + r) * stride + c]);
          if (mag > best_mag) { best_mag = mag; best = r; }
        }
        if (best >= 0) keep[best] = true;
      }
      for (int r = 0; r < m && g + r < k; r++)
        if (!keep[r]) w[(g + r) * stride + c] = 0;
    }
  }
}

// true if every m-group in every column has at most n nonzeros
static bool nm_check(size_t k, size_t j, const elem_t * w, size_t stride, int n, int m) {
  for (size_t c = 0; c < j; c++)
    for (size_t g = 0; g < k; g += m) {
      int nz = 0;
      for (int r = 0; r < m && g + r < k; r++)
        nz += w[(g + r) * stride + c] != 0;
      if (nz > n) return false;
    }
  return true;
}

// squash n:m pruned w (k x j) into wc ((nm_groups(k,m)*n) x j) + idx; short groups pad with zero weights
static void nm_compress(size_t k, size_t j, const elem_t * w, size_t w_stride,
    elem_t * wc, nm_idx_t * idx, size_t c_stride, int n, int m) {
  NM_REQUIRE(m >= 1 && m <= NM_MAX_M && n >= 0 && n <= m);
  for (size_t c = 0; c < j; c++) {
    for (size_t g = 0; g < nm_groups(k, m); g++) {
      int out = 0;
      bool used[NM_MAX_M] = {0};
      for (int r = 0; r < m; r++) {
        elem_t v = g * m + r < k ? w[(g * m + r) * w_stride + c] : 0;
        if (v != 0) {
          NM_REQUIRE(out < n); // w is not n:m pruned
          wc[(g * n + out) * c_stride + c] = v;
          idx[(g * n + out) * c_stride + c] = r;
          used[r] = true;
          out++;
        }
      }
      // pad, keeping idx increasing so it still looks like a sorted pattern
      for (int r = 0; out < n && r < m; r++) {
        if (used[r]) continue;
        wc[(g * n + out) * c_stride + c] = 0;
        idx[(g * n + out) * c_stride + c] = r;
        out++;
      }
      // re-sort the group by idx (padding may have landed out of order)
      for (int a = 0; a < n; a++)
        for (int b = a + 1; b < n; b++) {
          size_t pa = (g * n + a) * c_stride + c, pb = (g * n + b) * c_stride + c;
          if (idx[pb] < idx[pa]) {
            nm_idx_t ti = idx[pa]; idx[pa] = idx[pb]; idx[pb] = ti;
            elem_t tv = wc[pa]; wc[pa] = wc[pb]; wc[pb] = tv;
          }
        }
    }
  }
}

// undo nm_compress, gives back the pruned k x j matrix
static void nm_decompress(size_t k, size_t j, const elem_t * wc, const nm_idx_t * idx,
    size_t c_stride, elem_t * w, size_t w_stride, int n, int m) {
  for (size_t r = 0; r < k; r++)
    for (size_t c = 0; c < j; c++)
      w[r * w_stride + c] = 0;
  for (size_t g = 0; g < nm_groups(k, m); g++)
    for (int s = 0; s < n; s++)
      for (size_t c = 0; c < j; c++) {
        size_t p = (g * n + s) * c_stride + c;
        if (g * m + idx[p] < k) // padding past k has weight 0
          w[(g * m + idx[p]) * w_stride + c] += wc[p];
      }
}

// plain dense reference, c = a(i x k) * w(k x j), int accumulate
static void nm_ref_dense(size_t i, size_t k, size_t j, const elem_t * a, size_t a_stride,
    const elem_t * w, size_t w_stride, acc_t * c, size_t c_stride) {
  for (size_t y = 0; y < i; y++)
    for (size_t x = 0; x < j; x++) {
      acc_t s = 0;
      for (size_t z = 0; z < k; z++)
        s += (acc_t)a[y * a_stride + z] * (acc_t)w[z * w_stride + x];
      c[y * c_stride + x] = s;
    }
}

// sparse reference, the math the sparse array does: each kept weight grabs the activation its idx points at
static void nm_ref_sparse(size_t i, size_t k, size_t j, const elem_t * a, size_t a_stride,
    const elem_t * wc, const nm_idx_t * idx, size_t c_stride_w,
    acc_t * c, size_t c_stride, int n, int m) {
  size_t kc = nm_groups(k, m) * n;
  for (size_t y = 0; y < i; y++)
    for (size_t x = 0; x < j; x++) {
      acc_t s = 0;
      for (size_t r = 0; r < kc; r++) {
        size_t p = r * c_stride_w + x;
        size_t g = r / n;
        if (g * m + idx[p] < k) // padding past k has weight 0
          s += (acc_t)a[y * a_stride + g * m + idx[p]] * (acc_t)wc[p];
      }
      c[y * c_stride + x] = s;
    }
}

// ---------------------------------------------------------------------------
// gemmini side of sparse mode
// changed: added sparse config, index loading and reset wrappers
//
// protocol (the rtl and the spike model implement the same one):
//  - config_ex with rs1[10] set turns sparse mode on. legal only with WS and no a/b transpose,
//    and no LOOP_WS / LOOP_CONV_WS while it is on
//  - each weight buffer keeps the mode of its last real preload (GARBAGE preloads dont update it):
//    after changing rs1[10], issue a real preload before computing
//  - nm meta cmds (funct 26) send one DIM x DIM index tile into a NM_META_FIFO_DEPTH (4) deep fifo
//    (log2(m) bits per idx, DIM per row with col 0 lowest, whole rows per 64 bit reg, rs1 then rs2)
//  - each real sparse preload pops one tile: send each tile before its preload, and at most 4 tiles
//    ahead of their preloads (a 5th deadlocks the rtl). preload(GARBAGE_ADDR, c) pops nothing (weight reuse)
//  - compute_preloaded(a_lo, a_hi) streams a 2*DIM wide activation tile, no bias slot. a_lo rows are
//    A_stride apart, a_hi rows always 1 apart. A must not be GARBAGE
//  - gemmini_nm_reset() drops a half-sent tile and everything in the fifo (the mode bits stay)
//  - changed: Fix A. gemmini_nm_load_spad(row, n) (funct 26, rd == 2) appends n (1..4) index tiles read from
//    scratchpad rows row .. row + n*NM_SPAD_ROWS_PER_TILE - 1 (tile t at row + t*R). each tile is packed tight by
//    nm_pack_meta_spad() and moved in with plain int8 mvins (16 byte rows, <= DIM rows per mvin). rules:
//    row is a plain scratchpad row (no acc bit, not GARBAGE) and all n*R rows sit in one bank; the same
//    "at most 4 tiles ahead" budget as above (a load of n needs n free slots); not while a register tile is half
//    sent; after a load, no register index cmds (rd == 0) until the next gemmini_nm_reset() (which fences, so no
//    load is still outstanding). funct 26 rd 3..31 are reserved. index rows must come from mvin (not from a C
//    output still in flight). never put a load between a preload and its compute (the rtl can deadlock; spike
//    won't notice). issue it before the preload or after the compute, e.g. right after the compute of the
//    previous batch's last tile, so the hardware can fetch the rows before the preloads need them
//  - changed: sparse-loop-fix. LOOP_WS is legal in sparse mode when (and only when) its rs1[3] is set and a
//    LOOP_WS_CONFIG_NM (funct 28) came first: gemmini_loop_ws_nm(), see the end of this file. LOOP_CONV_WS stays illegal
// so one squashed DIM x DIM tile covers 2*DIM rows of the original weights

#include "include/gemmini.h"

#define k_NM_META 26

#define NM_CONFIG_EX_SPARSE_BIT 10

// a tile holds a slot until its preload fires (after its compute dispatches): a 5th sent before compute(t) deadlocks
#define NM_META_FIFO_DEPTH 4

// same as gemmini_extended3_config_ex but with the sparse bit
#define gemmini_nm_config_ex(sparse, sys_act, sys_shift, sys_acc_scale, C_stride, A_stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)sys_acc_scale) << 32) | ((uint64_t)(A_stride) << 16) | ((uint64_t)(sparse) << NM_CONFIG_EX_SPARSE_BIT) | ((sys_act) << 3) | ((WEIGHT_STATIONARY) << 2) | CONFIG_EX, ((uint64_t)(C_stride) << 48) | (sys_shift), k_CONFIG)

// fence, then funct 26 with rd == 1 (xd = 0, no writeback): clears the row assembler and the index fifo
#define gemmini_nm_reset() do { \
    gemmini_fence(); \
    asm volatile (".insn r " STR(CAT(CUSTOM_, XCUSTOM_ACC)) ", 0x3, " STR(k_NM_META) ", x1, %0, %1" \
                  :: "r"((uint64_t)0), "r"((uint64_t)0) : "memory"); \
  } while (0)

static inline int nm_idx_bits(int m) {
  int b = 0;
  while ((1 << b) < m) b++;
  return b;
}

// how many meta cmds one DIM x DIM index tile takes
static inline int nm_meta_cmds(int m) {
  const int rows_per_cmd = 2 * (64 / (DIM * nm_idx_bits(m)));
  return (DIM + rows_per_cmd - 1) / rows_per_cmd;
}

// pack the top-left rows x cols of an index tile (rest = 0) into 2*nm_meta_cmds(m) words, rs1 then rs2
static void nm_pack_meta(const nm_idx_t * idx, size_t stride, size_t rows, size_t cols, int m, uint64_t * out) {
  const int ib = nm_idx_bits(m);
  const int row_bits = DIM * ib;
  const int rows_per_reg = 64 / row_bits;
  const int words = 2 * nm_meta_cmds(m);
  size_t r = 0;
  for (int w = 0; w < words; w++) {
    uint64_t v = 0;
    for (int q = 0; q < rows_per_reg; q++, r++) {
      if (r >= DIM || r >= rows) continue;
      for (size_t c = 0; c < DIM && c < cols; c++)
        v |= (uint64_t)(idx[r * stride + c] & ((1 << ib) - 1)) << (q * row_bits + c * ib);
    }
    out[w] = v;
  }
}

// send an already packed index tile
static inline void nm_send_meta(const uint64_t * packed, int m) {
  const int cmds = nm_meta_cmds(m);
  for (int i = 0; i < cmds; i++)
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, packed[2 * i], packed[2 * i + 1], k_NM_META);
}

// pack + send in one go (slow-ish, it packs every time)
static void nm_load_meta(const nm_idx_t * idx, size_t stride, size_t rows, size_t cols, int m) {
  uint64_t packed[2 * DIM];
  nm_pack_meta(idx, stride, rows, cols, m, packed);
  nm_send_meta(packed, m);
}

// changed: Fix A. undo nm_pack_meta: the 2*nm_meta_cmds(m) words of one tile back into a DIM x DIM index tile
static void nm_unpack_meta(const uint64_t * packed, int m, nm_idx_t * idx, size_t stride) {
  const int ib = nm_idx_bits(m);
  const int row_bits = DIM * ib;
  const int rows_per_reg = 64 / row_bits;
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++)
      idx[r * stride + c] = (nm_idx_t)((packed[r / rows_per_reg] >> ((r % rows_per_reg) * row_bits + c * ib)) & ((1u << ib) - 1));
}

// ---------------------------------------------------------------------------
// changed: Fix A. index tiles in the scratchpad (fix-a-spec.md section 3)
// a tile is one bit string S of DIM*DIM*log2(m) bits, idx[r][c] at S[(r*DIM + c)*b ..], lowest bit first; in dram it
// is S as a little-endian byte array (NM_META_SPAD_BYTES per tile), moved in as NM_SPAD_ROWS_PER_TILE scratchpad
// rows of DIM int8 columns. 2:4: the same bytes as nm_pack_meta; 4:8: tighter (rows are not padded to 64 bits).

// scratchpad rows of one tile for pattern m (round up of the tile bits over the bits of one scratchpad row)
static inline size_t nm_spad_rows_per_tile(int m) {
  const size_t row_bits = DIM * 8 * sizeof(elem_t);
  return ((size_t)DIM * DIM * nm_idx_bits(m) + row_bits - 1) / row_bits;
}
#define NM_SPAD_ROWS_PER_TILE_CALC (((size_t)DIM * DIM * NM_LOG2_M + DIM * 8 * sizeof(elem_t) - 1) / (DIM * 8 * sizeof(elem_t)))
#ifdef GEMMINI_HW_NM_SPAD_ROWS_PER_TILE
// written by the elaboration of a Fix A sparse config
#define NM_SPAD_ROWS_PER_TILE GEMMINI_HW_NM_SPAD_ROWS_PER_TILE
_Static_assert(GEMMINI_HW_NM_SPAD_ROWS_PER_TILE == NM_SPAD_ROWS_PER_TILE_CALC,
               "GEMMINI_HW_NM_SPAD_ROWS_PER_TILE (gemmini_params.h) disagrees with the tight index tile packing");
#else
#define NM_SPAD_ROWS_PER_TILE NM_SPAD_ROWS_PER_TILE_CALC
#endif
// bytes of one packed tile in dram (R scratchpad rows)
#define NM_META_SPAD_BYTES (NM_SPAD_ROWS_PER_TILE * DIM * sizeof(elem_t))

// pack the top-left rows x cols of an index tile (rest = 0) tight, as the scratchpad wants; out gets
// nm_spad_rows_per_tile(m) * DIM * sizeof(elem_t) bytes (zeroed here, then the spec's reference packing)
static void nm_pack_meta_spad(const nm_idx_t * idx, size_t stride, size_t rows, size_t cols, int m, uint8_t * out) {
  const int b = nm_idx_bits(m);
  const size_t nbytes = nm_spad_rows_per_tile(m) * DIM * sizeof(elem_t);
  for (size_t i = 0; i < nbytes; i++) out[i] = 0;
  for (size_t r = 0; r < DIM && r < rows; r++)
    for (size_t c = 0; c < DIM && c < cols; c++) {
      const unsigned v = idx[r * stride + c] & ((1u << b) - 1);
      const size_t p = (r * DIM + c) * b;
      for (int k = 0; k < b; k++)
        if (v >> k & 1) out[(p + k) / 8] |= (uint8_t)(1u << ((p + k) % 8));
    }
}

// repack one register-format tile (nm_pack_meta words) into the scratchpad format: row r's DIM*log2(m) bits move from
// its register word to bit r*DIM*log2(m) of the tight string (word shifts, so it is cheap enough for big layers)
static void nm_meta_regs_to_spad(const uint64_t * packed, int m, uint8_t * out) {
  const int row_bits = DIM * nm_idx_bits(m);
  const int rows_per_reg = 64 / row_bits;
  const uint64_t mask = row_bits >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << row_bits) - 1);
  const size_t nbytes = nm_spad_rows_per_tile(m) * DIM * sizeof(elem_t);
  uint64_t s[DIM * DIM * 4 / 64 + 3]; // log2(m) <= 4 (NM_MAX_M), plus the round up to whole scratchpad rows
  NM_REQUIRE(nbytes <= sizeof(s));
  for (size_t i = 0; i < sizeof(s) / 8; i++) s[i] = 0;
  for (int r = 0; r < DIM; r++) {
    const uint64_t v = (packed[r / rows_per_reg] >> ((r % rows_per_reg) * row_bits)) & mask;
    const size_t p = (size_t)r * row_bits;
    s[p / 64] |= v << (p % 64);
    if (p % 64 + row_bits > 64) s[p / 64 + 1] |= v >> (64 - p % 64);
  }
  if ((uintptr_t)out % 8 == 0 && nbytes % 8 == 0) {
    for (size_t i = 0; i < nbytes / 8; i++) ((uint64_t *)out)[i] = s[i]; // little-endian: byte i = bits 8i..8i+7
  } else {
    for (size_t i = 0; i < nbytes; i++) out[i] = (uint8_t)(s[i / 8] >> (8 * (i % 8)));
  }
}

// funct 26 with rd == 2 (xd = 0, no writeback): append n (1..4) tiles from scratchpad rows spad_row .. + n*R - 1.
// rs1 = the plain scratchpad row (no acc / accumulate bits), rs2 = n. no checks here (see the protocol above).
#define gemmini_nm_load_spad(spad_row, n) \
  asm volatile (".insn r " STR(CAT(CUSTOM_, XCUSTOM_ACC)) ", 0x3, " STR(k_NM_META) ", x2, %0, %1" \
                :: "r"((uint64_t)(uint32_t)(spad_row)), "r"((uint64_t)(n)) : "memory")
#define gemmini_nm_load(spad_row, ntiles) gemmini_nm_load_spad(spad_row, ntiles)

// mvin ntiles packed tiles (NM_META_SPAD_BYTES each, back to back in dram) to scratchpad rows spad_row .. with the
// mvin of load state ld_id (0, 1 or 2), configured by the caller for a DIM * sizeof(elem_t) byte stride, scale 1,
// no shrink; at most DIM rows per mvin
static inline void nm_mvin_meta_spad(const uint8_t * packed, uint32_t spad_row, size_t ntiles, int ld_id) {
  const size_t rows = ntiles * NM_SPAD_ROWS_PER_TILE;
  for (size_t r = 0; r < rows; r += DIM) {
    const size_t nr = rows - r < DIM ? rows - r : DIM;
    const uint8_t * src = packed + r * DIM * sizeof(elem_t);
    if (ld_id == 0) { gemmini_extended_mvin(src, spad_row + r, DIM, nr); }
    else if (ld_id == 1) { gemmini_extended_mvin2(src, spad_row + r, DIM, nr); }
    else { gemmini_extended_mvin3(src, spad_row + r, DIM, nr); }
  }
}

// ---------------------------------------------------------------------------
// streaming tiled matmul (dense or n:m sparse), weight stationary, int32 (full_C) output
// changed: added for the streaming benchmark (gemmini-sparsity/experiments/streaming), self-contained
//
// C (M x N acc_t, row stride ldc) = A (M x K elem_t, row stride lda) * W (K x N)
//   tiled_matmul_stream_dense: B = W, K x N (row stride ldb)
//   tiled_matmul_nm:           B = the n:m compressed W, (K/2) x N (row stride ldb), as nm_compress() makes it;
//                              meta = packed index tiles (nm_pack_meta of each DIM x DIM tile of the index array),
//                              tile (t, jt) at meta + (t * (N/DIM) + jt) * 2*nm_meta_cmds(NM_M)
// both stream A and B from main memory with the same loop:
//  - K advances in steps of NM_STREAM_KSTEP*DIM columns of A (= 4 dense or 2 compressed weight tiles); each step
//    is mvin'd (block mvins, up to 4 tiles each) into one of NM_STREAM_NBUF scratchpad buffers while the previous
//    step computes (step g+1's mvins are issued before step g's preload/computes)
//  - the output is computed in blocks of TI x TJ tiles (TI*TJ*DIM <= ACC_ROWS/2) that alternate between the two
//    accumulator halves; each weight tile is preloaded once per block and reused for its TI row tiles; the mvouts
//    of a block are spread over the steps of the next block
//  - blocks go row-block outer, column-block inner (a row block of A is re-read for every column block)
// sparse protocol: gemmini_nm_reset() + sparse config_ex first; index tiles go in batches of at most
// NM_META_FIFO_DEPTH, each batch right before its tiles' preload/computes and after all earlier computes;
// a dense config_ex at the end. the caller fences (gemmini_fence) before reading C.
// changed: Fix A. tiled_matmul_nm_spad: the same, but meta_sp = the index tiles in the scratchpad format
// (nm_pack_meta_spad, NM_META_SPAD_BYTES each, tile (t, jt) at meta_sp + (t * (N/DIM) + jt) * NM_META_SPAD_BYTES);
// each step's index tiles are mvin'd with its weights (mvin3, 16 byte rows) into the A bank the step's computes do
// not read, and each batch is one gemmini_nm_load_spad() where tiled_matmul_nm sends its index tiles.
// requirements: N % DIM == 0; K % DIM == 0 (dense) or K % (2*DIM) == 0 (sparse); M may end in a partial row tile.

#define NM_STREAM_NBUF 4
#define NM_STREAM_KSTEP 4

// default block shape: up to 8 row tiles, then as many column tiles as half the accumulator holds
static inline void nm_stream_tiles(size_t M, size_t N, size_t * TI, size_t * TJ) {
  const size_t mt = (M + DIM - 1) / DIM, nt = N / DIM, max_tiles = (ACC_ROWS / 2) / DIM;
  size_t ti = mt < 8 ? mt : 8;
  size_t tj = max_tiles / ti;
  if (tj > nt) tj = nt;
  *TI = ti;
  *TJ = tj;
}

// sparse: 0 dense, 1 n:m with register index tiles (meta), 2 n:m with scratchpad index tiles (meta_sp, Fix A)
static inline __attribute__((always_inline)) void nm_stream_impl(const int sparse,
    size_t M, size_t N, size_t K, const elem_t * A, size_t lda, const elem_t * B, size_t ldb,
    const uint64_t * meta, const uint8_t * meta_sp, acc_t * C, size_t ldc, size_t TI, size_t TJ) {
  const uint32_t acc_bit = (uint32_t)1 << (ADDR_LEN - 1), accum_bit = (uint32_t)1 << (ADDR_LEN - 2),
                 full_bit = (uint32_t)1 << (ADDR_LEN - 3);
  const size_t MT = (M + DIM - 1) / DIM, NT = N / DIM, KT = K / DIM;
  const size_t S = (KT + NM_STREAM_KSTEP - 1) / NM_STREAM_KSTEP;
  const size_t JB = (NT + TJ - 1) / TJ, nsteps = ((MT + TI - 1) / TI) * JB * S;
  const size_t mw = 2 * nm_meta_cmds(NM_M);
  const size_t a_slot = TI * NM_STREAM_KSTEP * DIM, w_slot = NM_STREAM_KSTEP * TJ * DIM;
  NM_REQUIRE(M >= 1 && N % DIM == 0 && K % (sparse ? 2 * DIM : DIM) == 0 && K >= DIM);
  NM_REQUIRE(TI >= 1 && TJ >= 1 && TI * TJ * DIM <= ACC_ROWS / 2 && 2 * a_slot <= BANK_ROWS && 2 * w_slot <= BANK_ROWS);
#define NM_A_SP(buf) ((uint32_t)(((buf) % 2) * BANK_ROWS + ((buf) / 2) * a_slot))
#define NM_W_SP(buf) ((uint32_t)((2 + (buf) % 2) * BANK_ROWS + ((buf) / 2) * w_slot))
  // Fix A index rows of a step: upper half of the A bank the step's computes do not read
  const size_t i_slot = (NM_STREAM_KSTEP / 2) * TJ * NM_SPAD_ROWS_PER_TILE;
  if (sparse == 2) NM_REQUIRE(2 * a_slot <= BANK_ROWS / 2 && BANK_ROWS / 2 + 2 * i_slot <= BANK_ROWS);
#define NM_I_SP(buf) ((uint32_t)((1 - (buf) % 2) * BANK_ROWS + BANK_ROWS / 2 + ((buf) / 2) * i_slot))

  if (sparse) {
    gemmini_nm_reset();
    gemmini_nm_config_ex(1, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, 1);
  } else {
    gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  }
  gemmini_extended3_config_ld(lda * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(ldb * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
  if (sparse == 2) { gemmini_extended3_config_ld(DIM * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 2); }
  gemmini_config_st(ldc * sizeof(acc_t));

  // mvouts still owed by the last finished block
  size_t st_blk = 0, st_next = 0, st_total = 0, st_per = 0;

  for (size_t g = 0; g <= nsteps; g++) {
    // ---- loads of step g (issued one step ahead of its computes) ----
    if (g < nsteps) {
      const size_t blk = g / S, s = g % S, buf = g % NM_STREAM_NBUF;
      const size_t i0 = (blk / JB) * TI, j0 = (blk % JB) * TJ;
      const size_t ti = MT - i0 < TI ? MT - i0 : TI, tj = NT - j0 < TJ ? NT - j0 : TJ;
      const size_t k0 = s * NM_STREAM_KSTEP, kw = KT - k0 < NM_STREAM_KSTEP ? KT - k0 : NM_STREAM_KSTEP;
      const size_t wt = sparse ? kw / 2 : kw, wrow0 = sparse ? k0 / 2 * DIM : k0 * DIM;
      for (size_t i = 0; i < ti; i++) {
        const size_t r = (i0 + i) * DIM, rows = M - r < DIM ? M - r : DIM;
        gemmini_extended_mvin(A + r * lda + k0 * DIM, NM_A_SP(buf) + i * NM_STREAM_KSTEP * DIM, kw * DIM, rows);
      }
      for (size_t q = 0; q < wt; q++)
        for (size_t jj = 0; jj < tj; jj += MAX_BLOCK_LEN) {
          const size_t blocks = tj - jj < MAX_BLOCK_LEN ? tj - jj : MAX_BLOCK_LEN;
          gemmini_extended_mvin2(B + (wrow0 + q * DIM) * ldb + (j0 + jj) * DIM, NM_W_SP(buf) + (q * TJ + jj) * DIM,
                                 blocks * DIM, DIM);
        }
      if (sparse == 2) // index tile (c, j) of the step at NM_I_SP(buf) + (c * tj + j) * R, a row of tj tiles per mvin3 run
        for (size_t c = 0; c < wt; c++)
          nm_mvin_meta_spad(meta_sp + ((k0 / 2 + c) * NT + j0) * NM_META_SPAD_BYTES,
                            NM_I_SP(buf) + (uint32_t)(c * tj * NM_SPAD_ROWS_PER_TILE), tj, 2);
    }
    if (g == 0) continue;

    // ---- preload/computes of step h = g - 1 ----
    const size_t h = g - 1, blk = h / S, s = h % S, buf = h % NM_STREAM_NBUF;
    const size_t i0 = (blk / JB) * TI, j0 = (blk % JB) * TJ;
    const size_t ti = MT - i0 < TI ? MT - i0 : TI, tj = NT - j0 < TJ ? NT - j0 : TJ;
    const size_t k0 = s * NM_STREAM_KSTEP, kw = KT - k0 < NM_STREAM_KSTEP ? KT - k0 : NM_STREAM_KSTEP;
    const uint32_t acc_base = (uint32_t)((blk % 2) * (ACC_ROWS / 2));
    const uint32_t a_sp = NM_A_SP(buf), w_sp = NM_W_SP(buf);
    if (!sparse) {
      for (size_t q = 0; q < kw; q++) {
        const uint32_t cflags = acc_bit | (k0 + q > 0 ? accum_bit : 0);
        for (size_t j = 0; j < tj; j++) {
          const uint32_t w = w_sp + (q * TJ + j) * DIM;
          size_t rows = M - i0 * DIM < DIM ? M - i0 * DIM : DIM;
          gemmini_extended_preload(w, cflags | (acc_base + j * DIM), DIM, DIM, DIM, rows);
          gemmini_extended_compute_preloaded(a_sp + q * DIM, GARBAGE_ADDR, DIM, rows, DIM, DIM);
          for (size_t i = 1; i < ti; i++) {
            const size_t r = (i0 + i) * DIM;
            rows = M - r < DIM ? M - r : DIM;
            gemmini_extended_preload(GARBAGE_ADDR, cflags | (acc_base + (i * TJ + j) * DIM), DIM, DIM, DIM, rows);
            gemmini_extended_compute_accumulated(a_sp + (i * NM_STREAM_KSTEP + q) * DIM, GARBAGE_ADDR, DIM, rows, DIM, DIM);
          }
        }
      }
    } else {
      const size_t ntile = kw / 2 * tj;
      for (size_t u0 = 0; u0 < ntile; u0 += NM_META_FIFO_DEPTH) {
        const size_t u1 = u0 + NM_META_FIFO_DEPTH < ntile ? u0 + NM_META_FIFO_DEPTH : ntile;
        if (sparse == 2) {
          gemmini_nm_load_spad(NM_I_SP(buf) + (uint32_t)(u0 * NM_SPAD_ROWS_PER_TILE), u1 - u0);
        } else {
          for (size_t u = u0; u < u1; u++)
            nm_send_meta(meta + ((k0 / 2 + u / tj) * NT + j0 + u % tj) * mw, NM_M);
        }
        for (size_t u = u0; u < u1; u++) {
          const size_t c = u / tj, j = u % tj;
          const uint32_t cflags = acc_bit | (k0 / 2 + c > 0 ? accum_bit : 0);
          const uint32_t w = w_sp + (c * TJ + j) * DIM;
          size_t rows = M - i0 * DIM < DIM ? M - i0 * DIM : DIM;
          gemmini_extended_preload(w, cflags | (acc_base + j * DIM), DIM, DIM, DIM, rows);
          gemmini_extended_compute_preloaded(a_sp + 2 * c * DIM, a_sp + (2 * c + 1) * DIM, DIM, rows, DIM, rows);
          for (size_t i = 1; i < ti; i++) {
            const size_t r = (i0 + i) * DIM;
            rows = M - r < DIM ? M - r : DIM;
            const uint32_t a_lo = a_sp + (i * NM_STREAM_KSTEP + 2 * c) * DIM;
            gemmini_extended_preload(GARBAGE_ADDR, cflags | (acc_base + (i * TJ + j) * DIM), DIM, DIM, DIM, rows);
            gemmini_extended_compute_accumulated(a_lo, a_lo + DIM, DIM, rows, DIM, rows);
          }
        }
      }
    }

    // ---- mvouts: a share of the previous block's tiles; all of them once nothing is left to compute ----
    {
      const size_t n_out = g == nsteps ? st_total - st_next : (st_per < st_total - st_next ? st_per : st_total - st_next);
      const size_t si0 = (st_blk / JB) * TI, sj0 = (st_blk % JB) * TJ;
      const size_t stj = NT - sj0 < TJ ? NT - sj0 : TJ;
      const uint32_t sbase = (uint32_t)((st_blk % 2) * (ACC_ROWS / 2));
      for (size_t e = st_next; e < st_next + n_out; e++) {
        const size_t i = e / stj, j = e % stj, r = (si0 + i) * DIM, rows = M - r < DIM ? M - r : DIM;
        gemmini_extended_mvout(C + r * ldc + (sj0 + j) * DIM, acc_bit | full_bit | (sbase + (i * TJ + j) * DIM), DIM, rows);
      }
      st_next += n_out;
    }
    if (s == S - 1) {
      // block blk is fully issued: its mvouts go out during the next block's steps (or right now if it was the last)
      st_blk = blk; st_next = 0; st_total = ti * tj; st_per = (st_total + S - 1) / S;
      if (g == nsteps)
        for (size_t e = 0; e < st_total; e++) {
          const size_t i = e / tj, j = e % tj, r = (i0 + i) * DIM, rows = M - r < DIM ? M - r : DIM;
          gemmini_extended_mvout(C + r * ldc + (j0 + j) * DIM, acc_bit | full_bit | (acc_base + (i * TJ + j) * DIM), DIM, rows);
        }
    }
  }
  if (sparse) gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
#undef NM_A_SP
#undef NM_W_SP
#undef NM_I_SP
}

static void tiled_matmul_stream_dense(size_t M, size_t N, size_t K, const elem_t * A, size_t lda,
    const elem_t * B, size_t ldb, acc_t * C, size_t ldc, size_t TI, size_t TJ) {
  nm_stream_impl(0, M, N, K, A, lda, B, ldb, NULL, NULL, C, ldc, TI, TJ);
}

static void tiled_matmul_nm(size_t M, size_t N, size_t K, const elem_t * A, size_t lda,
    const elem_t * Bc, size_t ldb, const uint64_t * meta, acc_t * C, size_t ldc, size_t TI, size_t TJ) {
  nm_stream_impl(1, M, N, K, A, lda, Bc, ldb, meta, NULL, C, ldc, TI, TJ);
}

// changed: Fix A. meta_sp: tile (t, jt) at meta_sp + (t * (N/DIM) + jt) * NM_META_SPAD_BYTES (16-byte aligned)
static void tiled_matmul_nm_spad(size_t M, size_t N, size_t K, const elem_t * A, size_t lda,
    const elem_t * Bc, size_t ldb, const uint8_t * meta_sp, acc_t * C, size_t ldc, size_t TI, size_t TJ) {
  nm_stream_impl(2, M, N, K, A, lda, Bc, ldb, NULL, meta_sp, C, ldc, TI, TJ);
}

// ---------------------------------------------------------------------------
// changed: sparse-loop-fix. n:m sparse mode inside the LOOP_WS hardware loop
// spec: gemmini-sparsity experiments/streaming/worklogs/fix-b-spec.md section 2.5 (the rtl and spike implement the same)
//  - funct 27 LOAD_IDX: an mvin of 1..16 index rows with a fixed config (16 byte dram rows, identity scale), so the
//    index rows need no config_ld and leave the A, B and bias (D) load strides alone. legal outside loops too
//  - funct 28 LOOP_WS_CONFIG_NM: rs1 = dram address of the loop's index tile 0 (non-zero, 16 byte aligned), rs2 = 0
//  - LOOP_WS rs1[3] = 1: a sparse loop. needs sparse mode (config_ex rs1[10]) with A_stride 1, a funct 28 since the
//    previous LOOP_WS, an empty index fifo (gemmini_nm_reset() first), an even K tile count, pad_K 0, no transposes,
//    no resadd, no low_D, no spad-only. bias (full-width D) works as in a dense loop
//  - operands of a sparse loop of I x J x K tiles (K = tiles of the original K): A as dense; B = the compressed
//    weights, K/2 tiles down; index tiles (NM_META_SPAD_BYTES each) back to back in dram, tile (k', j) at
//    idx_dram + (k' * J + j) * NM_META_SPAD_BYTES
//  - the loop keeps its index rows at the top of its scratchpad half, B below them, A at the bottom:
//    I*K*DIM + (K/2)*J*DIM + (K/2)*J*R rows have to fit the half, and the (K/2)*J*R index rows one bank

#define k_NM_LOAD_IDX 27
#define k_LOOP_WS_CONFIG_NM 28

#define NM_LOOP_WS_NM_BIT 3

// one index row in dram, and the most rows one LOAD_IDX moves
#define NM_LOAD_IDX_COLS 16
#define NM_LOAD_IDX_MAX_ROWS 16
_Static_assert(DIM * sizeof(elem_t) == NM_LOAD_IDX_COLS, "LOAD_IDX (funct 27) moves 16 byte index rows: needs DIM * sizeof(elem_t) == 16");
// whole index tiles one LOAD_IDX can move
#define NM_LOAD_IDX_TILES (NM_LOAD_IDX_MAX_ROWS / NM_SPAD_ROWS_PER_TILE)

// funct 27 with any cols / rows (legal: cols == 16, rows 1..16, spad_addr a plain scratchpad row)
#define gemmini_nm_extended_load_idx(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_NM_LOAD_IDX)
#define gemmini_nm_load_idx(dram_addr, spad_addr, rows) \
  gemmini_nm_extended_load_idx(dram_addr, spad_addr, NM_LOAD_IDX_COLS, rows)

// LOAD_IDX ntiles packed tiles (NM_META_SPAD_BYTES each, back to back in dram) to scratchpad rows spad_row ..
static inline void nm_load_idx_meta_spad(const uint8_t * packed, uint32_t spad_row, size_t ntiles) {
  for (size_t t = 0; t < ntiles; t += NM_LOAD_IDX_TILES) {
    const size_t nt = ntiles - t < NM_LOAD_IDX_TILES ? ntiles - t : NM_LOAD_IDX_TILES;
    gemmini_nm_load_idx(packed + t * NM_META_SPAD_BYTES, spad_row + (uint32_t)(t * NM_SPAD_ROWS_PER_TILE),
                        nt * NM_SPAD_ROWS_PER_TILE);
  }
}

#define gemmini_loop_ws_config_nm(idx_dram) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, idx_dram, 0, k_LOOP_WS_CONFIG_NM)

// gemmini_loop_ws (gemmini.h) plus the funct 28 and LOOP_WS rs1[3]; B = compressed weights, K = original K tiles
#define gemmini_loop_ws_nm(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act, a_spad_id, b_spad_id, is_resadd, idx_dram) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
    gemmini_loop_ws_config_nm(idx_dram) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(a_spad_id) << 18) | ((uint64_t)(b_spad_id) << 16) | ((uint64_t)(act) << 8) | ((uint64_t)1 << NM_LOOP_WS_NM_BIT) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((is_resadd) << 2) | ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

// K rounded up to what a sparse loop takes (K % (2*DIM) == 0); pad A with zero columns and W with zero rows
static inline size_t nm_loop_pad_k(size_t K) { return (K + 2 * DIM - 1) / (2 * DIM) * (2 * DIM); }

// dst (prows x pcols) = src (rows x cols) in the top left corner, zeros elsewhere
static void nm_zero_pad(size_t rows, size_t cols, const elem_t * src, size_t s_stride,
    size_t prows, size_t pcols, elem_t * dst, size_t d_stride) {
  NM_REQUIRE(prows >= rows && pcols >= cols);
  for (size_t r = 0; r < prows; r++)
    for (size_t c = 0; c < pcols; c++)
      dst[r * d_stride + c] = (r < rows && c < cols) ? src[r * s_stride + c] : 0;
}

// scratchpad rows one sparse loop of I x J x K tiles needs in its half: A, compressed B, index rows
static inline size_t nm_loop_spad_rows(size_t I, size_t J, size_t K) {
  return I * K * DIM + (K / 2) * J * (DIM + NM_SPAD_ROWS_PER_TILE);
}
// true if the rtl / spike take a sparse loop of this size (double buffered: half the scratchpad and accumulator).
// bank-split-fix: the loop keeps even A K tiles in one bank and odd ones in the other (bank-split-fix-spec.md 2.1, B
// "straddle"); this bound already implies that layout fits (even tiles <= one bank, idx rows < B rows), so no extra rule
static inline bool nm_loop_fits(size_t I, size_t J, size_t K) {
  return I >= 1 && J >= 1 && K >= 2 && K % 2 == 0 &&
         nm_loop_spad_rows(I, J, K) <= BANK_NUM * BANK_ROWS / 2 &&
         (K / 2) * J * NM_SPAD_ROWS_PER_TILE <= BANK_ROWS &&
         I * J * DIM <= ACC_ROWS / 2;
}

// tile sizes for tiled_matmul_nm_loop: tiled_matmul_auto's (gemmini.h) double-buffered rule with the index rows
// reserved and tile_K kept even
static inline void nm_loop_tiles(size_t M, size_t N, size_t K, size_t * TI, size_t * TJ, size_t * TK) {
  const size_t mt = (M + DIM - 1) / DIM, nt = (N + DIM - 1) / DIM, kt = K / DIM;
  NM_REQUIRE(M >= 1 && N >= 1 && K >= 2 * DIM && K % (2 * DIM) == 0);
  size_t ij = 1; // floor(sqrt(tiles in half the accumulator)), as tiled_matmul_auto's db_max_tile_i_j
  while ((ij + 1) * (ij + 1) <= (ACC_ROWS / 2) / DIM) ij++;
  size_t ti = mt < ij ? mt : ij, tj = nt < ij ? nt : ij;
  size_t tk = ((BANK_NUM * BANK_ROWS / 4) / DIM) / ij;
  if (tk > kt) tk = kt;
  tk &= ~(size_t)1;
  if (tk < 2) tk = 2;
  while (tk > 2 && !nm_loop_fits(ti, tj, tk)) tk -= 2;
  while (tj > 1 && !nm_loop_fits(ti, tj, tk)) tj--;
  while (ti > 1 && !nm_loop_fits(ti, tj, tk)) ti--;
  NM_REQUIRE(nm_loop_fits(ti, tj, tk));
  for (;;) {
    bool increased = false;
    if (tj + 1 <= nt && nm_loop_fits(ti, tj + 1, tk)) { tj++; increased = true; }
    if (ti + 1 <= mt && nm_loop_fits(ti + 1, tj, tk)) { ti++; increased = true; }
    if (tk + 2 <= kt && nm_loop_fits(ti, tj, tk + 2)) { tk += 2; increased = true; }
    if (!increased) break;
  }
  *TI = ti; *TJ = tj; *TK = tk;
}

// index tiles from the (t, jt) order of tiled_matmul_nm_spad (tile (t, jt) at src + (t * src_nt + jt) * NM_META_SPAD_BYTES;
// src_nt = N / DIM there, or more when N x K is a corner of a bigger layer) into the order tiled_matmul_nm_loop's calls
// read them: column blocks of tile_J, inside them K blocks of tile_K / 2 compressed tiles, inside them the loop's own
// (k', j) order. one-time prep, like the compression; dst 16 byte aligned
static void nm_loop_order_idx(size_t N, size_t K, size_t tile_J, size_t tile_K, const uint8_t * src, size_t src_nt, uint8_t * dst) {
  const size_t NT = (N + DIM - 1) / DIM, KBT = K / (2 * DIM), tkb = tile_K / 2;
  NM_REQUIRE(tile_J >= 1 && tkb >= 1 && tile_K % 2 == 0 && K % (2 * DIM) == 0 && src_nt >= NT);
  const bool words = (uintptr_t)src % 8 == 0 && (uintptr_t)dst % 8 == 0 && NM_META_SPAD_BYTES % 8 == 0;
  size_t u = 0;
  for (size_t j0 = 0; j0 < NT; j0 += tile_J)
    for (size_t k0 = 0; k0 < KBT; k0 += tkb)
      for (size_t t = k0; t < k0 + tkb && t < KBT; t++)
        for (size_t jt = j0; jt < j0 + tile_J && jt < NT; jt++, u++) {
          const uint8_t * s = src + (t * src_nt + jt) * NM_META_SPAD_BYTES;
          uint8_t * d = dst + u * NM_META_SPAD_BYTES;
          if (words) for (size_t i = 0; i < NM_META_SPAD_BYTES / 8; i++) ((uint64_t *)d)[i] = ((const uint64_t *)s)[i];
          else for (size_t i = 0; i < NM_META_SPAD_BYTES; i++) d[i] = s[i];
        }
}

// C (M x N acc_t, row stride ldc) = D + A (M x K, row stride lda) * W, through sparse LOOP_WS calls of at most
// tile_I x tile_J x tile_K tiles (nm_loop_fits; nm_loop_tiles picks them), in tiled_matmul_outer's (gemmini.h) order
//   Bc       the n:m compressed W, (K/2) x N (row stride ldb), as nm_compress() makes it
//   D        bias (acc_t, row stride ldd) or NULL; repeating_bias: one row of N values for every row of C
//   idx_loop the index tiles in nm_loop_order_idx() order for the same tile_J / tile_K, 16 byte aligned
// requirements: K % (2*DIM) == 0 (nm_loop_pad_k / nm_zero_pad for other K); M and N may end in a partial tile.
// index rows are mvin'd by the loop itself (funct 27). ends with a dense config_ex and a fence, C is ready on return.
static void tiled_matmul_nm_loop(size_t M, size_t N, size_t K, const elem_t * A, size_t lda,
    const elem_t * Bc, size_t ldb, const acc_t * D, size_t ldd, bool repeating_bias,
    const uint8_t * idx_loop, acc_t * C, size_t ldc, size_t tile_I, size_t tile_J, size_t tile_K) {
  const size_t MT = (M + DIM - 1) / DIM, NT = (N + DIM - 1) / DIM, KT = K / DIM, KBT = KT / 2, tkb = tile_K / 2;
  NM_REQUIRE(M >= 1 && N >= 1 && K >= 2 * DIM && K % (2 * DIM) == 0);
  NM_REQUIRE(nm_loop_fits(tile_I, tile_J, tile_K));
  NM_REQUIRE(idx_loop != NULL && (uintptr_t)idx_loop % NM_LOAD_IDX_COLS == 0);
  const size_t I0 = (MT + tile_I - 1) / tile_I, J0 = (NT + tile_J - 1) / tile_J, K0 = (KT + tile_K - 1) / tile_K;
  const size_t pad_I_last = MT * DIM - M, pad_J_last = NT * DIM - N;
  const bool no_bias = D == NULL;

  gemmini_nm_reset();
  gemmini_nm_config_ex(1, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, 1);
  gemmini_extended_config_st(ldc * sizeof(acc_t), NO_ACTIVATION, ACC_SCALE_IDENTITY);
  gemmini_extended3_config_ld(lda * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(ldb * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
  gemmini_extended3_config_ld(repeating_bias ? 0 : (ldd * sizeof(acc_t)), MVIN_SCALE_IDENTITY, false, 2);

  for (size_t i0 = 0; i0 < I0; i0++)
    for (size_t j0 = 0; j0 < J0; j0++)
      for (size_t k0 = 0; k0 < K0; k0++) {
        const size_t I = i0 < I0 - 1 ? tile_I : MT - i0 * tile_I;
        const size_t J = j0 < J0 - 1 ? tile_J : NT - j0 * tile_J;
        const size_t Kt = k0 < K0 - 1 ? tile_K : KT - k0 * tile_K;
        const size_t pad_I = i0 == I0 - 1 ? pad_I_last : 0, pad_J = j0 == J0 - 1 ? pad_J_last : 0;
        const elem_t * a = A + i0 * tile_I * DIM * lda + k0 * tile_K * DIM;
        const elem_t * b = Bc + k0 * tkb * DIM * ldb + j0 * tile_J * DIM;
        // bias goes in with the first K block; the later ones accumulate on it
        const acc_t * d = (k0 == 0 && !no_bias) ? D + (repeating_bias ? 0 : i0 * tile_I * DIM) * ldd + j0 * tile_J * DIM : NULL;
        const acc_t * c = k0 == K0 - 1 ? C + i0 * tile_I * DIM * ldc + j0 * tile_J * DIM : NULL;
        const uint8_t * idx = idx_loop + (j0 * tile_J * KBT + J * k0 * tkb) * NM_META_SPAD_BYTES;
        gemmini_loop_ws_nm(I, J, Kt, pad_I, pad_J, 0, a, b, d, c, lda, ldb, repeating_bias ? 0 : ldd, ldc,
                           false, false, true, false, k0 != 0 || !no_bias, NO_ACTIVATION, 0, 0, false, idx);
      }

  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_fence();
}

#endif // GEMMINI_NM_H
