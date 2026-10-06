// n:m structured sparsity helpers for gemmini
// changed: new file, software side of the sparse-nm work
//
// weights w are k x j (same as gemmini's B in weight stationary).
// pruning runs down each column: every group of m rows keeps n values.
// compressed form is (k*n/m) x j values + same-shape index array,
// idx says which row inside the group (0..m-1) the value came from.
// kept values stay in increasing row order inside each group.

#ifndef GEMMINI_NM_H
#define GEMMINI_NM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "include/gemmini_params.h"

// default pattern, override with -DNM_N=4 -DNM_M=8 etc
#ifndef NM_N
#define NM_N 2
#endif
#ifndef NM_M
#define NM_M 4
#endif

// biggest group we bother supporting in the helpers
#define NM_MAX_M 16

typedef uint8_t nm_idx_t;

static inline int nm_abs(int x) { return x < 0 ? -x : x; }

// prune w in place so each m-group in each column keeps its n biggest (by magnitude)
// ties go to the lower row so the result is deterministic
static void nm_prune(size_t k, size_t j, elem_t * w, size_t stride, int n, int m) {
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

// squash pruned w (k x j) into wc ((k*n/m) x j) + idx (same shape as wc)
// groups with fewer than n nonzeros get padded with zeros, idx points at
// unused slots so the hardware just multiplies by 0
// k has to be a multiple of m
static void nm_compress(size_t k, size_t j, const elem_t * w, size_t w_stride,
    elem_t * wc, nm_idx_t * idx, size_t c_stride, int n, int m) {
  for (size_t c = 0; c < j; c++) {
    for (size_t g = 0; g < k / m; g++) {
      int out = 0;
      bool used[NM_MAX_M] = {0};
      for (int r = 0; r < m && out < n; r++) {
        elem_t v = w[(g * m + r) * w_stride + c];
        if (v != 0) {
          wc[(g * n + out) * c_stride + c] = v;
          idx[(g * n + out) * c_stride + c] = r;
          used[r] = true;
          out++;
        }
      }
      // pad, keep idx increasing so it still looks like a sorted pattern
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
  for (size_t g = 0; g < k / m; g++)
    for (int s = 0; s < n; s++)
      for (size_t c = 0; c < j; c++) {
        size_t p = (g * n + s) * c_stride + c;
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

// sparse reference, only walks the kept values - this is the math the
// sparse array does: each compressed weight grabs the activation its idx points at
static void nm_ref_sparse(size_t i, size_t k, size_t j, const elem_t * a, size_t a_stride,
    const elem_t * wc, const nm_idx_t * idx, size_t c_stride_w,
    acc_t * c, size_t c_stride, int n, int m) {
  size_t kc = k / m * n;
  for (size_t y = 0; y < i; y++)
    for (size_t x = 0; x < j; x++) {
      acc_t s = 0;
      for (size_t r = 0; r < kc; r++) {
        size_t p = r * c_stride_w + x;
        size_t g = r / n;
        s += (acc_t)a[y * a_stride + g * m + idx[p]] * (acc_t)wc[p];
      }
      c[y * c_stride + x] = s;
    }
}

// ---------------------------------------------------------------------------
// gemmini side of sparse mode
// changed: added sparse config + index loading wrappers
//
// how the sparse ws mode works:
//  - config_ex with rs1[10] set turns it on (ws only, no transposes)
//  - nm meta cmd (funct 26) stages index rows, call nm_load_meta before each preload
//  - preload(wc_tile, c) loads a DIM x DIM squashed weight tile and grabs the staged idx
//  - compute_preloaded(a_lo, a_hi) streams a dense activation tile that is 2*DIM wide,
//    a_lo = cols 0..DIM-1 and a_hi = cols DIM..2*DIM-1 (no bias slot in this mode)
// so one squashed DIM x DIM tile covers 2*DIM rows of the original weights

#include "include/gemmini.h"

#define k_NM_META 26

#define NM_CONFIG_EX_SPARSE_BIT 10

// same as gemmini_extended3_config_ex but with the sparse bit
#define gemmini_nm_config_ex(sparse, sys_act, sys_shift, sys_acc_scale, C_stride, A_stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)sys_acc_scale) << 32) | ((uint64_t)(A_stride) << 16) | ((uint64_t)(sparse) << NM_CONFIG_EX_SPARSE_BIT) | ((sys_act) << 3) | ((WEIGHT_STATIONARY) << 2) | CONFIG_EX, ((uint64_t)(C_stride) << 48) | (sys_shift), k_CONFIG)

static inline int nm_idx_bits(int m) {
  int b = 0;
  while ((1 << b) < m) b++;
  return b;
}

// pack one DIM x DIM tile of indices into meta cmds and send them
// packing: each idx is log2(m) bits, a row is DIM of them (col 0 lowest),
// each 64 bit reg takes as many whole rows as fit, rs1 first then rs2
static void nm_load_meta(const nm_idx_t * idx, size_t stride, int m) {
  const int ib = nm_idx_bits(m);
  const int row_bits = DIM * ib;
  const int rows_per_reg = 64 / row_bits;
  int r = 0;
  while (r < DIM) {
    uint64_t regs[2] = {0, 0};
    for (int ri = 0; ri < 2; ri++)
      for (int q = 0; q < rows_per_reg; q++, r++) {
        if (r >= DIM) continue;
        for (int c = 0; c < DIM; c++)
          regs[ri] |= (uint64_t)(idx[r * stride + c] & ((1 << ib) - 1)) << (q * row_bits + c * ib);
      }
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, regs[0], regs[1], k_NM_META);
  }
}

#endif // GEMMINI_NM_H
