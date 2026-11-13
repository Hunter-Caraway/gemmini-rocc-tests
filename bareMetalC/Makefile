include $(abs_top_srcdir)/Makefrag

tests = \
	mvin_mvout \
	mvin_mvout_zeros \
	mvin_mvout_stride \
	mvin_mvout_block_stride \
	mvin_mvout_acc \
	mvin_mvout_acc_zero_stride \
	mvin_mvout_acc_stride \
	mvin_mvout_acc_full \
	mvin_mvout_acc_full_stride \
	mvin_mvout_spad \
	matmul_os \
	matmul_ws \
	matmul \
	matmul_spad \
	raw_hazard \
	aligned \
	padded \
	mvin_scale \
	conv \
	conv_stride \
	conv_rect \
	conv_rect_pool \
	conv_with_pool \
	conv_with_rot180 \
	conv_with_kernel_dilation \
	conv_with_input_dilation \
	conv_with_input_dilation_and_rot180 \
	conv_with_input_dilation_and_neg_padding \
	conv_trans_output_1203 \
	conv_trans_weight_1203 \
	conv_trans_weight_0132 \
	conv_trans_input_3120 \
	conv_trans_input_3120_with_kernel_dilation \
	conv_first_layer \
	conv_dw \
	conv_perf \
	conv_dw_perf \
	tiled_matmul_os \
	tiled_matmul_ws \
	tiled_matmul_ws_At \
	tiled_matmul_ws_Bt \
	tiled_matmul_ws_full_C \
	tiled_matmul_ws_low_D \
	tiled_matmul_ws_igelu \
	tiled_matmul_ws_layernorm \
	tiled_matmul_ws_softmax \
	tiled_matmul_ws_perf \
	tiled_matmul_cpu \
	tiled_matmul_option \
	transpose \
	matrix_add \
	resadd \
	resadd_stride \
	global_average \
	gemmini_counter \
	template \
	nm_sparse_sw \
	nm_sparse_matmul \
	nm_sparse_debug \
	nm_sparse_perf \
	nm_sparse_protocol \
	nm_sparse_spad \
	nm_sparse_loop \

tests_baremetal = $(tests:=-baremetal)

ifeq ($(findstring spike,$(RUNNER)),spike)
# Currently don't support conv or conv-with-pool on spike
runs_baremetal = $(addsuffix .run,$(filter-out conv-baremetal conv_with_pool-baremetal,$(tests_baremetal)))
else
# Don't run very long benchmarks for RTL sim
runs_baremetal = $(addsuffix .run,$(filter-out tiled_matmul_cpu-baremetal tiled_matmul_option-baremetal,$(tests_baremetal)))
endif
# Nornaml configs dont have ext spad
runs_baremetal := $(filter-out matmul_spad-baremetal.run mvin_mvout_spad-baremetal.run,$(runs_baremetal))

ifdef BAREMETAL_ONLY
	tests_linux =
	tests_pk =
else
	tests_linux = $(tests:=-linux)
	tests_pk = $(tests:=-pk)
endif

BENCH_COMMON = $(abs_top_srcdir)/riscv-tests/benchmarks/common
GEMMINI_HEADERS = $(abs_top_srcdir)/include/gemmini.h $(abs_top_srcdir)/include/gemmini_params.h $(abs_top_srcdir)/include/gemmini_testutils.h

CFLAGS := $(CFLAGS) \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=gnu99 \
	-O2 \
	-ffast-math \
	-fno-common \
	-fno-builtin-printf \
	-fno-tree-loop-distribute-patterns \
	-march=rv64gc -Wa,-march=rv64gc \
	-lm \
	-lgcc \
	-I$(abs_top_srcdir)/riscv-tests \
	-I$(abs_top_srcdir)/riscv-tests/env \
	-I$(abs_top_srcdir) \
	-I$(BENCH_COMMON) \
	-DID_STRING=$(ID_STRING) \
	-DPRINT_TILE=0 \

CFLAGS_PK := \
	$(CFLAGS) \
	-static \
	-DBAREMETAL=1 \

CFLAGS_BAREMETAL := \
	$(CFLAGS) \
	-nostdlib \
	-nostartfiles \
	-static \
	-T $(BENCH_COMMON)/test.ld \
	-DBAREMETAL=1 \

# changed: n:m pattern for the nm_sparse_* tests (gemmini_nm.h #errors if it disagrees with gemmini_params.h)
NM_N ?= 2
NM_M ?= 4
nm_sparse_%: CFLAGS += -DNM_N=$(NM_N) -DNM_M=$(NM_M)
nm_sparse_%: CFLAGS_PK += -DNM_N=$(NM_N) -DNM_M=$(NM_M)
nm_sparse_%: CFLAGS_BAREMETAL += -DNM_N=$(NM_N) -DNM_M=$(NM_M)

# changed: sparse-loop-fix, one program per illegal use of nm_sparse_loop.c (-DNEG=<n>); built by `make nm-loop-neg`
# only, kept out of `tests` because a NEG program passes by aborting
nm_loop_negs = 10 11 20 30 40 50 51 52 60 61 70 71 72 73 74 80 90 91 100 120 121 122 130 131 132 133
# changed: sparse-loop-fix, N11 / N14: dense Spike lib only (libgemmini NM_DENSE_BUILD=1). The dense RTL has no funct
# 27 / 28 hardware and no assert, and on an n:m build both commands are legal: an RTL NEG runner must skip these
nm_loop_negs_dense_spike_only = 110 140
tests_nm_loop_neg = $(patsubst %,nm_sparse_loop_neg%-baremetal,$(nm_loop_negs) $(nm_loop_negs_dense_spike_only))

all: $(tests_baremetal) $(tests_linux) $(tests_pk)

vpath %.c $(src_dir)

%-baremetal: %.c $(GEMMINI_HEADERS)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $< $(LFLAGS) -o $@ \
		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS)

# changed: sparse-loop-fix, the NEG programs of nm_sparse_loop.c
nm_sparse_loop_neg%-baremetal: nm_sparse_loop.c $(GEMMINI_HEADERS)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) -DNEG=$* $< $(LFLAGS) -o $@ \
		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS)

nm-loop-neg: $(tests_nm_loop_neg)

%-linux: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS) $< $(LFLAGS) -o $@

%-pk: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS_PK) $< $(LFLAGS) -o $@

run-baremetal: $(runs_baremetal)

%-baremetal.run: %-baremetal
	$(RUNNER)$(abs_top_srcdir)/build/bareMetalC/$^

junk += $(tests_baremetal) $(tests_linux) $(tests_pk) $(tests_nm_loop_neg) # changed: sparse-loop-fix, NEG programs

