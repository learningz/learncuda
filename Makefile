NVCC = nvcc

# ── 自动检测 GPU 架构 ──────────────────────────────────────────────
# 优先使用用户指定的 ARCH（例如 make all ARCH=sm_80）
# 否则调用 nvidia-smi 自动检测当前 GPU 的 Compute Capability
# 检测失败时回退到 sm_70（Volta，最低支持架构）
ifndef ARCH
  DETECTED_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
  ifneq ($(DETECTED_ARCH),)
    ARCH := sm_$(DETECTED_ARCH)
  else
    ARCH := sm_70
    $(warning 未检测到 GPU，使用默认架构 sm_70。手动指定: make all ARCH=sm_80)
  endif
endif

NVCC_FLAGS = -O2 -arch=$(ARCH)

# ── 编译目标 ────────────────────────────────────────────────────────
TARGETS = 01_vector_add 02_matrix_mul 03_reduce 05_bank_conflict \
          06_coalescing 07_softmax 08_ncu_profiling 09_register_tiling \
          10_fused_kernel 11_warp_divergence 13_flash_attention \
          14_im2col_conv 15_wmma_gemm 16_streams

all: $(TARGETS)
	@echo "编译完成! 架构: $(ARCH)"

01_vector_add: 01_vector_add/vector_add.cu
	$(NVCC) $(NVCC_FLAGS) -o $@/vector_add $<

02_matrix_mul: 02_matrix_mul/matmul.cu
	$(NVCC) $(NVCC_FLAGS) -o $@/matmul $<

03_reduce: 03_reduce/reduce.cu
	$(NVCC) $(NVCC_FLAGS) -o $@/reduce $<

05_bank_conflict: 05_bank_conflict/bank_conflict.cu
	$(NVCC) $(NVCC_FLAGS) -o $@/bank_conflict $<

06_coalescing: 06_coalescing/coalescing.cu
	$(NVCC) $(NVCC_FLAGS) -o $@/coalescing $<

07_softmax: 07_softmax/softmax.cu
	$(NVCC) $(NVCC_FLAGS) -o $@/softmax $<

08_ncu_profiling: 08_ncu_profiling/ncu_demo.cu
	$(NVCC) $(NVCC_FLAGS) -lineinfo -o $@/ncu_demo $<

09_register_tiling: 09_register_tiling/gemm_register.cu
	$(NVCC) $(NVCC_FLAGS) -o $@/gemm_register $<

10_fused_kernel: 10_fused_kernel/fused_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@/fused_kernel $<

11_warp_divergence: 11_warp_divergence/warp_divergence.cu
	$(NVCC) $(NVCC_FLAGS) -o $@/warp_divergence $<

13_flash_attention: 13_flash_attention/flash_attention.cu
	$(NVCC) $(NVCC_FLAGS) -o $@/flash_attention $<

14_im2col_conv: 14_im2col_conv/im2col_conv.cu
	$(NVCC) $(NVCC_FLAGS) -o $@/im2col_conv $<

15_wmma_gemm: 15_wmma_gemm/wmma_gemm.cu
	$(NVCC) $(NVCC_FLAGS) -o $@/wmma_gemm $<

16_streams: 16_streams/streams.cu
	$(NVCC) $(NVCC_FLAGS) -o $@/streams $<

# ── 清理 ────────────────────────────────────────────────────────────
clean:
	rm -f 01_vector_add/vector_add 02_matrix_mul/matmul 03_reduce/reduce \
	      05_bank_conflict/bank_conflict 06_coalescing/coalescing \
	      07_softmax/softmax 08_ncu_profiling/ncu_demo \
	      09_register_tiling/gemm_register 10_fused_kernel/fused_kernel \
	      11_warp_divergence/warp_divergence 13_flash_attention/flash_attention \
	      14_im2col_conv/im2col_conv 15_wmma_gemm/wmma_gemm \
	      16_streams/streams

.PHONY: all clean $(TARGETS)
