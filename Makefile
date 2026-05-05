NVCC = nvcc

# ── 自动检测 GPU 架构 ──────────────────────────────────────────────
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

# ── 主示例编译目标 ──────────────────────────────────────────────────
TARGETS = 01_vector_add 02_matrix_mul 03_reduce 05_bank_conflict \
          06_coalescing 07_softmax 08_ncu_profiling 09_register_tiling \
          10_fused_kernel 11_warp_divergence 13_flash_attention \
          14_im2col_conv 15_wmma_gemm 16_streams 17_mixed_precision

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

17_mixed_precision: 17_mixed_precision/mixed_precision.cu
	$(NVCC) $(NVCC_FLAGS) -o $@/mixed_precision $<

# ── 练习题编译 ──────────────────────────────────────────────────────
# 编译所有练习题 (Level 1 + Level 2)
EXERCISE_DIRS = 01_vector_add/exercises 02_matrix_mul/exercises \
                03_reduce/exercises 05_bank_conflict/exercises \
                06_coalescing/exercises 07_softmax/exercises \
                08_ncu_profiling/exercises 09_register_tiling/exercises \
                10_fused_kernel/exercises 11_warp_divergence/exercises \
                13_flash_attention/exercises 14_im2col_conv/exercises \
                15_wmma_gemm/exercises 16_streams/exercises \
                17_mixed_precision/exercises

exercises:
	@echo "编译练习题..."
	@for dir in $(EXERCISE_DIRS); do \
		if [ -d $$dir ]; then \
			for f in $$dir/*.cu; do \
				if [ -f "$$f" ]; then \
					out="$${f%.cu}"; \
					echo "  $$f -> $$out"; \
					$(NVCC) $(NVCC_FLAGS) -o "$$out" "$$f" 2>&1 || echo "  (编译失败，可能依赖额外库)"; \
				fi; \
			done; \
		fi; \
	done
	@echo "练习题编译完成!"

# ── 调试练习编译 ────────────────────────────────────────────────────
debug:
	@echo "编译调试练习..."
	@for f in debug_exercises/bug*.cu; do \
		if [ -f "$$f" ]; then \
			out="$${f%.cu}"; \
			echo "  $$f -> $$out"; \
			$(NVCC) $(NVCC_FLAGS) -o "$$out" "$$f"; \
		fi; \
	done
	@echo "调试练习编译完成!"

# ── theory 练习题编译 ────────────────────────────────────────────────
theory_exercises:
	@echo "编译 theory 练习题..."
	@for f in theory/exercises/*.cu; do \
		if [ -f "$$f" ]; then \
			out="$${f%.cu}"; \
			echo "  $$f -> $$out"; \
			$(NVCC) $(NVCC_FLAGS) -o "$$out" "$$f" 2>&1 || echo "  (需要更高架构支持)"; \
		fi; \
	done
	@echo "Theory 练习题编译完成!"

# ── 全部编译 (主示例 + 练习) ────────────────────────────────────────
world: all exercises debug theory_exercises
	@echo "全部编译完成!"

# ── 清理 ────────────────────────────────────────────────────────────
clean:
	rm -f 01_vector_add/vector_add 02_matrix_mul/matmul 03_reduce/reduce \
	      05_bank_conflict/bank_conflict 06_coalescing/coalescing \
	      07_softmax/softmax 08_ncu_profiling/ncu_demo \
	      09_register_tiling/gemm_register 10_fused_kernel/fused_kernel \
	      11_warp_divergence/warp_divergence 13_flash_attention/flash_attention \
	      14_im2col_conv/im2col_conv 15_wmma_gemm/wmma_gemm \
	      16_streams/streams 17_mixed_precision/mixed_precision
	@echo "清理完成!"

clean_exercises:
	@for dir in $(EXERCISE_DIRS); do \
		if [ -d $$dir ]; then \
			for f in $$dir/*.cu; do \
				[ -f "$$f" ] && rm -f "$${f%.cu}" ; \
			done; \
		fi; \
	done
	@for f in debug_exercises/bug*.cu; do \
		[ -f "$$f" ] && rm -f "$${f%.cu}" ; \
	done
	@for f in theory/exercises/*.cu; do \
		[ -f "$$f" ] && rm -f "$${f%.cu}" ; \
	done
	@echo "练习题清理完成!"

distclean: clean clean_exercises

.PHONY: all exercises debug theory_exercises world clean clean_exercises distclean $(TARGETS)
