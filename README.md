# leanCuda — 从零开始的 CUDA 算子开发教程

一个**精简、实战驱动**的 CUDA 学习项目。  
边学概念、边写代码、边看性能数据——用最短路径掌握 GPU 算子开发。

**学完你能做什么:**
- 手写 CUDA kernel（向量加 → 矩阵乘 → Reduce → Softmax → LayerNorm）
- 用 Shared Memory + Warp Shuffle 优化性能
- 用 ncu 分析瓶颈（Memory/Compute/Latency Bound）
- 接入 PyTorch（autograd.Function + C++ binding）
- 理解混合精度和 Tensor Core 的工作原理

## 快速开始

- 会 C 语言（数组、指针、for 循环、malloc/free）
- 有一块 NVIDIA GPU（Volta / Turing / Ampere / Hopper 均可）
- 想学会自己写 CUDA kernel，而不是只调库

**新手从这里开始**：[`quickstart.md`](./quickstart.md)（10 分钟跑通第一个 CUDA 程序）→ [`tutorial.md`](./tutorial.md)（3-7 天完整学习路径）

```bash
# 编译所有示例（自动检测 GPU 架构）
make all

# 跑第一个 CUDA 程序
./01_vector_add/vector_add
```

## 怎么学

| 你的情况 | 从这里开始 |
|---------|-----------|
| 从没写过 CUDA | 直接读 [`tutorial.md`](./tutorial.md)，边学边做 |
| 能写简单 kernel，想写高性能算子 | [`LEARNING_PATH.md`](./LEARNING_PATH.md) → "不是新手"章节 |
| 做深度学习，想自定义 CUDA 算子 | [`tutorial.md` Part 5.4](./tutorial.md#54-动手接入-pytorch) → [`04_pytorch_extension/`](./04_pytorch_extension/) |
| 想理解 FlashAttention / Tensor Core 原理 | [`theory/06_tensor_core.md`](./theory/06_tensor_core.md) + [`theory/07_classic_operators.md`](./theory/07_classic_operators.md) |

**核心文件**:
- [`quickstart.md`](./quickstart.md) — 10 分钟快速上手（先跑通第一个程序）
- [`tutorial.md`](./tutorial.md) — 主线教程，3-7 天从零到完整算子开发
- [`LEARNING_PATH.md`](./LEARNING_PATH.md) — 所有文件索引 + 按需学习路径
- [`DEBUG_AND_OPTIMIZE.md`](./DEBUG_AND_OPTIMIZE.md) — 调试与优化手册（出错或变慢时翻）
- [`CHEAT_SHEET.md`](./CHEAT_SHEET.md) — CUDA 优化速查表（Roofline、ncu、内存层级）
- [`INTERVIEW_QUESTIONS.md`](./INTERVIEW_QUESTIONS.md) — 各章面试题（自检用）

## 编译要求

- NVIDIA GPU（Compute Capability ≥ 7.0，即 Volta 及以上）
- CUDA Toolkit ≥ 11.0，`nvcc` 在 PATH 中
- （可选）PyTorch + Python 3.8+（仅 `04_pytorch_extension/` 和 `12_layernorm_project/` 需要）

```bash
make all        # 编译全部
make 01_vector_add  # 只编译某个示例
make clean      # 清理
make all ARCH=sm_80  # 手动指定 GPU 架构
```

```
├── tutorial.md ......................... 主线教程（入口）
├── LEARNING_PATH.md .................... 学习路线 + 文件索引
├── DEBUG_AND_OPTIMIZE.md ............... 调试与优化手册
├── EXERCISE_ANSWERS.md ................. 练习题参考答案
├── Makefile ............................ 一键编译所有示例
│
├── 01_vector_add/ ...................... ⭐ 第一个 CUDA 程序
├── 02_matrix_mul/ ...................... ⭐ Shared Memory + Tiling
├── 03_reduce/ .......................... ⭐ 并行归约 7 版演进
├── 04_pytorch_extension/ ............... ⭐⭐ PyTorch 自定义算子
├── 05_bank_conflict/ ................... ⭐⭐ Bank Conflict 实验
├── 06_coalescing/ ...................... ⭐⭐ 合并访问实验
├── 07_softmax/ ......................... ⭐⭐ Softmax 3 版本
├── 08_ncu_profiling/ ................... ⭐⭐ ncu 性能分析实战
├── 09_register_tiling/ ................. ⭐⭐⭐ GEMM 寄存器分块
├── 10_fused_kernel/ .................... ⭐⭐ 算子融合对比
├── 11_warp_divergence/ ................. ⭐⭐ Warp 分歧实验
├── 12_layernorm_project/ ............... ⭐⭐⭐ 综合项目：手写 LayerNorm
├── 13_flash_attention/ ................. ⭐⭐⭐ 简化版 FlashAttention
├── 14_im2col_conv/ ..................... ⭐⭐ im2col 卷积
├── 15_wmma_gemm/ ....................... ⭐⭐⭐ Tensor Core WMMA
├── 16_streams/ ......................... ⭐⭐ CUDA Stream 与异步执行
├── 17_mixed_precision/ ................. ⭐⭐ 混合精度实战 (FP16/BF16/TF32)
│
├── debug_exercises/ .................... ⭐ 调试实战练习 (3 个有 bug 的程序)
│
└── theory/ ............................. 深度参考手册（9 章 + 附录）
    ├── 00_prerequisites.md ............. 术语与概念
    ├── 01_gpu_architecture.md .......... GPU 硬件架构
    ├── 02_cuda_programming_model.md .... CUDA 编程模型
    ├── 03_memory_hierarchy.md .......... 内存层级与优化
    ├── 04_warp_and_sync.md ............. Warp 与同步
    ├── 05_operator_development.md ...... 算子开发方法论
    ├── 06_tensor_core.md ............... Tensor Core 编程
    ├── 07_classic_operators.md ......... 经典算子剖析
    ├── 08_advanced_optimization.md ..... 高级优化技术
    ├── software_hardware_mapping.md .... 软硬件映射全景
    └── appendix_hardware.md ............ 半导体物理与封装
```