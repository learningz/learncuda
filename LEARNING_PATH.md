# 学习路线

## 新手？从这里开始

**读 [`tutorial.md`](./tutorial.md)** — 这是一个从头到尾的线性教程，边学边做。
大约 3-7 天走完，不需要跳来跳去。

其他文件什么时候用：
- [`theory/`](./theory/) = 深度参考手册，tutorial 中遇到想深入的地方再翻
- [`DEBUG_AND_OPTIMIZE.md`](./DEBUG_AND_OPTIMIZE.md) = 代码出错或性能不好时翻

## GPU 架构兼容性

| 模块 | Volta (sm_70) | Turing (sm_75) | Ampere (sm_80) | Hopper (sm_90) | 说明 |
|------|:---:|:---:|:---:|:---:|------|
| 01-03 (基础) | ✓ | ✓ | ✓ | ✓ | 纯 CUDA Core，无特殊要求 |
| 04_pytorch_extension | ✓ | ✓ | ✓ | ✓ | 需要 PyTorch |
| 05_bank_conflict | ✓ | ✓ | ✓ | ✓ | SMEM 实验，通用 |
| 06_coalescing | ✓ | ✓ | ✓ | ✓ | 合并访问实验，通用 |
| 07_softmax | ✓ | ✓ | ✓ | ✓ | 纯 CUDA Core |
| 08_ncu_profiling | ✓ | ✓ | ✓ | ✓ | 需要 ncu 工具 |
| 09_register_tiling | ✓ | ✓ | ✓ | ✓ | 纯 CUDA Core 寄存器优化 |
| 10_fused_kernel | ✓ | ✓ | ✓ | ✓ | 纯 CUDA Core |
| 11_warp_divergence | ✓ | ✓ | ✓ | ✓ | 纯 CUDA Core |
| 12_layernorm_project | ✓ | ✓ | ✓ | ✓ | 需要 PyTorch |
| 13_flash_attention | ✓ | ✓ | ✓ | ✓ | 纯 CUDA Core (简化版) |
| 14_im2col_conv | ✓ | ✓ | ✓ | ✓ | 纯 CUDA Core |
| **15_wmma_gemm** | **✓** | **✓** | **✓** | **✓** | **至少 Volta (第一代 TC)** |
| 16_streams | ✓ | ✓ | ✓ | ✓ | CUDA Stream，通用 |
| **17_mixed_precision BF16** | **✗** | **✗** | **✓** | **✓** | **BF16 需要 Ampere+ (sm_80)** |
| theory/exercises WMMA 练习 | ✓ | ✓ | ✓ | ✓ | 至少 Volta |
| theory/exercises TF32 练习 | ✗ | ✗ | ✓ | ✓ | TF32 需要 Ampere+ |

> **注意**: `17_mixed_precision` 中的 BF16 部分和 `theory/exercises/ch06_ex2_tf32.cu` 需要 Ampere+，
> 其余所有代码都能在 Volta 及以上的 GPU 上编译运行。
- [`EXERCISE_ANSWERS.md`](./EXERCISE_ANSWERS.md) = 练习题卡住时翻


## 不是新手？按需查阅

**"我从没写过 CUDA"**
→ 从这里开始: [tutorial.md](./tutorial.md)（从零开始的线性教程）
→ 然后: [theory/01_gpu_architecture.md](./theory/01_gpu_architecture.md) → [theory/02_cuda_programming_model.md](./theory/02_cuda_programming_model.md) → [theory/03_memory_hierarchy.md](./theory/03_memory_hierarchy.md)（建立硬件直觉）

**"我能写简单 kernel，想写高性能算子"**
→ [theory/03_memory_hierarchy.md](./theory/03_memory_hierarchy.md)（内存优化）→ [theory/04_warp_and_sync.md](./theory/04_warp_and_sync.md)（Warp）→ [theory/05_operator_development.md](./theory/05_operator_development.md)（算子方法论）
→ 动手: [05_bank_conflict/](./05_bank_conflict/) → [06_coalescing/](./06_coalescing/) → [07_softmax/](./07_softmax/)
→ 做完每章的练习题

**"我做深度学习，想自定义 CUDA 算子"**
→ [tutorial.md Part 5.3](./tutorial.md#53-动手接入-pytorch) + [theory/05_operator_development.md](./theory/05_operator_development.md)（算子开发）+ [04_pytorch_extension/](./04_pytorch_extension/)
→ 然后: [theory/07_classic_operators.md](./theory/07_classic_operators.md)（经典算子）+ [07_softmax/](./07_softmax/)
→ 综合项目: [12_layernorm_project/](./12_layernorm_project/)（手写 LayerNorm 接入 PyTorch）

**"我想理解 FlashAttention / CUTLASS / cuBLAS 的原理"**
→ [theory/03_memory_hierarchy.md](./theory/03_memory_hierarchy.md) → [theory/05_operator_development.md](./theory/05_operator_development.md) → [theory/06_tensor_core.md](./theory/06_tensor_core.md)（Tensor Core）→ [theory/07_classic_operators.md](./theory/07_classic_operators.md)（FlashAttention）

**"我要做到极致优化 (>90% 硬件效率)"**
→ 全部读完，重点: [theory/06_tensor_core.md](./theory/06_tensor_core.md) → [theory/07_classic_operators.md](./theory/07_classic_operators.md) → [theory/08_advanced_optimization.md](./theory/08_advanced_optimization.md)
→ 做 Ch8 的练习题（ILP + 算子融合）
→ 然后读 CUTLASS 源码


## 完整路径 (从零到专家)

**阶段 1: 能跑 (1 天)**
- ⭐ [`tutorial.md`](./tutorial.md) Part 1 — 30 分钟跑通第一个 CUDA 程序
- ⭐ [`01_vector_add/`](./01_vector_add/) — 理解 kernel + 内存管理
- ⭐ [`02_matrix_mul/`](./02_matrix_mul/) — 理解 Shared Memory + 2D Grid
- ⭐ [`03_reduce/`](./03_reduce/) — 理解 Warp Shuffle

**阶段 2: 能写 (1 周)**
- ⭐⭐ [`theory/01_gpu_architecture.md`](./theory/01_gpu_architecture.md) (1.1, 1.5, 1.6) — SM 结构 + Slot 模型
- ⭐⭐ [`theory/02_cuda_programming_model.md`](./theory/02_cuda_programming_model.md) (2.1, 2.2, 2.6) — 线程层级 + Grid-Stride Loop
- ⭐⭐ → 做 Ch2 练习题（blockSize 实验 + 异步观察）
- ⭐⭐ [`theory/03_memory_hierarchy.md`](./theory/03_memory_hierarchy.md) (3.1-3.4) — 合并访问 + Shared Memory + Bank Conflict
- ⭐⭐ → 跑 [`05_bank_conflict/`](./05_bank_conflict/) 和 [`06_coalescing/`](./06_coalescing/) 实验
- ⭐⭐ → 做 Ch3 练习题（Roofline 分析 + padding）
- ⭐⭐ [`theory/04_warp_and_sync.md`](./theory/04_warp_and_sync.md) (4.1-4.2) — Warp 分歧 + Shuffle
- ⭐⭐ → 跑 [`11_warp_divergence/`](./11_warp_divergence/) 实验
- ⭐⭐ → 做 Ch4 练习题（Shuffle 变种 + 死锁演示）
- ⭐⭐ [`theory/05_operator_development.md`](./theory/05_operator_development.md) (5.1-5.3) — 算子分类 + 开发流程 + GELU 案例
- ⭐⭐ [`04_pytorch_extension/`](./04_pytorch_extension/) — 接入 PyTorch
- ⭐⭐ → 做 Ch5 练习题（Roofline + 算子融合）

**阶段 3: 能优化 (2-4 周)**
- ⭐⭐ [`theory/00_prerequisites.md`](./theory/00_prerequisites.md) — 术语参考（按需查阅）
- ⭐⭐ [`theory/03_memory_hierarchy.md`](./theory/03_memory_hierarchy.md) (3.7-3.9) — cp.async + Roofline + 优化决策树
- ⭐⭐ [`theory/04_warp_and_sync.md`](./theory/04_warp_and_sync.md) (4.3-4.5) — 同步机制 + 原子操作 + Occupancy
- ⭐⭐ [`theory/05_operator_development.md`](./theory/05_operator_development.md) (5.3-5.8) — Reduce 7版演进 + Scan + 数值稳定性
- ⭐⭐ [`theory/07_classic_operators.md`](./theory/07_classic_operators.md) (7.1-7.4) — Softmax/LayerNorm/FlashAttention/GEMM
- ⭐⭐ → 跑 [`07_softmax/`](./07_softmax/) 对比三版 Softmax
- ⭐⭐ → 做 Ch7 练习题（数值稳定性 + Warp-level 变种）
- ⭐⭐⭐ [`12_layernorm_project/`](./12_layernorm_project/) — 综合实战: 手写 LayerNorm + PyTorch 接入

**阶段 3.5: 异步与工程 (1 天)**
- ⭐⭐ [`16_streams/`](./16_streams/) — CUDA Stream 与异步执行
- ⭐⭐ [`DEBUG_AND_OPTIMIZE.md`](./DEBUG_AND_OPTIMIZE.md) — 调试手册 + CUDA_CHECK 最佳实践

**阶段 4: 能极致 (持续)**
- ⭐⭐⭐ [`theory/01_gpu_architecture.md`](./theory/01_gpu_architecture.md) (1.2-1.7) — HBM/NoC/MSHR/Scoreboard 完整硬件
- ⭐⭐⭐ [`theory/02_cuda_programming_model.md`](./theory/02_cuda_programming_model.md) (2.0) — Kernel Launch 全路径
- ⭐⭐⭐ [`theory/03_memory_hierarchy.md`](./theory/03_memory_hierarchy.md) (3.10-3.12) — L1/L2 微架构 + SASS 级分析
- ⭐⭐⭐ [`theory/04_warp_and_sync.md`](./theory/04_warp_and_sync.md) (4.6-4.8) — 内存一致性 + 汇合算法 + 原子SASS
- ⭐⭐⭐ [`theory/06_tensor_core.md`](./theory/06_tensor_core.md) — Tensor Core + CUTLASS
- ⭐⭐⭐ → 做 Ch6 练习题（WMMA + TF32 实验）
- ⭐⭐⭐ [`theory/08_advanced_optimization.md`](./theory/08_advanced_optimization.md) — ILP/持久化内核/量化/Multi-GPU
- ⭐⭐⭐ → 做 Ch8 练习题（ILP + 算子融合对比）
- ⭐⭐⭐ [`theory/appendix_hardware.md`](./theory/appendix_hardware.md) — 半导体物理（可选）


## 文件结构

```
leanCuda/ → 实际目录名为 learnCuda/
├── README.md ............................ 项目入口 (GitHub 首页)
├── tutorial.md .......................... ⭐ 主线教程 (从零到算子开发)
├── LEARNING_PATH.md ..................... 本文件 (学习路线)
├── DEBUG_AND_OPTIMIZE.md ................ ⭐⭐ 调试与优化手册 (含错误检查最佳实践)
├── EXERCISE_ANSWERS.md .................. 练习题参考答案 (Ch2-Ch8 全部)
├── Makefile ............................. 编译所有代码示例 (自动检测 GPU 架构)
│
│ ── 入门示例 ──
├── 01_vector_add/vector_add.cu .......... ⭐ 向量加法 (第一个 CUDA 程序)
├── 02_matrix_mul/matmul.cu .............. ⭐ 矩阵乘法 (朴素 vs Tiled)
├── 03_reduce/reduce.cu .................. ⭐ 并行归约 (V0-V6 七版演进)
├── 04_pytorch_extension/ ................ ⭐⭐ PyTorch 自定义 GELU 算子
│
│ ── 实验: 亲眼看到性能差异 ──
├── 05_bank_conflict/bank_conflict.cu .... ⭐⭐ Bank Conflict 对比 (stride 1/2/32)
├── 06_coalescing/coalescing.cu .......... ⭐⭐ 合并访问对比 (连续/跨步/随机)
├── 07_softmax/softmax.cu ................ ⭐⭐ Softmax 3版本 (3-pass/online/warp)
├── 08_ncu_profiling/ncu_demo.cu ......... ⭐⭐ ncu 实战: 3 个有"病"的 kernel + 输出解读
├── 09_register_tiling/gemm_register.cu .. ⭐⭐⭐ GEMM Register Blocking (4×4 thread tile)
├── 10_fused_kernel/fused_kernel.cu ...... ⭐⭐ 算子融合 3版对比 (未融合/融合/融合+vec4)
├── 11_warp_divergence/warp_divergence.cu  ⭐⭐ Warp 分歧对比 (有/无分歧/无分支)
│
│ ── 综合实战 ──
├── 12_layernorm_project/ ................ ⭐⭐⭐ 综合项目: 手写 LayerNorm + PyTorch
│   ├── README.md ........................ 项目说明、任务、提示、评估标准
│   ├── layernorm_starter.cu ............. Starter Code (填空完成)
│   ├── layernorm_solution.cu ............ 参考答案 (纯 CUDA 版)
│   ├── layernorm_cuda.cu ................ PyTorch C++ Binding (含详细注释)
│   ├── setup.py ......................... PyTorch 扩展编译配置
│   └── test_layernorm.py ................ 正确性 + 性能测试脚本
│
│ ── 进阶示例 ──
├── 13_flash_attention/ .................. ⭐⭐⭐ 简化版 FlashAttention
├── 14_im2col_conv/ ...................... ⭐⭐ im2col 卷积
├── 15_wmma_gemm/ ........................ ⭐⭐⭐ Tensor Core WMMA GEMM
├── 16_streams/ .......................... ⭐⭐ CUDA Stream 与异步执行
│   ├── streams.cu ....................... 3 个实验: 多Stream/Pinned Memory/Event计时
│   └── streams.md ....................... 异步执行原理 + 在 DL 框架中的应用
│
│ ── 调试练习 ──
├── debug_exercises/ ..................... ⭐ 3 个有 bug 的程序 (配合 DEBUG_AND_OPTIMIZE.md)
│   ├── bug1_vector_add.cu ............... cudaMemcpy 方向写反
│   ├── bug2_reduce.cu ................... __syncthreads 缺失
│   └── bug3_softmax.cu .................. exp 溢出 (数值稳定性)
│
│ ── 理论教程 (每章末尾有练习题) ──
└── theory/
    ├── 00_prerequisites.md .............. ⭐⭐ 术语与概念参考 [自检×3]
    ├── 01_gpu_architecture.md ........... ⭐⭐~⭐⭐⭐ GPU 硬件架构 [自检×3]
    ├── 02_cuda_programming_model.md ..... ⭐⭐~⭐⭐⭐ CUDA 编程模型 [练习×3]
    ├── 03_memory_hierarchy.md ........... ⭐⭐~⭐⭐⭐ 内存层级与优化 [练习×3]
    ├── 04_warp_and_sync.md .............. ⭐⭐~⭐⭐⭐ Warp 与同步 [练习×3]
    ├── 05_operator_development.md ....... ⭐⭐~⭐⭐⭐ 算子开发方法论 [练习×3]
    ├── 06_tensor_core.md ................ ⭐⭐⭐ Tensor Core 编程 [练习×2]
    ├── 07_classic_operators.md .......... ⭐⭐~⭐⭐⭐ 经典算子剖析 [练习×3]
    ├── 08_advanced_optimization.md ...... ⭐⭐⭐ 高级优化技术 [练习×2]
    ├── software_hardware_mapping.md ..... ⭐⭐ Grid/Block/Warp 到 SM/PB/Core 的完整映射
    └── appendix_hardware.md ............. ⭐⭐⭐ 半导体物理 + 封装技术 + 互连
```


## 学完之后，去哪里？

完成本教程后，你已经有能力手写 CUDA kernel 并接入 PyTorch。下一步可以：

**广度方向 (工程实践)**
- 研究 [CUTLASS](https://github.com/NVIDIA/cutlass) 的 GEMM 实现，理解 warp-level tiling 和 pipeline
- 阅读 [FlashAttention 官方实现](https://github.com/Dao-AILab/flash-attention)，对比你写的简化版
- 给 PyTorch 贡献算子 (`torch.utils.cpp_extension`)

**深度方向 (硬件理解)**
- 阅读 NVIDIA [白皮书](https://docs.nvidia.com/cuda/) (PTX ISA, CUDA C++ Programming Guide)
- 研究 [RasterGrid](https://github.com/nvidia/cutlass/blob/main/media/docs/efficient_gemm.md) 和 CUTLASS 的 swizzle 模式
- 用 ncu 分析 cuBLAS kernel，和自己写的做对比

**前沿方向**
- Hopper GPU 的新特性: TMA, FP8 Tensor Core, Thread Block Cluster
- 用 Triton 写 kernel (Python DSL, 自动调优)
- 关注 FlashAttention-3, FlexAttention 等最新论文

**社区资源**
- [NVIDIA Technical Blog](https://developer.nvidia.com/blog/)
- [CUDA MODE Discord](https://discord.gg/cuda-mode) (英文社区, 高手很多)
- [Bilibili — CUDA 相关教程](https://www.bilibili.com/) (中文视频教程)
- [Triton 文档](https://triton-lang.org/) (下一代 GPU 编程模型)
