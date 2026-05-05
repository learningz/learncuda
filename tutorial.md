# CUDA 算子开发教程 — 从零开始的完整学习路径

这是一个**线性教程**：从头到尾跟着做，不需要跳来跳去。
每学一个概念，立刻写代码验证，看到效果，再学下一个。

> **和 theory/ 的关系**: theory/ 是深度参考手册（硬件细节、SASS 指令、Hopper 架构等）。
> 本教程是"主线剧情"，theory/ 是"百科全书"。主线中会用以下标注指引你：
>
> | 标注 | 含义 | 建议 |
> |------|------|------|
> | 📖 **推荐阅读** | 对理解当前内容有直接帮助 | 学完当前 Part 后就读 |
> | 🔍 **想深入？** | 硬件细节或高级话题 | 按兴趣选读，跳过不影响主线 |
>
> 你完全可以只读本教程就学会 CUDA 算子开发，等需要极致优化时再翻 theory/。

**前置要求**: 会 C 语言（数组、指针、for 循环、malloc/free）。有 NVIDIA GPU。
**总时长**: 按自己的节奏，大概 3-7 天可以完整走完。

## 一览表

```
 Part  时间     学什么
 ───  ────────  ─────────────────────────────────────────
 0    5 分钟    环境检查：nvidia-smi, nvcc, Compute Capability
 1    1-2 小时  你的第一个 CUDA kernel：线程编号、GPU 内存管理
 2    2-3 小时  Shared Memory Tiling：矩阵乘法加速、__syncthreads
 3    3-4 小时  Warp 执行模型：Warp Shuffle 归约、分支分歧
 4    3-5 小时  合并访问、Bank Conflict、Roofline、ncu 性能分析
 5    4-6 小时  Softmax 三版本、Online 算法、PyTorch 接入、算子融合
 6    6-10 小时 综合：手写 LayerNorm、Stream 异步、混合精度、进阶方向
```

每个 Part 末尾有 ✅ Checkpoint 帮你确认是否掌握，可以继续下一步。


---

# Part 0: 环境准备 — 5 分钟检查清单

在写第一行 CUDA 代码之前，确认你的环境能用。

## 0.1 你需要什么

```
1. NVIDIA GPU (Volta 及以上, 即 Compute Capability ≥ 7.0)
   包括: V100, T4, RTX 20/30/40 系列, A100, H100 等
   不支持: GTX 9xx 及更早, AMD GPU, Apple M 系列

2. NVIDIA 驱动 (Driver ≥ 470)

3. CUDA Toolkit ≥ 11.0 (提供 nvcc 编译器)

4. (可选) Python 3.8+ 和 PyTorch (仅 04_pytorch_extension 和 12_layernorm_project 需要)
```

## 0.2 一分钟验证

运行以下命令，全部通过就可以开始：

```bash
# 1. 检查 GPU 驱动和 GPU 型号
nvidia-smi
# 应该看到你的 GPU 型号和驱动版本
# 如果报错 "command not found" → 没装驱动

# 2. 检查 nvcc 编译器
nvcc --version
# 应该显示 CUDA 版本号 (如 "release 12.2")
# 如果报错 → CUDA Toolkit 没装或没加到 PATH

# 3. 检查 GPU 的 Compute Capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# 应该输出类似 "8.0" (A100) 或 "8.6" (RTX 3090)
# 如果 < 7.0 → 本教程的部分示例可能不兼容
```

## 0.3 常见安装问题

```
"nvcc: command not found"
  → CUDA Toolkit 没装, 或装了但没加 PATH
  → 解法: export PATH=/usr/local/cuda/bin:$PATH
  → 永久: 加到 ~/.bashrc 中

"no CUDA-capable device is detected"
  → 驱动没装, 或驱动版本和 CUDA Toolkit 不兼容
  → 解法: nvidia-smi 先确认驱动正常, 再确认版本兼容性
  → 参考: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/

"nvcc fatal: Unsupported gpu architecture 'compute_90'"
  → 你的 CUDA Toolkit 版本太老, 不认识新 GPU 架构
  → 解法: 升级 CUDA Toolkit, 或手动指定低一点的架构:
    make all ARCH=sm_80

在云服务器 (AWS/GCP/AutoDL/Colab) 上?
  → 通常已经预装好了, 直接 nvcc --version 和 nvidia-smi 验证即可
  → 如果用 Docker, 确保用的是 nvidia/cuda 基础镜像
```

一切就绪？开始写你的第一个 CUDA 程序。


---

# Part 1: 从 CPU 到 GPU — 你的第一个 CUDA 程序

> **学习目标**: 理解 CPU vs GPU 的设计差异，能写第一个 CUDA kernel，掌握线程编号公式和 5 步编程流程。
> **预估时间**: 1-2 小时
> **前置要求**: 会 C 语言，Part 0 环境验证通过

## 1.1 问题：这个 for 循环太慢了

假设你有一段 CPU 代码：

```c
void vector_add(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

n = 1 亿时，1 亿次加法串行执行 → ~0.1 秒。CPU 再快也是一个接一个。

**核心想法**：如果有 1 亿个"工人"，每人只加 1 对数字，全部同时开工？

CPU 做不到（最多几十个核心）。但 GPU 可以——它有数千个小核心，
能同时跑**几百万个线程**。

两者的设计哲学完全不同：

```
CPU (快但少):                    GPU (慢但多):
┌─────────┐                      ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐
│ 超强核心 │ ← 4-64 个            │ ││ ││ ││ ││ ││ ││ ││ │ ← 数千个
│  乱序执行 │    每个很快           │ ││ ││ ││ ││ ││ ││ ││ │    每个很慢
│  大缓存   │    善于复杂逻辑       │ ││ ││ ││ ││ ││ ││ ││ │    但胜在人多
└─────────┘                      └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘

好比: 1 位教授解 100 道题        好比: 100 个学生各解 1 道题
→ 聪明但只有 1 个人              → 每个人只需会简单操作
→ 适合复杂的串行逻辑             → 适合大量相同的并行计算
```

深度学习大多是"对百万个数做同样的运算"→ 天然适合 GPU。

## 1.2 把 for 循环拆成线程

CPU 版本的循环变量 `i` 从 0 到 n-1。GPU 版本不要循环——
启动 n 个线程，每个线程知道自己的编号 `i`，算 `c[i] = a[i] + b[i]`。

```cuda
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

这段代码有几个新东西，逐个解释：

**`__global__`** = "这个函数在 GPU 上执行，由 CPU 调用"。
这样的函数叫 **kernel**。

**`threadIdx.x`** = "我在小组内是第几号"（0 到 255）。
**`blockIdx.x`** = "我的小组是第几组"（0 到 gridSize-1）。
**`blockDim.x`** = "每组有几个人"（= 256）。

为什么要分"小组"（Block）？因为硬件规定每组最多 1024 人。
1 亿个线程 → 分成 ~39 万个 Block，每个 256 人。

**`if (i < n)`** = 边界检查。总线程数可能比 n 多一点（向上取整），多余的不能干活。

## 1.3 动手：编译运行

> **编译说明**：本教程统一用项目根目录的 `Makefile` 编译所有示例。
> 你只需要记住三条命令：
>
> | 命令 | 作用 |
> |------|------|
> | `make all` | 编译全部 15 个示例 |
> | `make 01_vector_add` | 只编译某个示例 |
> | `make clean` | 清理所有编译产物 |
>
> Makefile 会自动通过 `nvidia-smi` 检测你的 GPU 架构（如 sm_80 = A100, sm_86 = RTX 3090）。
> 检测失败时回退到 sm_70（Volta）。你也可以手动指定：`make all ARCH=sm_80`。
>
> 编译选项：默认 `-O2`（优化级别 2）+ 自动检测的 `-arch=sm_XX`。
> `08_ncu_profiling` 额外加了 `-lineinfo`（保留行号信息，方便 ncu 定位到源码行）。
>
> 如果你想手动编译单个文件（不用 Makefile），每节都会给出对应的 `nvcc` 命令。

```bash
make 01_vector_add      # 或者手动: cd 01_vector_add && nvcc -O2 -o vector_add vector_add.cu
./01_vector_add/vector_add
```

你应该看到正确性通过的输出（不同版本的提示文字可能略有差异）。

**现在打开 [`01_vector_add/vector_add.cu`](./01_vector_add/vector_add.cu)**，通读一遍。代码的 5 步流程：

```
1. CPU 上 malloc + 填数据
2. GPU 上 cudaMalloc (GPU 有自己的独立内存叫"显存")
3. cudaMemcpy: CPU 内存 → GPU 显存 (数据搬过去)
4. kernel<<<gridSize, blockSize>>>(): GPU 计算
5. cudaMemcpy: GPU 显存 → CPU 内存 (结果搬回来)
```

**为什么要搬来搬去？** 因为 CPU 内存和 GPU 显存是物理上分开的两块芯片。
GPU 只能读写自己的显存。就像你和同事在不同城市，要用快递传文件。

> 🔍 **想深入？** 理解编译过程（.cu → PTX → SASS）和 kernel launch 的硬件路径？
> 看 [`01_vector_add/vector_add.md`](./01_vector_add/vector_add.md) 和 [`theory/02_cuda_programming_model.md`](./theory/02_cuda_programming_model.md) 2.0 节。

## 1.4 动手实验（改着玩）

```
1. 把 blockSize 从 256 改成 128 → gridSize 变大了 (需要更多组)
2. 把 N 改成 100 → 只有 1 个 Block, 256 线程中只有 100 个干活
3. 去掉 if (i < n) → 可能崩溃! (越界访问)
```

**你现在掌握了**: kernel 写法、线程编号、GPU 内存管理。

**想巩固？** 做 [`01_vector_add/exercises/`](./01_vector_add/exercises/) 里的 3 道练习题（SAXPY / ReLU / FMA），每道都有 Level 1（只填 kernel）和 Level 2（host 端也要自己写）两个版本。

### ✅ Checkpoint: Part 1

```
在继续之前，确认你理解了:

□ __global__ 函数在 GPU 上执行, 由 CPU 调用
□ threadIdx.x + blockIdx.x * blockDim.x = 全局线程编号
□ CPU 内存和 GPU 显存是分开的, 需要 cudaMemcpy 搬运
□ if (i < n) 是必须的 (多余线程不能越界!)

能回答这个问题吗:
  如果 N=1000, blockSize=256, gridSize=4, 总线程数=1024:
  第 1000-1023 号线程在干什么? (答: 什么都不干, 被 if 跳过了)
```


---

# Part 2: 让它更快 — Shared Memory 和矩阵乘法

> **学习目标**: 理解 GPU 内存层级，能写 Tiled 矩阵乘法，理解 `__syncthreads()` 的必要性和 Shared Memory 的数据复用优势。
> **预估时间**: 2-3 小时
> **前置要求**: Part 1 (能写基础 kernel + 理解线程编号)

## 2.1 向量加法的瓶颈是什么？

向量加法每个元素只做 1 次加法，但要从显存读 2 个数、写 1 个数。
显存很大（80GB）但很慢（每次读要等 ~500 个时钟周期）。

计算只要 1 个周期，等数据要 500 个周期 → **99% 的时间在等数据！**

这叫 **Memory Bound**：性能被内存速度限制，计算单元大部分时间空闲。

GPU 内部其实有好几层存储，越快的越小、越慢的越大：

```
              容量        延迟        谁能访问
            ─────────  ──────────  ──────────────
寄存器       ~256KB/SM   0 周期     只有本线程
  ↓
Shared Mem   ~100KB/SM   ~5 周期    同一 Block 内共享
  ↓
L1 Cache     128KB/SM    ~28 周期   硬件自动管理
  ↓
L2 Cache     ~40MB       ~200 周期  所有 SM 共享
  ↓
HBM (显存)   80GB        ~500 周期  所有 SM 共享 (但很慢!)

寄存器 > Shared Memory >> L1 >> L2 >>>> HBM

好比: 寄存器 = 手里拿着的东西   (随拿随用)
      Shared Mem = 桌上的工具    (伸手就到)
      HBM = 仓库里的货物          (要派人去取, 来回很久)
```

写 CUDA 的核心技巧就是：**尽量让数据待在靠近计算单元的快存储里**。
这就是接下来我们要学的 Shared Memory 和 Tiling 的动机。

向量加法无法进一步优化（每个元素必须读一次写一次，已经是最少了）。
但有些算法的数据可以**重复使用**——比如矩阵乘法。

## 2.2 矩阵乘法的数据复用机会

```
C[i][j] = Σ(k) A[i][k] × B[k][j]

计算 C 的第 0 行: 需要 A 的第 0 行 (读 1 次) × B 的每一列
计算 C 的第 1 行: 需要 A 的第 1 行 (读 1 次) × B 的每一列 (和上面相同!)

→ B 的同一列被 C 的每一行都需要 → 大量重复读取!
```

朴素做法：每个线程独立从显存读 A 的一行和 B 的一列。
相邻线程需要 B 的同一列 → 从慢显存重复读同一份数据！

## 2.3 Shared Memory — Block 内的快速共享存储

GPU 的每个 SM（计算单元）内部有一小块超快的存储叫 **Shared Memory**：
- 容量小（~100KB），但访问快（5 个时钟周期，比显存快 100×）
- 同一 Block 的所有线程可以共享读写
- 不同 Block 之间不可见

**Tiling 思路**：
1. Block 内所有线程协作，把一小块 A 和 B 从显存搬到 Shared Memory（搬 1 次）
2. 大家从 Shared Memory 里读（读很多次，但每次只要 5 周期）
3. 显存读取次数减少 ~16 倍！

```cuda
__shared__ float tile_A[16][16];  // 在 Shared Memory 中声明
tile_A[threadIdx.y][threadIdx.x] = A[row][k];  // 每人搬 1 个元素
__syncthreads();  // 等所有人搬完!
// 现在从 tile_A 读, 而不是从显存读
```

**`__syncthreads()`** = "Block 内所有线程到这里集合，都到了再继续"。
不加这个 → 你可能读到别人还没搬完的数据 → 结果错误。

## 2.4 动手：跑矩阵乘法，看加速比

```bash
make 02_matrix_mul      # 或者手动: cd 02_matrix_mul && nvcc -O2 -o matmul matmul.cu
./02_matrix_mul/matmul
```

你会看到朴素版和 Tiled 版的耗时和 GFLOPS。**Tiled 版快多少？**

**打开 [`02_matrix_mul/matmul.cu`](./02_matrix_mul/matmul.cu)**，对比两个 kernel：
- `matmul_naive`：每次从显存读 → 慢
- `matmul_tiled`：先搬到 Shared Memory 再读 → 快

Tiled 版的核心循环逐行解读：

```cuda
// 沿 K 维度循环，每次处理 TILE_SIZE=16 个 K
for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {

    // 步骤 1: 每个线程搬运 1 个元素到 Shared Memory
    // 256 线程同时搬 → 一次搬完 16×16 = 256 个元素
    As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
    
    // 步骤 2: 等! 所有人搬完了吗?
    __syncthreads();
    // 如果不等: 线程 0 可能已经在读 As[0][5], 但线程 5 还没写完 As[0][5]!
    
    // 步骤 3: 从 Shared Memory 读 (快!) 并计算
    for (int i = 0; i < TILE_SIZE; i++) {
        // 这两次读都是从 Shared Memory, 延迟只有 5 cycles (vs 显存 500 cycles)
        // 而且同一份 Bs[i][threadIdx.x] 被同一列的所有线程共用!
        sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }
    
    // 步骤 4: 再等一次, 确保大家都算完了, 再覆盖 Shared Memory 搬下一块
    __syncthreads();
}
```

**数据复用的量化**：
```
不用 Shared Memory: 每个线程独立读 A 的一行 (K 次) + B 的一列 (K 次) = 2K 次显存访问
用 Shared Memory: 每 TILE_SIZE 步, 整个 Block 只从显存读 2×16×16 = 512 次
                  然后每线程从 Shared Memory 读 16 次 (快 100×)
                  显存访问减少 ~TILE_SIZE = 16 倍!
```

### ✅ Checkpoint: Part 2

```
在继续之前，确认你理解了:

□ Shared Memory 和全局显存的区别 (容量 vs 速度)
□ 为什么 __syncthreads() 不能省
□ Tiling 的核心思想: 搬 1 次到快内存, 读 N 次从快内存

动手验证:
  去掉 matmul_tiled 中的第一个 __syncthreads()
  重新编译运行 → 结果还对吗? (大概率不对!)
  → 因为某些线程可能在别人写完 Shared Memory 前就开始读了
```

### 2.4.5 Shared Memory 的隐藏陷阱: Bank Conflict

在你为 Tiled 矩阵乘法的性能提升感到高兴之前，有一个细节需要知道：
Shared Memory 在物理上被切成了 **32 个 Bank**（每个 4 字节宽）。

```
地址到 Bank 的映射:
  smem[0]  → addr 0  → Bank 0
  smem[1]  → addr 4  → Bank 1
  ...
  smem[31] → addr 124 → Bank 31
  smem[32] → addr 128 → Bank 0  ← 又回到 Bank 0!

当同一 Warp 的多个线程访问同一个 Bank 的不同地址时:
  2 线程 → 2-way conflict → 2 cycles (慢了 2×)
  32 线程全打在一个 Bank → 32-way conflict → 32 cycles (慢了 32×!)
  
但如果多个线程访问 Bank 的**同一地址** → 硬件广播 → 只有 1 cycle (没问题!)
```

在矩阵乘法中，我们用的 `As[threadIdx.y][threadIdx.x]` 访问模式：
- 同一 Warp 的 `threadIdx.x` 连续 → 访问连续地址 → 不同 Bank → 无冲突。

但其他访问模式（如转置: `As[threadIdx.x][threadIdx.y]`）会引发 Bank Conflict。
你将在 Part 4 的专项实验中看到这个问题。

> 📖 **推荐阅读**: Shared Memory 的硬件结构（32 个 Bank、怎么分配、Bank Conflict 是什么）——
> 看 [`02_matrix_mul/matmul.md`](./02_matrix_mul/matmul.md) 和 [`theory/03_memory_hierarchy.md`](./theory/03_memory_hierarchy.md) 3.3 节。
> 理解 Bank 结构会让你在 Part 4 的 Bank Conflict 实验中收获更多。

## 2.5 动手实验

```
1. 把 TILE_SIZE 从 16 改成 32 → 性能有变化吗？
2. 矩阵从 512 改成 2048 → 加速比变大了 (矩阵越大, tiling 收益越高)
3. 去掉一个 __syncthreads() → 结果可能出错! (数据竞争)
```

**你现在掌握了**: Shared Memory、`__syncthreads()`、Tiling 优化。

**想巩固？** 做 [`02_matrix_mul/exercises/`](./02_matrix_mul/exercises/) 里的 3 道练习题（矩阵转置 / 矩阵加法 / GEMV 归约），每道都有 Level 1（只填 kernel）和 Level 2（host 端也要自己写）两个版本。

> **过渡思考**：Shared Memory 让我们加速了内存访问。但你有没有注意到——
> `__syncthreads()` 本身也有开销？每次调用，所有线程都要停下来等。
> 有没有办法不用 `__syncthreads()` 就能让线程通信？有——Warp Shuffle。

**`__syncthreads()` 到底有多大的开销？**

```
__syncthreads() 编译为 BAR.SYNC 指令。

延迟: ~20-100 cycles, 取决于 Block 中的 Warp 数量。
  8 个 Warp → ~20 cycles  (只需等 7 个 Warp 到达 barrier)
  16 个 Warp → ~40 cycles
  32 个 Warp → ~80 cycles

为什么有开销？
  1. 每个到达的 Warp 设一个标志位 → 最后一个到达后广播给所有 Warp
  2. Barrier 硬件是共享资源, 多个 Block 可能排队等用
  3. 每个 Warp 从 Eligible 变成 Stalled (等 barrier), 再变回 Eligible
     → Warp Scheduler 在此期间只能选其他 Warp, 可能选不到 → PB 空闲

但 __syncthreads 的主要代价不在这 ~20-100 cycles:
  真正贵的代价是它强制"同步点"——所有 Warp 必须到达同一点。
  如果一个 Warp 先完成前面的工作 → 必须在 BAR 处空转等慢的 Warp。
  这就是 "Stall Barrier" (ncu 中的主要 stall 原因之一)。

降低 BAR 开销:
  1. 减少 __syncthreads 次数: 能用 Warp Shuffle 就不要用 SMEM+sync
  2. 减少 Block 中的 Warp 数: blockDim ≤ 128 (4 Warp) vs 1024 (32 Warp)
  3. 把 __syncthreads 放在相同的位置 (对齐 barrier 到达时间)
  4. 用 Cooperative Groups 替代原始 syncthreads (编译器可以做更好的优化)
```

这就是为什么 Part 3 要学 Warp Shuffle: 同一个 Warp 内的线程通信完全不需要 barrier。


### 🏋 过渡练习: SMEM 矩阵转置

在进入 Part 3 之前，用 Shared Memory 做一个简单但经典的练习巩固理解：

```
目标: 写一个矩阵转置 kernel (A[i][j] → B[j][i])
要求: 
  1. 先用朴素版本 (直接从 Global Memory 读, 非合并写)
  2. 再用 SMEM 版本 (先合并读到 SMEM, 再从 SMEM 合并写到 Global)

对比:
  朴素版: 读是合并的 (连续行), 写是非合并的 (跨行跳列)
  SMEM 版: 读和写都是合并的

预期: SMEM 版本有效带宽显著更高 (特别是转置写避免了跨步访问)

练习代码: 02_matrix_mul/exercises/ex1_transpose_level1.cu (只填 kernel)
                         ex1_transpose_level2.cu (完整实现)
```

这个练习帮你巩固 Shared Memory 的读写模式, 为 Part 3 的 Warp Shuffle 做准备。
关键在于体会: SMEM 不仅是数据复用, 还可以用来 **重排访问模式**。


---

# Part 3: Warp 的秘密 — 32 线程为一组

> **学习目标**: 理解 Warp 的执行模型，能用 Warp Shuffle 做线程间通信，理解 Warp Divergence 的代价。
> **预估时间**: 3-4 小时
> **前置要求**: Part 2 (Shared Memory + `__syncthreads()`)

## 3.1 GPU 不是逐个线程执行的

你已经知道线程被分成 Block。但 Block 内部还有更底层的分组：

```
Block (256 线程)
├── Warp 0: Thread   0 ~ 31   (这 32 个线程每时每刻执行同一条指令)
├── Warp 1: Thread  32 ~ 63
├── ...
└── Warp 7: Thread 224 ~ 255
```

**Warp** = 32 个连续线程的一组。GPU 硬件以 Warp 为单位调度和执行。
同一 Warp 的 32 线程在同一时刻执行同一条指令——只是各自操作不同的数据。

这有一个重要后果：**如果同一 Warp 的线程走了不同的 if/else 分支**，
GPU 无法让它们同时做不同的事——一个 Warp 只有一套指令指针（PC）和指令解码器。

假设 if/else 分支各有一半线程：
```
__global__ void divergent_kernel(float *a, int n) {
    int tid = threadIdx.x;
    if (tid % 2 == 0)
        a[tid] = expensive_op_1(a[tid]);  // 偶数线程走这里
    else
        a[tid] = expensive_op_2(a[tid]);  // 奇数线程走这里
}
```

GPU 的执行过程（SIMT — 单指令多线程）：

```
Step 1: 执行 if 分支
  ┌──────────────────────────────────────────────────────┐
  │ 偶数线程 (0,2,4,...,30): 执行 expensive_op_1         │
  │ 奇数线程 (1,3,5,...,31): 被 mask 掉, 空转等待!      │
  │ 两个分支都暴露在同一个 Warp 中 → 一半算力闲置        │
  └──────────────────────────────────────────────────────┘

Step 2: 执行 else 分支
  ┌──────────────────────────────────────────────────────┐
  │ 偶数线程: 被 mask 掉, 空转等待                       │
  │ 奇数线程: 执行 expensive_op_2                        │
  └──────────────────────────────────────────────────────┘

总时间 = time(expensive_op_1) + time(expensive_op_2)
而不是 max(time1, time2)
```

关键：`tid % 2` 导致相邻线程走不同路径 → 同一个 Warp 内分支不同 → 串行执行。

**分界线是 Warp 边界，不是 Block 边界**：
```
// ✓ 无分歧: 所有偶数线程在一个 Warp, 所有奇数线程在另一个 Warp
if (threadIdx.x / 32 == 0)  // Warp 0 走 if, Warp 1 走 else
    foo();
else
    bar();

// ✗ 有分歧: 相邻线程走不同路径 (同一 Warp 内)
if (threadIdx.x % 2 == 0)   // 线程 0 走 if, 线程 1 走 else → 同一 Warp!
    foo();
else
    bar();
```

这和你前面学的 Bank Conflict 不同：
- **Bank Conflict**: 多个线程同时打在同一 Bank → 串行化（延迟叠加）
- **Warp Divergence**: 同一 Warp 的线程走了不同分支 → 路径串行（时间叠加）

两者都让程序变慢，但解决方式不同：
Bank Conflict 靠改变数据布局（padding），Warp Divergence 靠改变线程逻辑（让 Warp 边界对齐分支边界）。

## 3.2 Warp Shuffle — 线程间直接交换数据

同一 Warp 的 32 个线程可以**直接读取彼此的变量值**，不需要经过 Shared Memory：

```cuda
float neighbor_val = __shfl_down_sync(0xffffffff, my_val, 16);
// 我拿到了 "编号比我大 16" 的线程的 my_val
```

这比 Shared Memory 更快（~1 周期 vs ~5 周期），而且不需要 `__syncthreads()`。

**经典用途：Warp 内快速求和**

```
32 个值 → shfl_down 16 → 16 个部分和
        → shfl_down 8  → 8 个
        → shfl_down 4  → 4 个
        → shfl_down 2  → 2 个
        → shfl_down 1  → 1 个总和 (5 步搞定!)
```

## 3.3 动手：跑归约，看 Shuffle 的威力

```bash
make 03_reduce          # 或者手动: cd 03_reduce && nvcc -O2 -o reduce reduce.cu
./03_reduce/reduce
```

对比 V0（朴素交错）到 V6（atomicAdd 单 kernel）的 7 个版本——你会看到
每一步优化背后的硬件原理。

**打开 [`03_reduce/reduce.cu`](./03_reduce/reduce.cu)**，重点看 `warp_reduce_sum` 函数：

```cuda
__device__ float warp_reduce_sum(float val) {
    // 轮 1: 每个线程加上 "编号大 16" 的线程的值
    //   Thread 0:  val = val + Thread_16.val
    //   Thread 1:  val = val + Thread_17.val
    //   ...Thread 15: val = val + Thread_31.val
    //   → 32 个值变成 16 个部分和 (在 Thread 0-15)
    val += __shfl_down_sync(0xffffffff, val, 16);
    
    // 轮 2: 每个线程加上 "编号大 8" 的线程的值
    //   → 16 个部分和变成 8 个 (在 Thread 0-7)
    val += __shfl_down_sync(0xffffffff, val, 8);
    
    // 轮 3-5: 继续减半
    val += __shfl_down_sync(0xffffffff, val, 4);  // 8 → 4
    val += __shfl_down_sync(0xffffffff, val, 2);  // 4 → 2
    val += __shfl_down_sync(0xffffffff, val, 1);  // 2 → 1
    
    // Thread 0 现在持有整个 Warp (32 线程) 的总和!
    return val;
}
```

**`0xffffffff`** 是参与的线程掩码：32 位全 1 = 所有 32 个线程都参与。

**为什么比 Shared Memory 归约快？**
- Shared Memory 归约：写 SMEM → `__syncthreads()` → 读 SMEM → 写 SMEM → ... (8 轮!)
- Warp Shuffle：直接在寄存器间传值，不需要 SMEM，不需要 `__syncthreads()`，5 轮搞定。

然后看 V3（Warp Shuffle 版）怎么把 Warp 级归约组合成 Block 级归约：
```
阶段 1: 每个 Warp 内部用 Shuffle 归约 (32 → 1)
        8 个 Warp → 8 个部分和
阶段 2: 8 个部分和写入 Shared Memory (只需 8 个 float!)
        __syncthreads()  (只有 1 次!)
阶段 3: 第一个 Warp 读出 8 个值, 再做一次 Shuffle 归约 (8 → 1)
```

7 个版本的优化路径一目了然：
```
V0 → V1: 消除 Warp Divergence (让连续线程工作, 而不是间隔线程)
V1 → V2: Grid-Stride Loop (每线程处理多元素, 减少 Block 数)
V2 → V3: Warp Shuffle (Warp 内归约零同步, __syncthreads 从 8 次降到 1 次)
V3 → V4: 循环展开 4× (ILP: 4 条加载背靠背发射, 隐藏内存延迟)
V4 → V5: float4 向量化 (1 条 LDG.128 替代 4 条 LDG.32, 指令数减 4×)
V5 → V6: atomicAdd (Block 结果直接累加到全局, 无需 CPU 汇总)
```

> 每个版本只改一个东西，性能提升的来源清清楚楚。
> 这就是 GPU 优化的核心方法论：**一次只改一个变量，对比前后数据**。

### ✅ Checkpoint: Part 3

```
在继续之前，确认你理解了:

□ Warp = 32 个线程一组, 同时执行同一条指令
□ __shfl_down_sync: 从编号更大的线程取值, 不经过内存
□ 分支分歧: 同一 Warp 走不同 if/else → 两条路径串行 → 性能减半

动手验证:
  运行 11_warp_divergence/, 观察 "无分歧" vs "50%分歧" 的耗时差异
  然后把两条路径各加长到 10 次乘法 → 分歧的代价更明显了吗?
```

**想巩固？** 做 [`03_reduce/exercises/`](./03_reduce/exercises/) 里的 3 道练习题（求最大值 / 点积 / 条件计数），每道都有 Level 1 和 Level 2。

> **过渡思考**：到目前为止，我们学了三种加速手段——并行（Part 1）、
> Shared Memory 复用（Part 2）、Warp Shuffle（Part 3）。但你可能已经发现，
> 归约和矩阵乘的"有效带宽"远低于 GPU 标称的 2TB/s。
> 为什么？因为内存访问**模式**也会极大影响性能——这就是 Part 4 的主题。


---

# Part 4: 内存是瓶颈 — 合并访问和性能分析

> **学习目标**: 理解合并访问和 Bank Conflict 的硬件原理，能用 Roofline 模型判断 kernel 瓶颈，会用 ncu 分析性能。
> **预估时间**: 3-5 小时
> **前置要求**: Part 3 (Warp Shuffle + Divergence)

## 4.1 为什么内存访问模式这么重要？

GPU 的显存带宽很高（例如 A100 约 2TB/s，具体数值随型号和配置而变），但有一个条件：
**同一 Warp 的 32 个线程必须访问连续的地址**，才能合并成 1 次传输。

如果 32 线程的地址是随机的 → 每人各发一次请求 → 带宽利用率暴跌到 3%。

## 4.2 动手：亲眼看到合并 vs 不合并的差异

```bash
make 06_coalescing      # 或者手动: cd 06_coalescing && nvcc -O2 -o coalescing coalescing.cu
./06_coalescing/coalescing
```

看三种模式的有效带宽：连续（高）、跨步（低）、随机（极低）。

## 4.3 动手：亲眼看到 Bank Conflict

```bash
make 05_bank_conflict   # 或者手动: cd 05_bank_conflict && nvcc -O2 -o bank_conflict bank_conflict.cu
./05_bank_conflict/bank_conflict
```

Shared Memory 也有类似的"访问模式决定性能"的问题，叫 Bank Conflict。

> 📖 **推荐阅读**: [`05_bank_conflict/bank_conflict.md`](./05_bank_conflict/bank_conflict.md)（32 个 Bank 的硬件结构图）
> 和 [`theory/03_memory_hierarchy.md`](./theory/03_memory_hierarchy.md) 3.3 节。理解 Bank 冲突的硬件原理，才能在实际开发中避免它。

## 4.4 Roofline 模型 — 快速判断瓶颈

给你一个 kernel，怎么 3 秒钟判断它是被内存还是被计算限制？用 Roofline 模型：

```
算术强度 (AI) = 计算量 (FLOP) ÷ 数据搬运量 (Byte)

性能
(GFLOPS)
  │
  │          ╱ Compute Bound 天花板 (如 A100: 19.5 TFLOPS)
  │─ ─ ─ ─╱─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  │      ╱│
  │    ╱  │
  │  ╱    │← Ridge Point (平衡点)
  │╱      │   = 峰值计算 ÷ 峰值带宽
  │       │   = 19.5T / 2T = 9.7 FLOP/Byte
  │       │
  └───────┴──────────────── 算术强度 (FLOP/Byte)
  Memory Bound    Compute Bound
  区域 (左侧)      区域 (右侧)

向量加法: AI = 1/12 = 0.08   → 远在左侧 → Memory Bound
GELU:     AI = 15/8 = 1.9    → 还是左侧 → Memory Bound
GEMM:     AI ≈ 170           → 远在右侧 → Compute Bound
```

**关键洞察**：深度学习中几乎所有 elementwise/reduce 算子都是 Memory Bound。
只有矩阵乘法（GEMM）和大卷积是 Compute Bound。

```
Memory Bound → 优化内存访问 (合并、Shared Memory、融合)
Compute Bound → 优化计算 (Tensor Core、ILP、Register Tiling)
```

> 📖 **推荐阅读**: Roofline 模型的完整推导和实战应用——看 [`theory/03_memory_hierarchy.md`](./theory/03_memory_hierarchy.md) 3.8 节。这是你以后分析任何 kernel 性能的第一个工具。

## 4.5 动手：用 ncu 分析你的 kernel

```bash
cd 08_ncu_profiling
nvcc -O2 -lineinfo -o ncu_demo ncu_demo.cu
./ncu_demo
```

这个示例包含 3 个故意有"问题"的 kernel，输出会告诉你 ncu 的关键指标怎么解读。

> 🔍 **想深入？** ncu 每个指标的硬件含义——看 [`08_ncu_profiling/ncu_profiling.md`](./08_ncu_profiling/ncu_profiling.md)。

### ✅ Checkpoint: Part 4

```
在继续之前，确认你理解了:

□ 合并访问: 同一 Warp 的线程访问连续地址 → 1 次事务 → 快
□ 非合并: 地址散乱 → 最多 32 次事务 → 浪费带宽
□ Bank Conflict: Shared Memory 的 32 个 Bank, 同 Bank 冲突 → 串行
□ Roofline: AI < Ridge Point → Memory Bound, AI > Ridge Point → Compute Bound

能回答这个问题吗:
  向量加法的算术强度 = 1 FLOP / 12 Byte = 0.08
  矩阵乘法 1024×1024 的算术强度 = 2×1024³ / (3×1024²×4) ≈ 170
  哪个是 Memory Bound? 哪个是 Compute Bound?
  (答: 向量加法 Memory Bound, 矩阵乘法 Compute Bound)
```

**你现在掌握了**: 合并访问、Bank Conflict、Roofline、ncu 基础。

**调试练手**：做 [`debug_exercises/`](./debug_exercises/) 里的 3 道 debug 练习。每个文件包含 1 个故意埋入的 bug（cudaMemcpy 方向写反 / `__syncthreads()` 缺失 / Softmax 溢出），编译运行后观察错误现象，定位并修复。详见 [`debug_exercises/README.md`](./debug_exercises/README.md)。

**想巩固？** 做 [`05_bank_conflict/exercises/`](./05_bank_conflict/exercises/) 里的练习题（padding 消除冲突 / SMEM 矩阵转置对比），每道都有 Level 1 和 Level 2。

> **过渡思考**：到此为止，你已经有了完整的"武器库"——知道如何并行（Part 1）、
> 复用数据（Part 2）、用 Shuffle 快速通信（Part 3）、分析瓶颈（Part 4）。
> 是时候用这些武器写一个**真正的深度学习算子**了。


---

# Part 5: 写一个真正的算子 — Softmax

> **学习目标**: 综合前面所有技术写出 Softmax 的 3 个优化版本，理解 Online 算法，能接入 PyTorch，理解算子融合。
> **预估时间**: 4-6 小时
> **前置要求**: Part 1-4 (kernel/Shared Memory/Warp Shuffle/合并访问/ncu)

## 5.1 综合前面所有知识

Softmax 是深度学习中最常用的算子之一：

```
softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```

它综合了你学过的所有技术：
- **Reduce**（求 max、求 sum，归约实现可参考 [`03_reduce/`](./03_reduce/) 和 [`theory/04_warp_and_sync.md`](./theory/04_warp_and_sync.md)）
- **数值稳定性**（减 max 防止 exp 溢出）
- **Memory Bound 优化**（减少遍历次数 → Online 算法）

## 5.2 动手：跑 3 个版本的 Softmax

```bash
make 07_softmax         # 或者手动: cd 07_softmax && nvcc -O2 -o softmax softmax.cu
./07_softmax/softmax
```

三个版本：
- V1 (3-pass)：先求 max → 再求 sum(exp) → 最后归一化。遍历数据 3 次。
- V2 (2-pass Online)：一次遍历同时算 max 和 sum → 只遍历 2 次！
- V3 (Warp-level)：当 N ≤ 32 时，一个 Warp 搞定，全用 Shuffle，零 Shared Memory。

**打开 [`07_softmax/softmax.cu`](./07_softmax/softmax.cu)**，重点看 V2 的 Online 算法——这是 FlashAttention 的核心思想。

**先用直觉理解 Online 算法的核心难点**：

V1 的 3-pass 很简单：先扫一遍求 max，再扫一遍求 sum(exp)。但问题是——扫了 2 遍。
V2 想一遍就同时求 max 和 sum(exp)。但这有个矛盾：

```
你在边扫描边算 sum(exp(x - max))
但 max 在不断变化! 每当遇到更大的值, max 就更新了。
那之前算的那些 exp(x_j - old_max) 全都"算错"了,
因为它们应该用 new_max 而不是 old_max!
```

难道每次 max 变了就要回去重算？那不是又变成多遍了？

**关键洞察：不需要回去重算，只需要乘一个修正因子。**

用一个具体例子来看：

```
假设你已经处理了 x=[3, 1], 当前 max=3, sum = exp(3-3) + exp(1-3) = 1 + 0.135 = 1.135

现在来了新元素 x=5, max 需要更新为 5。
之前的 sum = exp(3-3) + exp(1-3) = 1.135  ← 这是基于 old_max=3 算的
我们想要的 sum = exp(3-5) + exp(1-5)       ← 这是基于 new_max=5 的

观察:
  exp(3-5) = exp(3-3) × exp(3-5) = exp(3-3) × exp(old_max - new_max)
  exp(1-5) = exp(1-3) × exp(3-5) = exp(1-3) × exp(old_max - new_max)

所以:
  新 sum = 旧 sum × exp(old_max - new_max) + exp(新元素 - new_max)
         = 1.135 × exp(3-5) + exp(5-5)
         = 1.135 × 0.0183 + 1.0
         = 1.021

验证: exp(3-5) + exp(1-5) + exp(5-5) = 0.0183 + 0.0025 + 1.0 = 1.021 ✓
```

**一句话总结**：`exp(x - old_max) × exp(old_max - new_max) = exp(x - new_max)`。
因子 `exp(old_max - new_max)` 一次性把之前所有的 sum 从旧基准修正到新基准。

理解了这个原理，代码就很清晰了：

```cuda
// V2 的关键循环: 一次遍历同时追踪 max 和 sum
float local_max = -INFINITY;
float local_sum = 0.0f;

for (int i = tid; i < cols; i += blockDim.x) {
    float val = x[i];
    float old_max = local_max;
    local_max = fmaxf(local_max, val);  // 更新 max
    
    // 修正之前的 sum: 旧基准 → 新基准
    local_sum = local_sum * expf(old_max - local_max)  // 旧 sum × 修正因子
              + expf(val - local_max);                  // 加上新元素
}
// 现在 local_sum 就是基于 local_max 的正确部分和!
```

**为什么 V2 比 V1 快？**
```
V1: 读数据 3 次 (max 遍历 + sum 遍历 + 归一化遍历)
V2: 读数据 2 次 (max+sum 合并遍历 + 归一化遍历)
Softmax 是 Memory Bound → 少读 1 遍 ≈ 快 33%
```

**V2 的修正因子 `exp(old_max - new_max)` 就是 FlashAttention 的秘密武器。**
FlashAttention 把同样的技巧扩展到 Attention 的输出向量 O 上：
不仅修正 sum，还修正 O = softmax(QK^T) × V 的部分累加结果。

## 5.3 动手实验

```
1. 把 softmax.cu 中的 BLOCK_SIZE 从 256 改成 1024
   → Performance 变了吗? 为什么? (提示: Shared Memory 够不够?)

2. 对照着跑 V1 (3-pass) 和 V2 (2-pass Online), 用 ncu 对比 Memory Throughput
   → V2 的 DRAM 读取量应该是 V1 的 2/3? 为什么不是? (提示: 还有写操作)

3. 在 V3 (Warp-level) 中把 N 改成 64
   → 还能正常运行吗? 需要怎么改? (提示: 每线程处理 2 个元素)
   → 如果 N=128 呢?

4. 把 V2 的 "修正因子" 改成错误的: sum = sum * expf(new_max - old_max)
   → 看结果的数值变化 (不是变成 NaN 就是严重错误!)
```

### ✅ Checkpoint: Part 5

```
在继续之前，确认你理解了:

□ 为什么 Softmax 要减 max (数值稳定性: exp(100) = 溢出!)
□ Online 算法的修正因子: exp(old_max - new_max)
□ 融合的本质: 中间结果留在寄存器 → 不经过显存 → 省 2N 次访存

动手验证:
  在 softmax.cu 的 CPU 参考中, 去掉"减 max"这一步:
  直接写 y[i] = exp(x[i]) / sum(exp(x[j]))
  把输入数据改成大值 (100.0 + rand) → 观察: 全是 NaN!
  恢复"减 max" → 输出正常
```

## 5.4 动手：接入 PyTorch

```bash
cd 04_pytorch_extension
pip install -e .
python test_gelu.py
```

这个示例展示了怎么把 CUDA kernel 变成 Python 可调用的 PyTorch 算子。

核心结构：
```
CUDA kernel (gelu_cuda.cu) → C++ 封装 (pybind11) → Python 调用 (test_gelu.py)
```

> 🔍 **想深入？** PyTorch 接入的编译链和 autograd.Function？
> 看 [`04_pytorch_extension/pytorch_extension.md`](./04_pytorch_extension/pytorch_extension.md) 和 [`theory/05_operator_development.md`](./theory/05_operator_development.md) 5.8 节。

**想巩固？** 做 [`04_pytorch_extension/exercises/`](./04_pytorch_extension/exercises/) 里的练习：用同样的模式写一个 Sigmoid 算子（ex1）和一个带反向传播的 LeakyReLU 算子（ex2）。

## 5.5 动手：算子融合

```bash
make 10_fused_kernel    # 或者手动: cd 10_fused_kernel && nvcc -O2 -o fused_kernel fused_kernel.cu
./10_fused_kernel/fused_kernel
```

对比 3 个独立 kernel vs 1 个融合 kernel 的性能。
融合的核心：中间结果留在寄存器里（0 延迟），不写回显存（500 周期延迟）。

**你现在掌握了**: 完整的算子开发流程——从数学公式到优化到框架接入。

> **过渡思考**：Part 1-5 你跟着教程一步步走。但真正的能力要在"自己动手"中检验。
> Part 6 不再手把手——给你一个真实的算子需求，用前面学到的所有东西独立完成。
> 
> **为什么选 LayerNorm？** 因为它是 Transformer 里最常见的算子之一，而且算法结构恰好覆盖了前面学过的所有核心技能：
> 
> ```
> LayerNorm 需要...                       对应前面学的...
> ─────────────────                       ──────────────
> 2D 线程索引 (每 Block 处理一行)          Part 2 (matmul, 2D block)
> 求均值和方差 → 这是 reduce 操作!          Part 3 (reduce V1-V6)
> Block 内用 Shared Memory 做线程通信       Part 2 (matmul tiled)
> Warp Shuffle 归约 (减少 BAR.SYNC)        Part 3 (reduce V3)
> Welford 在线算法 (一次遍历算均值和方差)   Part 5 (Online Softmax 的思维)
> PyTorch autograd.Function 接入           Part 5 (PyTorch extension)
> 反向传播 kernel                          Part 5 (Softmax backward)
> ```
> 
> **所以 LayerNorm 不是"学新东西"，而是"把你会的全部用一遍"。**
> 
> 如果你在写的时候卡住了，回头看对应章节即可——知识点你都学过了，现在只是组合起来。


---

# Part 6: 挑战自己

> **学习目标**: 独立完成 LayerNorm 算子（含前向+反向+PytTorch 接入），理解异步执行和多 Stream，探索进阶方向。
> **预估时间**: 6-10 小时
> **前置要求**: Part 5 (Softmax + PyTorch 接入 + 算子融合)

## 6.1 综合项目：手写 LayerNorm

打开 [`12_layernorm_project/README.md`](./12_layernorm_project/README.md)，按任务说明完成：
1. 用 Welford 算法一次遍历算均值+方差（[`theory/07_classic_operators.md`](./theory/07_classic_operators.md) 7.2 节）
2. 用 Warp Shuffle 做 Block 内归约（参考 [`03_reduce/`](./03_reduce/)）
3. 接入 PyTorch（参考 [`04_pytorch_extension/`](./04_pytorch_extension/)）

Starter code 在 [`12_layernorm_project/layernorm_starter.cu`](./12_layernorm_project/layernorm_starter.cu)，填完 TODO 即可。
参考答案在 [`12_layernorm_project/layernorm_solution.cu`](./12_layernorm_project/layernorm_solution.cu)（先自己做！）。

PyTorch 接入的完整代码也已准备好：
```bash
cd 12_layernorm_project
pip install -e .                    # 编译 CUDA 扩展
python test_layernorm.py            # 正确性 + 性能对比
```

## 6.2 Stream 与异步执行

在前面的学习中，所有操作都在默认 Stream 上串行执行。
但 GPU 有独立的传输引擎和计算引擎——它们可以同时工作！

```bash
make 16_streams         # 或者手动: cd 16_streams && nvcc -O2 -o streams streams.cu
./16_streams/streams
```

### Stream 的核心概念

```
GPU 有 3 个独立的硬件引擎:
  Copy Engine (H2D):  CPU → GPU 数据传输
  Copy Engine (D2H):  GPU → CPU 数据传输  
  Compute Engine:     kernel 计算

Stream = GPU 上的一个操作队列 (FIFO)
  同一 Stream 内: 操作串行 (保持依赖关系)
  不同 Stream 间: 操作可以并行 (不同引擎同时工作)
```

你会看到：
- **单 Stream**：传输和计算严格串行，总时间 = T_传输 + T_计算
- **多 Stream**：传输和计算重叠执行，总时间 ≈ max(T_传输, T_计算)
- **Pinned Memory** 比普通 malloc 快 1.5-2×（DMA 直传 vs 额外拷贝）

### 关键陷阱: Default Stream 的隐式同步

```
如果不显式创建 Stream, 所有操作在 Default Stream (stream 0) 中执行。
Legacy default stream 有一个坑: 它会隐式同步所有其他 Stream!

// 这样写没有 overlap!
kernel<<<grid, block>>>(d_a);          // default stream
cudaStream_t s; cudaStreamCreate(&s);
kernel<<<grid, block, 0, s>>>(d_b);    // stream s
// → default stream 会等 s, s 也会等 default stream → 完全串行!

解决方法: 要么所有操作都用显式 Stream, 要么用 --default-stream per-thread
```

### 另一个关键陷阱: cudaMemcpyAsync + Pageable Memory

```
cudaMemcpyAsync 虽然名叫 "Async", 但如果你传了 pageable (普通 malloc) 内存:
  → CUDA 驱动退化为同步行为! CPU 线程阻塞直到拷贝完成!
  → 即使用了显式 Stream, 重叠也完全消失!

原因: DMA 引擎只能访问物理地址固定的内存。
Pageable memory 的物理页可能被 OS 换出 → 驱动必须先把数据复制到内部 pinned buffer。

正确做法: 用 cudaMallocHost 分配 pinned memory, 然后 cudaMemcpyAsync 才是真异步。
```

### PCIe 带宽的物理限制

```
注意: 在 PCIe 系统 (绝大多数用户的场景) 上, H2D 和 D2H 共享同一条 PCIe 总线。
双向同时传输时带宽各减半! (NVLink 系统如 DGX 不受此限制)
```

这是深度学习训练中 DataLoader `pin_memory=True` 和 NCCL 通信重叠的基础。

### 动手实验

```
1. 在 streams.cu 中, 把数据大小从 64MB 改成 1MB
   → 重叠效果还有吗? 为什么? (提示: 传输时间太短, 来不及重叠)

2. 对比 cudaMallocHost (Pinned) 和 malloc (Pageable) 的 cudaMemcpy 耗时
   → Pinned Memory 为什么快? (DMA 直传 GPU 显存)

3. 尝试 4 个 Stream 而不是 2 个 → 性能有提升吗?
   → H2D engine 只有 1 个, 多个 Stream 只是排队
```

## 6.3 混合精度实战

```bash
make 17_mixed_precision
./17_mixed_precision/mixed_precision
```

### 三种精度的本质区别

```
FP32 (4 bytes): 1 sign + 8 exp + 23 mantissa
  范围: ±3.4×10³⁸, 精度: ~7 位有效数字 → 训练基准

FP16 (2 bytes): 1 sign + 5 exp + 10 mantissa  
  范围: ±65504, 精度: ~3-4 位有效数字
  风险: 梯度 < 6×10⁻⁸ 下溢为 0! 训练需要 Loss Scaling

BF16 (2 bytes): 1 sign + 8 exp + 7 mantissa
  范围: 和 FP32 相同! 精度: ~2-3 位有效数字
  优势: 不需要 Loss Scaling, 截断 FP32 高 16 位即可互转

TF32 (19 bits, Ampere+ 默认): 1 sign + 8 exp + 10 mantissa
  Tensor Core 内部格式, FP32 输入自动转换
  范围 = FP32, 精度 ≈ FP16 → 训练时的"免费午餐"
```

### Loss Scaling — FP16 训练的必需品

```
问题: FP16 最小正数 ≈ 6×10⁻⁸, 训练中很多梯度比这小 → 变成 0!
解决: 前向完成后, loss 乘以一个大数 (如 1024):
  小梯度 × 1024 → FP16 能表示
  
  反向传播后再除以 1024 恢复:
  scaled_loss = loss * 1024
  scaled_loss.backward()  // 梯度也被放大 1024×
  for p in model.parameters():
      p.grad /= 1024       // 恢复真实梯度

PyTorch 自动管理: torch.cuda.amp.GradScaler
BF16 不需要: 指数范围和 FP32 相同, 小梯度不会下溢
```

### FP16 vs BF16 — 训练中怎么选

```
选 BF16 如果:
  ✓ GPU 是 Ampere (A100/RTX 3090) 或更新
  ✓ 不想维护 Loss Scaling 代码
  ✓ 训练稳定性优先

选 FP16 如果:
  ✓ GPU 是 Volta/Turing (V100/T4/RTX 2080) — 不支持 BF16!
  ✓ 需要更高的尾数精度 (10 bit vs 7 bit)
  ✓ 老代码库已经基于 FP16

推理: 通常用 FP16 (精度够 + 模型更小)
```

### 动手实验

```
1. 在 mixed_precision.cu 中, 把 Loss Scaling 的 scale 从 1024 改成 1
   → 小梯度能存活吗? (查看 FP16 回读后的值)

2. 找一个接近 65504 (FP16 max) 的值, 用 __float2half 转换
   → 观察溢出后的值是什么 (变成 inf!)

3. BF16 和 FP32 范围相同: 找一个 FP32 能表示但 FP16 溢出的值
   (如 70000.0f), 分别用 BF16 和 FP16 存储再读回
   → BF16 正确, FP16 变成 inf

4. (思考) 为什么 GEMM 用 FP16/BF16 收益巨大, 但 LayerNorm 用 FP16 几乎没提升?
   提示: 算一下两者的算术强度.
```

> 📖 **推荐阅读**: Stream 与异步执行的完整原理——看 [`16_streams/streams.md`](./16_streams/streams.md) 和 [`theory/02_cuda_programming_model.md`](./theory/02_cuda_programming_model.md) 2.4-2.5 节。

## 6.4 进阶代码

已经完成上面的内容? 这些代码等着你:

- [`09_register_tiling/`](./09_register_tiling/) — 手写 Register Blocked GEMM（[`theory/05_operator_development.md`](./theory/05_operator_development.md) 5.9 节），每线程算 4×4 个输出元素，数据复用率提高 4×
- [`13_flash_attention/`](./13_flash_attention/) — 简化版 FlashAttention（[`theory/07_classic_operators.md`](./theory/07_classic_operators.md) 7.3 节），Online Softmax + 分块注意力，理解 LLM 推理的核心
- [`14_im2col_conv/`](./14_im2col_conv/) — im2col 卷积（[`theory/05_operator_development.md`](./theory/05_operator_development.md) 5.5 节），把卷积转化为 GEMM，理解 cuDNN 的基本思路
- [`15_wmma_gemm/`](./15_wmma_gemm/) — Tensor Core WMMA（[`theory/06_tensor_core.md`](./theory/06_tensor_core.md) 6.3 节），一条指令做 16×16×16 矩阵乘，吞吐是 CUDA Core 的 ~1000×

每个目录都有配套的 `.md` 文档，解释硬件上的精确行为。
建议顺序：09 → 15 → 13（从 GEMM 优化到 Tensor Core 到 Attention）。

## 6.5 深入理论

现在你有了足够的实战经验，回头读 theory/ 会有不同的感受:

- [`theory/01_gpu_architecture.md`](./theory/01_gpu_architecture.md) — 你跑的代码在硬件上到底怎么执行的
- [`theory/03_memory_hierarchy.md`](./theory/03_memory_hierarchy.md) — 为什么合并访问快（从 HBM Bank 结构讲起）
- [`theory/04_warp_and_sync.md`](./theory/04_warp_and_sync.md) — Warp Divergence 的 SASS 级分析
- [`theory/06_tensor_core.md`](./theory/06_tensor_core.md) — Tensor Core / CUTLASS / Hopper TMA
- [`theory/08_advanced_optimization.md`](./theory/08_advanced_optimization.md) — 持久化内核 / 量化 / Multi-GPU
- 软硬件映射总览: [`theory/software_hardware_mapping.md`](./theory/software_hardware_mapping.md)
- 调试与优化手册: [`DEBUG_AND_OPTIMIZE.md`](./DEBUG_AND_OPTIMIZE.md)（遇到问题时翻，现在加了错误检查最佳实践）
- 练习题参考答案: [`EXERCISE_ANSWERS.md`](./EXERCISE_ANSWERS.md)（Ch2-Ch8 全部有答案）

## 6.6 你的下一步

根据你的方向选择:

**"我做 AI 推理优化"**
- 精读 [`theory/06_tensor_core.md`](./theory/06_tensor_core.md)（Tensor Core）+ [`theory/07_classic_operators.md`](./theory/07_classic_operators.md)（FlashAttention）
- 尝试用 CUTLASS 写一个自定义 GEMM
- 看 TensorRT 的 plugin 机制

**"我做 AI 训练框架"**
- 精读 [`theory/08_advanced_optimization.md`](./theory/08_advanced_optimization.md)（Multi-GPU，持久化内核）
- 理解 NCCL 的 Ring AllReduce
- 看 Megatron-LM 的并行策略

**"我做科学计算 / HPC"**
- 精读 [`theory/03_memory_hierarchy.md`](./theory/03_memory_hierarchy.md)（内存优化全篇）+ [`theory/08_advanced_optimization.md`](./theory/08_advanced_optimization.md)（ILP，双缓冲）
- 学习 CUDA Graph 和 MPS
- 看 cuFFT / cuSPARSE 的设计思路

**"我想面试 CUDA 岗位"**
- 做完所有 theory/ 章节的练习题
- 能手写 Softmax + LayerNorm + Reduce 的优化版本
- 能看 ncu 报告定位性能瓶颈


---

# 附录：这条路线你学到了什么

```
Part 0: ✓ 环境验证, nvidia-smi, nvcc, Compute Capability
Part 1: ✓ kernel 写法, 线程编号, GPU 内存管理
Part 2: ✓ Shared Memory, __syncthreads, Tiling
Part 3: ✓ Warp, Warp Shuffle, 分支分歧
Part 4: ✓ 合并访问, Bank Conflict, Roofline, ncu, 调试练习
Part 5: ✓ Softmax 3 版本, PyTorch 接入, 算子融合
Part 6: ✓ LayerNorm 综合项目, Stream 异步, 混合精度

你已经具备独立开发 CUDA 算子的能力 —— 从写 kernel 到分析性能到接入框架。
接下来就是在实际项目中练习 + 按需深入 theory/ 的特定章节。

> 面试准备：做 [`INTERVIEW_QUESTIONS.md`](./INTERVIEW_QUESTIONS.md) 看看自己能不能答出来。
> 速查表：[`CHEAT_SHEET.md`](./CHEAT_SHEET.md) — Roofline、ncu、内存层级一页回顾。
```
