# 第二章：CUDA 编程模型 — 从 CPU 调用到 GPU 执行的每一步

**难度**: ⭐⭐ 进阶 (2.1-2.6) / ⭐⭐⭐ 专家 (2.0 Launch 全路径)
**前置知识**: 能写简单的 CUDA kernel (完成 [`tutorial.md`](../tutorial.md) Part 1-2)
**读完你能做什么**: 理解 <<<>>> 背后发生的一切；正确使用 Stream/Event/Graph
**配套代码**: `01_vector_add/` (线程索引), `02_matrix_mul/` (2D Grid)
**新手建议**: 先读 2.1 (线程层级) 和 2.6 (Grid-Stride Loop)，2.0 的 Launch 全路径在你需要减少 launch 开销时再读

## 2.0 Kernel Launch 全路径 — 你写下 `<<<>>>` 之后到底发生了什么

> 这是本章最深入的一节。如果你刚完成 [`tutorial.md`](../tutorial.md) Part 1，可以先跳到 2.1 (线程层级)，
> 等你需要理解 launch 开销、或者需要用 CUDA Graph 优化时再回来读。
>
> 本节首次出现的术语:
> - **CUDA Runtime**: NVIDIA 提供的 C++ 库 (libcudart.so)，封装了 GPU 的操作接口
> - **cubin / fatbin**: 编译好的 GPU 二进制程序 (类似 CPU 的 .exe)
> - **PTX**: GPU 的虚拟指令集 (类似 Java 字节码)，可以被 JIT 编译到任何 GPU 架构
> - **SASS**: GPU 的真实机器指令 (类似 CPU 的 x86 汇编)，每代架构不同
> - **Constant Bank**: GPU 上的一小块只读高速存储 (64KB)，kernel 参数存放在这里
> - **Command Buffer**: CPU 和 GPU 之间的命令队列 (环形缓冲区)
> - **Doorbell Register**: CPU 写这个寄存器来通知 GPU "有新命令了"
> - **MMIO (Memory-Mapped IO)**: CPU 通过写特定内存地址来控制 GPU 硬件
> - **CTA (Cooperative Thread Array)**: NVIDIA 硬件文档对 "Block" 的称呼，两者完全相同
> - **CTA Scheduler (GigaThread Engine)**: GPU 的全局调度器，把 Block 分配到 SM

这一节追踪从 CPU 端 `kernel<<<grid, block>>>(args)` 到 GPU 上第一条指令执行
的完整硬件路径。每一步都涉及具体的硬件单元。

### Step 0: CPU 端 — CUDA Runtime 打包命令

```
当你写:
  my_kernel<<<dim3(128,1,1), dim3(256,1,1), 4096, stream>>>(ptr, n);

CUDA Runtime (libcudart.so) 做的事:

1. 查找 kernel 函数的 GPU 二进制 (cubin/fatbin):
   - 编译时 nvcc 将 kernel 编译成多种架构的二进制, 打包在 fatbinary 中
   - Runtime 根据当前 GPU 的 Compute Capability 选择匹配的 cubin
   - 如果没有精确匹配 → JIT 编译嵌入的 PTX → SASS

2. 准备 Kernel Descriptor (内核描述符):
   ┌────────────────────────────────────────────┐
   │ Kernel Descriptor                          │
   │                                            │
   │ kernel_func_ptr:  0x7f3a2000 (设备端地址)   │
   │ grid_dim:         {128, 1, 1}              │
   │ block_dim:        {256, 1, 1}              │
   │ shared_mem_bytes:  4096                    │
   │ stream:           0x55a3b8c0               │
   │                                            │
   │ 从 cubin 中提取的信息:                       │
   │ regs_per_thread:   48                      │
   │ static_smem:       2048                    │
   │ max_threads_per_block: 1024                │
   │ const_mem_size:     340                    │
   │ local_mem_size:     0 (无 spilling)         │
   │ bar_count:          1 (使用了几个 barrier)   │
   └────────────────────────────────────────────┘

3. 传递 kernel 参数:
   参数打包到 Constant Memory Bank (CB):
   - NVIDIA GPU 有多个 Constant Bank, kernel 参数使用 CB0
   - 最大 4KB 参数
   - 参数在 launch 时被 DMA 到 GPU 的 Constant Memory
   
   参数在 kernel 代码中通过 SASS 指令访问:
   LDC R0, c[0x0][0x168];   // 从 Constant Bank 0, 偏移 0x168 读取
                              // c[0x0] = CB0 = 参数区
                              // 0x168 = 第 N 个参数的偏移

4. 写入 Command Buffer:
   CUDA Driver 将 launch 命令写入 GPU Command Buffer:
   ┌─────────────────────────────────────┐
   │ Command Buffer (Ring Buffer in      │
   │ pinned system memory / GPU BAR)     │
   │                                     │
   │ ... 之前的命令 ...                   │
   │ [COMPUTE_LAUNCH]                    │
   │   kernel_desc_ptr: 0xGPU_ADDR       │
   │   param_buffer_ptr: 0xGPU_ADDR      │
   │ ... 之后的命令 ...                   │
   └───────────────┬─────────────────────┘
                   │
                   ▼
   CPU 写 GPU 的 Doorbell Register (MMIO)
   通知 GPU: "Command Buffer 有新命令了"
   
   CPU 立即返回! (异步) 不等 GPU 执行。
   这一步 CPU 端延迟: ~3-8 μs (Launch Overhead)
```

### Step 1: GPU Front-End — 命令处理器

```
GPU 的 Host Interface 检测到 Doorbell:

Host Interface / Command Processor:
┌────────────────────────────────────────────┐
│ 1. 从 Command Buffer 读取 COMPUTE_LAUNCH    │
│                                            │
│ 2. 解析 Kernel Descriptor:                  │
│    - 计算总 Block 数 = 128×1×1 = 128       │
│    - 每 Block 线程数 = 256×1×1 = 256       │
│    - 每 Block 寄存器 = 256 × 48 = 12288    │
│    - 每 Block Shared Mem = 2048 + 4096      │
│      = 6144 bytes (static + dynamic)        │
│                                            │
│ 3. 检查合法性:                              │
│    - blockDim ≤ 1024? ✓                    │
│    - shared_mem ≤ max_smem_per_block? ✓     │
│    - regs_per_thread ≤ 255? ✓              │
│                                            │
│ 4. 将 kernel launch 传递给 CTA Scheduler    │
│    (也叫 GigaThread Engine / Work            │
│     Distribution Unit)                      │
└────────────────────────────────────────────┘
```

### Step 2: CTA Scheduler — 核心的 Block 分配引擎

```
CTA = Cooperative Thread Array = CUDA 编程模型中的 "Block"。
CTA Scheduler 负责将 CTA 分配到 SM。

CTA Scheduler 维护的状态:
┌──────────────────────────────────────────────────┐
│ CTA Scheduler (GigaThread Engine)                │
│                                                  │
│ Pending CTA Queue:                               │
│   [Kernel A: CTA 0-127, 128 CTAs remaining]      │
│   [Kernel B: CTA 0-63, 64 CTAs remaining]        │
│   ...                                            │
│                                                  │
│ SM Resource Table (每个 SM 一个 entry):            │
│ ┌──────────────────────────────────────────────┐ │
│ │ SM 0:                                        │ │
│ │   Active CTAs: 5 / max 32                    │ │
│ │   Active Warps: 40 / max 64                  │ │
│ │   Used Registers: 49152 / 65536              │ │
│ │   Used Shared Mem: 30720 / 167936            │ │
│ │   Available Slots:                           │ │
│ │     CTA: 27, Warp: 24, Regs: 16384, Smem: 137216 │
│ │                                              │ │
│ │ SM 1:                                        │ │
│ │   Active CTAs: 4 / max 32                    │ │
│ │   ...                                        │ │
│ └──────────────────────────────────────────────┘ │
│                                                  │
│ 每个周期 (或几个周期) 的操作:                       │
│                                                  │
│ for each SM:                                     │
│   取队列头部的 CTA, 检查 SM 是否有足够资源:         │
│                                                  │
│   need_regs = blockDim × regs_per_thread         │
│   → 向上取整到分配粒度 (Warp × 256 regs 为单位)   │
│   实际: 8 Warps × ceil(48×32/256)×256            │
│       = 8 × 1536 = 12288 regs / CTA             │
│                                                  │
│   need_smem = static_smem + dynamic_smem         │
│   → 向上取整到 128 bytes                          │
│   实际: ceil(6144/128)×128 = 6144 bytes          │
│                                                  │
│   need_warps = blockDim / 32 = 8                 │
│   need_cta_slot = 1                              │
│                                                  │
│   如果 SM 所有资源都够:                            │
│     分配! 更新 SM Resource Table                  │
│     CTA counter++                                │
│   否则:                                          │
│     跳过这个 SM, 试下一个 SM                       │
│                                                  │
│ CTA 分配顺序:                                     │
│   通常按 SM ID 轮询 (Round-Robin)                 │
│   但会跳过资源不足的 SM                            │
│   一旦所有 SM 都满了 → 等待某个 SM 上的 CTA 完成    │
└──────────────────────────────────────────────────┘
```

### Step 3: SM 接收 CTA — 资源初始化

```
CTA Scheduler 告诉 SM X: "给你一个新的 CTA, 参数如下"

SM 的 CTA Setup 硬件:
┌────────────────────────────────────────────────┐
│ SM X 收到 CTA 分配请求:                          │
│                                                │
│ 1. Register File 分配:                          │
│    从未使用的寄存器区域划出 12288 个寄存器           │
│    为 8 个 Warp 各分配 1536 个寄存器               │
│    (= 48 regs/thread × 32 threads)              │
│                                                │
│    Warp 0: R[base+0]    ~ R[base+1535]          │
│    Warp 1: R[base+1536] ~ R[base+3071]          │
│    ...                                          │
│    Warp 7: R[base+10752] ~ R[base+12287]        │
│                                                │
│ 2. Shared Memory 分配:                          │
│    从 Shared Memory 空间划出 6144 bytes            │
│    设置 Shared Memory base address 寄存器         │
│    kernel 中 __shared__ 变量的地址相对于 base       │
│                                                │
│ 3. Warp 初始化:                                  │
│    将 8 个 Warp 分配到 4 个 Processing Block:     │
│    PB0: Warp 0, Warp 4                          │
│    PB1: Warp 1, Warp 5                          │
│    PB2: Warp 2, Warp 6                          │
│    PB3: Warp 3, Warp 7                          │
│    (Round-Robin 分配)                            │
│                                                │
│    每个 Warp 的初始状态:                           │
│    - PC = kernel 入口地址                         │
│    - Active Mask = 0xFFFFFFFF (全部 32 线程活跃)   │
│      (如果总线程数不是 32 的倍数, 最后一个 Warp     │
│       的 mask 会部分为 0)                         │
│    - 初始化特殊寄存器:                             │
│      %tid.x = 该线程的 threadIdx.x               │
│      %tid.y = threadIdx.y                        │
│      %tid.z = threadIdx.z                        │
│      %ctaid.x = blockIdx.x                      │
│      %ctaid.y = blockIdx.y                       │
│      %ctaid.z = blockIdx.z                       │
│      %ntid.x = blockDim.x (= 256)               │
│      %nctaid.x = gridDim.x (= 128)              │
│                                                │
│ 4. Barrier 初始化:                               │
│    CTA 的 Barrier 0 初始化:                       │
│      participant_count = 256 (所有线程)           │
│      arrived_count = 0                           │
│    (Barrier 0 对应 __syncthreads())              │
│                                                │
│ 5. 指令预取:                                     │
│    I-Cache 开始预取 kernel 入口地址附近的指令        │
│                                                │
│ 6. 所有 8 个 Warp 变为 Eligible:                  │
│    下一个周期, Warp Scheduler 可以选中它们执行     │
│                                                │
│ 整个 CTA Setup 过程: ~几十个周期                   │
└────────────────────────────────────────────────┘
```

### Step 4: 第一条指令执行

```
SM Setup 完成后, Warp 0 (在 PB0 上) 通常是第一个被 Scheduler 选中的:

Warp 0, PC = kernel 入口:

SASS 指令可能是:
  S2R R0, SR_TID.X;      // 读取 threadIdx.x 到 R0
                           // S2R = Special Register Read
                           // 这条指令读的是硬件初始化时设置的特殊寄存器

  IMAD.MOV R1, RZ, RZ, c[0x0][0x168];  // 从 Constant Bank 读取第一个 kernel 参数
                           // c[0x0] = Constant Bank 0
                           // [0x168] = 参数在 CB 中的偏移
                           // IMAD.MOV 是 "整数乘加作为 MOV" 的编码技巧
                           // (RZ = 零寄存器, RZ×RZ + c[0x0][0x168] = 参数值)

  IMAD R2, R0, 0x4, R1;  // R2 = threadIdx.x * 4 + base_ptr
                           // 计算全局内存地址

  LDG.E R3, [R2];         // 从计算出的地址加载第一个元素
                           // 这条指令发射后:
                           // - Long Scoreboard 标记 R3 为 pending
                           // - LD/ST Unit 开始地址生成和合并
                           // - Warp 0 进入 Stall (Long Scoreboard)
                           // - Scheduler 切换到 Warp 1 (或其他 Eligible Warp)

从 kernel<<<>>> 调用到第一条 LDG 发射:
  CPU 端: ~3-8 μs (launch overhead)
  GPU Command Processor: ~几百 ns
  CTA Scheduler: ~几百 ns
  SM Setup: ~几十个 cycles (~几十 ns at 1.4 GHz)
  第一条指令: ~1 cycle
  
  总计: ~4-10 μs (dominated by CPU-side launch overhead)
  
  这就是为什么:
  - 小 kernel 的 launch 开销是主要瓶颈
  - CUDA Graph 可以减少这个开销 (~1 μs for graph launch)
  - Persistent Kernel 可以消除重复 launch 的开销
```

### CTA 执行完毕后的清理

```
当一个 CTA 的所有 Warp 都执行到 EXIT 指令:

1. SM 检测到 CTA 所有 Warp 都已退出
   (所有 Warp 的 PC 到达 EXIT, Active Mask 清零)

2. 释放资源:
   - Register File: 12288 个寄存器回到可用池
   - Shared Memory: 6144 字节回到可用池
   - Warp Slot: 8 个 Warp Slot 释放
   - CTA Slot: 1 个 Block Slot 释放
   - Barrier 硬件释放

3. 通知 CTA Scheduler:
   "SM X 有资源空出来了"
   → CTA Scheduler 可以分配新的 CTA 到该 SM

4. 如果这是该 kernel 的最后一个 CTA:
   → GPU 向 Host Interface 发信号
   → 如果 CPU 在 cudaDeviceSynchronize() 等待 → 返回
   → Stream 中的下一个命令可以开始执行

CTA 退出的 Tail Effect:
  设 108 SM, 128 CTA:
  Wave 1: CTA 0-107 → 108 SM 全忙
  Wave 2: CTA 108-127 → 只有 20 SM 忙, 88 SM 空闲!
  → 最后 20 个 CTA 执行时, 82% 的 GPU 在空转
  → 总效率 ≈ (108 + 20) / (108 × 2) ≈ 59%
  
  优化: Grid-Stride Loop 让 gridDim = 108 × N (N=每SM同时Block数)
  → 没有 tail wave
```

---

## 2.1 线程层级体系 (Thread Hierarchy) — 完整细节

CUDA 采用三层线程组织结构。每一层的存在都有其硬件和编程模型上的必要性。

```
Grid (网格) — 对应整个 GPU
│
├── Block (0,0,0)    Block (1,0,0)    Block (2,0,0)   ...
├── Block (0,1,0)    Block (1,1,0)    Block (2,1,0)   ...
├── Block (0,0,1)    Block (1,0,1)    Block (2,0,1)   ...
└── ...
    每个 Block 内部:
    ├── Warp 0: Thread (0,0,0) ~ Thread (31,0,0)
    ├── Warp 1: Thread (0,1,0) ~ Thread (31,1,0)
    └── ...
```

### Grid — 最外层
- 一个 kernel launch 产生一个 Grid
- Grid 可以是 1D、2D 或 3D 的 Block 数组
- Grid 维度通过 `gridDim.x / .y / .z` 获取
- 当前 Block 在 Grid 中的位置通过 `blockIdx.x / .y / .z` 获取
- **Grid 维度上限** (Compute Capability 7.x+):
  - `gridDim.x`: 2^31 - 1 (2,147,483,647)
  - `gridDim.y`: 65535
  - `gridDim.z`: 65535

### Block — 中间层
- Block 可以是 1D、2D 或 3D 的线程数组
- Block 维度通过 `blockDim.x / .y / .z` 获取
- **Block 的硬件限制**:
  - 总线程数上限: **1024** (所有维度乘积)
  - `blockDim.x` 上限: 1024
  - `blockDim.y` 上限: 1024
  - `blockDim.z` 上限: 64
  - 比如 (1024, 1, 1) ✓ 但 (32, 32, 2) = 2048 ✗

### Thread — 最内层
- 通过 `threadIdx.x / .y / .z` 获取自己在 Block 内的位置
- **线程到 Warp 的映射规则**:
  先按 x 维度排列，再按 y，最后按 z:

```
线性 ID = threadIdx.x + threadIdx.y * blockDim.x 
        + threadIdx.z * blockDim.x * blockDim.y

Warp 编号 = 线性 ID / 32
Lane 编号 = 线性 ID % 32
```

```
示例: blockDim = (8, 4, 1), 共 32 个线程 = 1 个 Warp

threadIdx.y:
3 │ (0,3) (1,3) (2,3) (3,3) (4,3) (5,3) (6,3) (7,3) │ ID 24-31
2 │ (0,2) (1,2) (2,2) (3,2) (4,2) (5,2) (6,2) (7,2) │ ID 16-23
1 │ (0,1) (1,1) (2,1) (3,1) (4,1) (5,1) (6,1) (7,1) │ ID  8-15
0 │ (0,0) (1,0) (2,0) (3,0) (4,0) (5,0) (6,0) (7,0) │ ID  0-7
  └──────────────────────────────────────────────────┘
    0     1     2     3     4     5     6     7  threadIdx.x

这些全在同一个 Warp 内。
```

### 全局索引计算 — 详细推导

```cuda
// 1D 情况:
// Block 0: threads 0 ~ blockDim.x-1
// Block 1: threads blockDim.x ~ 2*blockDim.x-1
// Block b: threads b*blockDim.x ~ (b+1)*blockDim.x-1
int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D Grid × 2D Block (矩阵处理):
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int globalIdx = row * width + col;

// 为什么 y 对应行, x 对应列?
// 因为同一个 Warp 的线程 threadIdx.x 连续
// → col 连续 → 内存地址连续 → 合并访问!
// 如果反过来，同一 Warp 的线程访问不同行 → 跨步访问 → 性能差

// 3D Grid × 3D Block (体数据):
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int globalIdx = x + y * dimX + z * dimX * dimY;
```

### 多维 Grid/Block 的设计理由

1. **自然映射问题结构**：
   - 图像处理 → 2D Grid
   - 体数据/3D 卷积 → 3D Grid
   - 向量操作 → 1D Grid

2. **简化索引计算**：避免手动做除法/取模（GPU 上整数除法很慢: ~80 cycles）

3. **优化内存访问模式**：多维布局可以自然对齐内存访问方向


## 2.2 Kernel 的声明与启动 — 完整细节

### 函数修饰符

```cuda
__global__ void kernel(...)    // 从 CPU (Host) 调用，在 GPU (Device) 执行
                                // 必须返回 void
                                // 支持递归 (CC ≥ 5.0, 不推荐)
                                // 不能取地址 (除了 dynamic parallelism)

__device__ float helper(...)   // 只能从 GPU 代码调用
                                // 可以有返回值
                                // 可以内联 (编译器通常会自动内联)
                                // 可以递归 (不推荐)

__host__ float cpu_func(...)   // 在 CPU 上执行 (默认, 可省略)

__host__ __device__ float portable(float x) {
    // CPU 和 GPU 都能调用
    // 编译器会为两个架构各生成一份代码
    // 条件编译:
    #ifdef __CUDA_ARCH__
        // GPU 路径: __CUDA_ARCH__ 的值是计算能力 (如 800 表示 CC 8.0)
        return __expf(x);    // GPU 快速 exp (精度较低但快)
    #else
        // CPU 路径
        return expf(x);      // 标准 exp
    #endif
}

__forceinline__ __device__ float fast_func(float x) {
    // __forceinline__: 强制内联，减少函数调用开销
    // 对性能关键的小函数很有用
    return x * x;
}

__noinline__ __device__ void big_func(...) {
    // __noinline__: 禁止内联
    // 对很大的函数有用，避免代码膨胀导致 I-Cache 压力
}
```

### 启动配置 — 完整参数

```cuda
kernel<<<gridDim, blockDim, dynamicSharedMem, stream>>>(args...);
```

各参数详解:

```
gridDim (dim3):
  Grid 中有多少个 Block。
  dim3 是三维的: dim3(x, y, z)，默认 y=1, z=1
  也可以用 int: kernel<<<128, 256>>>(...) 等价于 <<<dim3(128), dim3(256)>>>

blockDim (dim3):
  每个 Block 有多少个线程。
  blockDim.x * blockDim.y * blockDim.z ≤ 1024

dynamicSharedMem (size_t, 默认 0):
  每个 Block 动态分配的 Shared Memory 字节数。
  使用 extern __shared__ 声明时需要指定。
  注意: 当需要 > 48KB 时要调用:
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);

stream (cudaStream_t, 默认 0):
  指定在哪个 CUDA Stream 上执行。
  默认 Stream 0 是隐式同步的 (legacy default stream)。
  
  注意: 编译时加 --default-stream per-thread 可以让每个 CPU 线程
  有独立的默认 Stream，避免意外的跨线程同步。
```

### Kernel 参数传递机制

```
kernel 的参数不是通过栈传递的 (GPU 没有传统意义的栈):

1. 所有参数被打包成一个结构体
2. 这个结构体被写入 Constant Memory 的特殊区域
3. 每个 SM 的 Constant Cache 会缓存这些参数
4. 所有线程读取参数时 → Constant Cache 命中 → 1 周期

限制:
- 参数总大小: ≤ 4KB (CUDA 规范)
- 参数必须是 POD 类型 (Plain Old Data): 不能传 std::vector 等
- 可以传指针 (指向 Device Memory)
- 可以传结构体 (被逐字节拷贝到 GPU)
```

### blockSize 深度分析

```
blockSize 的选择涉及多个因素的权衡:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  blockSize 太小 (如 32)                                      │
│  ├── 每 Block 只有 1 个 Warp                                 │
│  ├── SM 需要容纳更多 Block 才能达到好的 Occupancy              │
│  ├── Block 调度开销增加 (更多 Block 要分配)                    │
│  ├── 每 Block 的 Shared Memory 利用率低                       │
│  └── 但: Block 间的并行度高，尾部效应可能更小                   │
│                                                             │
│  blockSize 适中 (128-256)  ← 最常用                          │
│  ├── 4-8 个 Warp / Block，足够做 Warp 级并行                  │
│  ├── Shared Memory 利用率好                                  │
│  ├── Occupancy 通常不错                                      │
│  └── 适用于大多数 kernel                                     │
│                                                             │
│  blockSize 较大 (512-1024)                                   │
│  ├── 16-32 个 Warp / Block                                   │
│  ├── 每 Block 可用 Shared Memory 更多 (因为 Block 数少)        │
│  ├── __syncthreads() 开销增加 (要同步更多 Warp)               │
│  ├── 寄存器压力大: 每线程可用寄存器减少                         │
│  ├── 可能只有 1-2 个 Block / SM → 灵活性差                     │
│  └── 适用于: 需要大量线程间协作的 kernel (如大 reduce)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 动态 Occupancy 查询 API

```cuda
// 让 CUDA Runtime 帮你选最优的 blockSize
int blockSize, minGridSize;
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,   // 输出: 建议的最小 Grid 大小
    &blockSize,     // 输出: 建议的 Block 大小
    kernel,         // 你的 kernel 函数指针
    0,              // 动态 Shared Memory 大小
    0               // Block 大小上限 (0 = 不限制)
);

// 查询指定配置的 Occupancy
int maxActiveBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxActiveBlocks,
    kernel,
    blockSize,
    dynamicSharedMemSize
);
float occupancy = (maxActiveBlocks * blockSize) / 
                  (float)deviceProp.maxThreadsPerMultiProcessor;
printf("理论 Occupancy: %.1f%%\n", occupancy * 100);
```


## 2.3 SIMT 执行模型 — 深入到指令级别

### SIMT vs SIMD — 根本区别

```
SIMD (CPU, 如 AVX-512):
  程序员写:  __m512 result = _mm512_fmadd_ps(a, b, c);
  硬件执行:  一条指令，16 个 float 同时计算
  特点:
  - 必须显式使用向量类型和 intrinsic
  - 所有 lane 执行完全相同的操作
  - 分支: 需要用 mask 手动处理 (_mm512_mask_XXX)
  - 如果只需要 5 个元素? 还是要执行 16 个 lane，浪费 11 个

SIMT (GPU):
  程序员写:  float result = a * b + c;  // 标量代码!
  硬件执行:  32 个线程各自独立执行这行代码
  特点:
  - 程序员写的是单线程逻辑
  - 编译器和硬件自动将 32 个线程的执行向量化
  - 分支: 每个线程可以走不同路径 (有性能代价, 但功能正确)
  - 每个线程有自己的 PC (程序计数器) (Volta+)
```

### Volta 之前 vs 之后: 独立线程调度

**Pascal 及更早 (锁步执行)**:
```
Warp 中 32 个线程共享一个 Program Counter (PC)。
同一时刻所有线程必须执行同一条指令。

问题场景:
Thread 0: lock();   work_a();   unlock();
Thread 1: lock();   work_b();   unlock();

如果 Thread 0 持有锁，Thread 1 也要获取锁:
→ 死锁! 因为 Thread 1 和 Thread 0 共享同一个 PC，
  Thread 0 无法前进到 unlock() (Thread 1 卡在 lock())
```

**Volta+ (独立线程调度)**:
```
每个线程有自己的 PC 和调用栈。
线程可以在不同的代码位置执行。

Warp Scheduler 会将执行相同 PC 位置的线程组合在一起执行 (Reconvergence)。
不同 PC 位置的线程分批执行。

好处:
- 支持 Warp 内的细粒度同步 (如 Warp 级锁)
- Producer-Consumer 模式可行

代价:
- 需要显式 __syncwarp() 来保证汇合
- 某些依赖锁步假设的旧代码可能需要修改
- 每个线程需要额外的硬件资源 (PC + 调用栈)

重要: 这不意味着同一 Warp 的线程可以"真正并行"执行不同指令。
硬件还是只有一组执行单元。独立线程调度意味着 Scheduler 可以
选择执行 Warp 中某个子集的线程的指令，而不是必须执行所有 32 个线程的。
```

### 指令发射的详细过程

```
一个 Warp Scheduler 每周期的操作 (Ampere):

1. 从管辖的 Warp 中选择一个 Eligible Warp (Scoreboard 检查通过)

2. 获取该 Warp 的下一条指令 (从 I-Cache)

3. 解码指令:
   SASS 指令格式 (NVIDIA 的机器码):
   ┌──────────┬──────────┬──────────┬──────────┬──────┐
   │ Opcode   │ Dest Reg │ Src Reg1 │ Src Reg2 │ Pred │
   │ (操作码) │ (目标)    │ (源1)    │ (源2)    │(谓词)│
   └──────────┴──────────┴──────────┴──────────┴──────┘
   
   SASS 指令还包含控制信息:
   - Stall Count: 下一条指令要等多少周期才能发射 (编译器静态分析)
   - Yield Flag: 提示调度器可以切换到另一个 Warp
   - Write Barrier: 这条指令设置了内存写屏障
   - Read Barrier: 等待哪个屏障完成才能执行
   - Reuse Flag: 寄存器值在寄存器重用缓存中可复用
   
   这些控制位是编译器 (ptxas) 精心计算的! 好的指令调度 = 高性能。

4. 发射到执行单元:
   根据指令类型发射到:
   - FP32 Datapath (FFMA, FADD, FMUL, FMNMX, FSET, ...)
   - FP32/INT Datapath (可以做 FP32 或 INT32, 看指令)
   - FP64 Datapath (DFMA, DADD, DMUL, ...)
   - LD/ST Unit (LDG, STG, LDS, STS, ATOM, RED, ...)
   - SFU (MUFU: sin, cos, rsqrt, exp2, lg2, rcp, ...)
   - Tensor Core (HMMA, IMMA, DMMA, ...)
   - Uniform Datapath (统一标量操作, Ampere+)

5. 更新 Scoreboard:
   标记目标寄存器为"等待写入"
   当执行完成时，取消标记 → 依赖该寄存器的 Warp 变为 Eligible
```

### PTX 与 SASS — 两层指令集

```
你的 CUDA 代码经历以下编译过程:

.cu 源码
   │ nvcc (前端)
   ▼
PTX (Parallel Thread Execution) — 虚拟指令集
   │ ptxas (后端)
   ▼
SASS (Streaming ASSembler) — 真实机器码

PTX 的角色:
- NVIDIA 定义的虚拟 ISA (类似 Java bytecode)
- 和具体 GPU 架构无关
- 可以在运行时被 JIT 编译成 SASS
- 保证向前兼容: 老的 PTX 在新 GPU 上也能跑

SASS 的角色:
- 真正在 GPU 上执行的机器码
- 每代架构不同 (SM_70 和 SM_80 的 SASS 不同)
- 编译时确定具体调度 (stall count, yield, barrier)

查看 PTX:
nvcc -ptx kernel.cu -o kernel.ptx

查看 SASS:
nvcc -cubin kernel.cu -o kernel.cubin
cuobjdump -sass kernel.cubin

或者:
nvcc kernel.cu -o program
cuobjdump -sass program
```

### SASS 指令示例解读

```
看一个简单的 FMA 操作在 SASS 层面长什么样:

FFMA R4, R0, R2, R4;    // R4 = R0 * R2 + R4  (Fused Multiply-Add)
    [B------:R-:W-:-:S01]  ← 控制码

控制码含义:
B------  → 不等待任何 barrier
R-       → 不设置 Read Barrier
W-       → 不设置 Write Barrier
-        → 不 Yield
S01      → 下条指令 stall 1 个周期

另一个例子:
LDG.E.128 R4, [R2];     // 从全局内存加载 128 bit (4 个 float)
    [B------:R-:W0:-:S01]
    
W0 → 设置 Write Barrier 0 (后续指令可以等待这个 barrier)

DEPBAR.LE SB0, 0x0;     // 等待 barrier 0 完成 (数据加载完毕)
    [B0-----:R-:W-:-:S04]
    
B0 → 等待 barrier 0
S04 → stall 4 个周期 (数据加载的最小延迟)
```


## 2.4 CUDA Stream 与异步执行 — 完整理解

### Stream 的本质

Stream 是 GPU 命令的**有序队列**。每个 Stream 内的操作保持严格的先后顺序。

```
Stream 实际上是软件概念 + 硬件支持:

软件层 (CUDA Driver):
  维护每个 Stream 的命令队列
  跟踪依赖关系

硬件层:
  GPU 有多个独立的硬件引擎:
  ├── Copy Engine 0 (Host → Device 方向的 DMA)
  ├── Copy Engine 1 (Device → Host 方向的 DMA)
  └── Compute Engine (kernel 执行)
  
  不同引擎可以真正并行工作:
  Copy Engine 搬数据的同时, Compute Engine 跑 kernel
  
  同类操作 (如两个 kernel) 是否并发取决于:
  - 不同 Stream 中的 kernel 如果 SM 有空闲资源，可以并发
  - 但实际上大 kernel 通常会占满所有 SM，不留空间给其他 kernel
```

### Stream 并发的详细示例

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 流水线化的数据处理:
for (int i = 0; i < numChunks; i++) {
    int offset = i * chunkSize;
    cudaStream_t s = (i % 2 == 0) ? stream1 : stream2;
    
    cudaMemcpyAsync(d_in + offset, h_in + offset, 
                    chunkBytes, cudaMemcpyHostToDevice, s);
    kernel<<<grid, block, 0, s>>>(d_in + offset, d_out + offset, chunkSize);
    cudaMemcpyAsync(h_out + offset, d_out + offset,
                    chunkBytes, cudaMemcpyDeviceToHost, s);
}

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

```
时间轴 (理想情况):

Stream 1: [H2D_0] [Kernel_0] [D2H_0]         [H2D_2] [Kernel_2] [D2H_2]
Stream 2:         [H2D_1] [Kernel_1] [D2H_1]         [H2D_3] ...

Copy Eng:  H2D_0   H2D_1              D2H_0   D2H_1   H2D_2  ...
Compute:          Kernel_0  Kernel_1           Kernel_2  ...

重叠效果:
  H2D_1 和 Kernel_0 同时进行
  D2H_0 和 Kernel_1 同时进行
  → 数据传输延迟被隐藏
```

### Default Stream 的陷阱

```cuda
// Legacy Default Stream (默认):
// 所有没有指定 Stream 的操作都在 Stream 0 上
// Stream 0 会与所有其他 Stream 隐式同步!

kernel<<<grid, block>>>(args);           // Stream 0
kernel<<<grid, block, 0, stream1>>>(a);  // Stream 1
// Stream 1 必须等待 Stream 0 的 kernel 完成后才能开始!
// 因为 Stream 0 (legacy) 会同步所有 Stream

// Per-Thread Default Stream (推荐):
// 编译时加 --default-stream per-thread
// 或 #define CUDA_API_PER_THREAD_DEFAULT_STREAM (在 #include <cuda_runtime.h> 之前)
// 每个 CPU 线程有独立的默认 Stream，不会隐式同步其他 Stream
```

### Pinned Memory — 异步传输的前提

**异步传输必须使用 Pinned Memory!**

```cuda
// 普通 malloc 分配的内存:
float *h_data = (float*)malloc(size);
// 内存页可以被 OS 换出到磁盘 (page out)
// DMA 引擎无法直接访问 → GPU driver 必须先拷贝到内部 staging buffer
// → cudaMemcpyAsync 实际上变成同步的!

// Pinned Memory:
float *h_data;
cudaMallocHost(&h_data, size);  // 或 cudaHostAlloc
// 内存页被锁定在物理内存中，不会被换出
// DMA 引擎可以直接访问 → 真正的异步传输
// 传输速度也更快 (~60% 提升 on PCIe)

// 缺点: 
// - 占用物理内存，不可被 OS 换出
// - 分配/释放比 malloc 慢 (~10-100×)
// - 分配过多会导致系统内存压力

// 更高级: Write-Combined Memory
float *h_data;
cudaHostAlloc(&h_data, size, cudaHostAllocWriteCombined);
// CPU 写入快 (绕过 CPU cache)，但 CPU 读取极慢
// 适用于: CPU 只写、GPU 只读的场景 (如上传顶点数据)

// Mapped Memory (Zero-Copy)
float *h_data;
cudaHostAlloc(&h_data, size, cudaHostAllocMapped);
float *d_data;
cudaHostGetDevicePointer(&d_data, h_data, 0);
// GPU 直接通过 PCIe 访问主机内存，无需显式拷贝
// 适用于: 数据只访问一次，且量不大
// 不适用于: 数据被重复访问 (每次都走 PCIe, 极慢)
```

### CUDA Event — 精确计时与跨 Stream 同步

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
// 精度: ~0.5 微秒

// 跨 Stream 同步:
cudaEventRecord(event_A, streamA);
cudaStreamWaitEvent(streamB, event_A);
// streamB 中后续的操作会等待 event_A 完成
// streamA 不受影响，继续执行

// 查询事件状态 (非阻塞):
cudaError_t status = cudaEventQuery(event);
if (status == cudaSuccess) {
    // event 已完成
} else if (status == cudaErrorNotReady) {
    // event 还没完成
}
```

### CUDA Graph — 减少 Launch 开销

```cuda
// 问题: 小 kernel 的 launch 开销 (~5-10 μs) 可能比执行时间还长
// 如果有 100 个小 kernel → ~1 ms 浪费在 launch 上

// CUDA Graph: 录制一系列操作，一次提交
cudaGraph_t graph;
cudaGraphExec_t instance;

// 方法1: Stream Capture
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernelA<<<gridA, blockA, 0, stream>>>(args);
kernelB<<<gridB, blockB, 0, stream>>>(args);
kernelC<<<gridC, blockC, 0, stream>>>(args);
cudaStreamEndCapture(stream, &graph);

// 实例化 (编译图, 只需做一次)
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

// 回放 (每次 ~5 μs 启动整个图, 而不是 3 × 5 μs)
for (int i = 0; i < 1000; i++) {
    cudaGraphLaunch(instance, stream);
}

// 方法2: 手动构建图 (更灵活)
cudaGraphCreate(&graph, 0);
cudaGraphNode_t nodeA, nodeB;
// 添加 kernel 节点, 指定依赖关系...
// 可以表达复杂的 DAG (有向无环图) 依赖

// 更新图参数 (不需要重新实例化):
cudaGraphExecKernelNodeSetParams(instance, nodeA, &newParams);
```


## 2.5 Error Handling — 完整的错误处理策略

### CUDA 错误模型

```
CUDA 错误有两类:

1. 同步错误 (Synchronous Error):
   API 调用立即返回错误码
   例: cudaMalloc 失败 (显存不足)
       cudaMemcpy 参数非法
       kernel launch 配置非法 (如 blockDim > 1024)

2. 异步错误 (Asynchronous Error):
   kernel 在 GPU 上执行时产生的错误
   不能通过 launch 调用获取
   必须通过后续的同步操作检查
   
   例: kernel 中的非法内存访问 (越界)
       kernel 中的 assert 失败
       kernel 超时 (GPU Watchdog, 通常 ~2-5 秒, 仅有显示器的 GPU)
```

### 推荐错误检查宏

```cuda
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: [%d] %s\n",            \
                    __FILE__, __LINE__, (int)err, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define CUDA_CHECK_KERNEL()                                               \
    do {                                                                  \
        cudaError_t err = cudaGetLastError();                              \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "Kernel launch error at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
        err = cudaDeviceSynchronize();                                     \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "Kernel execution error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// 使用:
CUDA_CHECK(cudaMalloc(&ptr, size));
kernel<<<grid, block>>>(args);
CUDA_CHECK_KERNEL();  // 注意: 会触发同步, 仅调试时使用!
```

### Sticky Error — 致命错误

```
某些错误是 "Sticky" 的:
一旦发生，所有后续 CUDA 调用都会返回同一个错误码。

常见 Sticky Error:
- cudaErrorIllegalAddress: 非法内存访问
- cudaErrorLaunchTimeout: Kernel 执行超时
- cudaErrorECCUncorrectable: 不可纠正的 ECC 错误

处理方式:
cudaError_t err = cudaDeviceSynchronize();
if (err != cudaSuccess) {
    fprintf(stderr, "Fatal: %s\n", cudaGetErrorString(err));
    cudaDeviceReset();  // 重置设备 (丢失所有 GPU 数据!)
    // 需要重新初始化
}
```

### GPU 端调试: assert 和 printf

```cuda
__global__ void kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // GPU 上的 assert (触发时会导致 cudaErrorAssert)
    assert(idx < n);
    
    // GPU 上的 printf (缓冲区有限, ~1MB)
    // 仅用于调试小规模数据!
    if (idx == 0) {
        printf("Block %d, Thread %d: data[0] = %f\n", 
               blockIdx.x, threadIdx.x, data[0]);
    }
}

// compute-sanitizer: CUDA 的内存检查工具 (类似 CPU 的 Valgrind)
// 命令: compute-sanitizer --tool memcheck ./my_program
// 检测: 越界访问、未初始化读取、竞争条件等
// 性能影响: 约 10-100× 减速, 仅调试使用
```


## 2.6 Grid-Stride Loop — 深度分析

### 朴素 vs Grid-Stride 对比

```cuda
// 朴素方式: 每个线程处理一个元素
__global__ void add_naive(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
// n = 100M, blockSize = 256 → gridSize = 390,625
// 390,625 个 Block 需要调度!

// Grid-Stride Loop: 每个线程处理多个元素
__global__ void add_stride(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}
// gridSize = 108 * 4 = 432, 总线程 = 432 * 256 = 110,592
// 每个线程处理 ~905 个元素
```

### Grid-Stride Loop 的六大优势

```
1. 减少调度开销:
   390,625 Block → 432 Block, 调度压力小很多
   
2. 消除 Tail Effect:
   gridSize = SM数 × 每SM最优Block数
   所有 SM 都有相同数量的 Block → 完美负载均衡
   
3. 合并访问保持:
   每一轮迭代中:
   Thread 0 → data[0],     Thread 1 → data[1],     ... (第1轮)
   Thread 0 → data[stride], Thread 1 → data[stride+1], ... (第2轮)
   每轮都是连续地址 → 合并访问 ✓

4. L2 Cache 友好:
   如果 stride * sizeof(float) < L2 size:
   第 2 轮可能命中 L1/L2 Cache 中第 1 轮的数据

5. 可调试:
   gridSize=1, blockSize=1 → 完全串行执行, 可 printf

6. 向量化兼容:
   配合 float4:
   for (int i = idx; i < n/4; i += stride) {
       float4 a4 = reinterpret_cast<const float4*>(a)[i];
       ...
   }
```

### gridSize 最优选择

```cuda
// 查询设备属性
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
int blocksPerSM;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel, blockSize, 0);

// 方法1: 刚好填满所有 SM
int gridSize = numSMs * blocksPerSM;

// 方法2: 2-4 波 (wave), 给调度器更多灵活性
int gridSize = numSMs * blocksPerSM * 2;

// 方法3: 让 CUDA 自动选择
int gridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, kernel, 0, 0);
```


## 2.7 Unified Memory 与内存管理高级话题

### Unified Memory — 自动地址翻译

```cuda
float *data;
cudaMallocManaged(&data, size);  // CPU 和 GPU 用同一个指针

for (int i = 0; i < n; i++) data[i] = i;  // CPU 访问
kernel<<<grid, block>>>(data);              // GPU 访问
cudaDeviceSynchronize();
printf("%f\n", data[0]);                    // CPU 访问结果
```

### Unified Memory 底层机制

```
底层: 基于 GPU 页表和 CPU 页表的统一管理

1. cudaMallocManaged 分配时:
   - 在 GPU 页表和 CPU 页表中都创建映射
   - 初始数据可能在 CPU 内存

2. GPU 访问时:
   - GPU MMU (Memory Management Unit) 查页表
   - 如果页不在 GPU 显存 → TLB Miss → Page Fault
   - GPU Page Fault 触发页面迁移:
     a. 通知 CPU driver
     b. CPU 取消映射该页
     c. 通过 PCIe/NVLink 将页面传到 GPU 显存
     d. GPU 更新页表
     e. 重试访问
   - 整个过程 ~20-50 μs (!!!)

3. CPU 访问时:
   - 同理，如果页在 GPU → 迁移回 CPU

4. 页面粒度:
   - 默认 4KB (小页)
   - 可以用 2MB 大页 (减少 TLB miss, 但迁移量更大)
   - cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device);
```

### Prefetch 与 Advise — 避免 Page Fault

```cuda
float *data;
cudaMallocManaged(&data, size);

// 初始化 (CPU)
for (int i = 0; i < n; i++) data[i] = i;

// 在 kernel 之前预迁移到 GPU
cudaMemPrefetchAsync(data, size, deviceId, stream);

kernel<<<grid, block, 0, stream>>>(data);

// 在 CPU 读取之前预迁移回来
cudaMemPrefetchAsync(data, size, cudaCpuDeviceId, stream);
cudaStreamSynchronize(stream);
printf("%f\n", data[0]);

// Advise: 告诉 runtime 数据的使用模式
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, deviceId);
// "这块数据主要是只读的" → GPU 和 CPU 各保留副本, 不用迁移
// 写入时触发 invalidation

cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, deviceId);
// "这块数据会被 GPU 访问" → runtime 可以预先建立映射
```

### 生产环境的内存管理策略

```
推荐方案 (按场景):

1. 性能关键路径 → 手动管理 (cudaMalloc + cudaMemcpy)
   - 完全控制传输时机
   - 可以和计算重叠
   - 代码量多但性能最好

2. 开发/原型阶段 → Unified Memory + Prefetch
   - 代码简单
   - 加 Prefetch 后性能接近手动管理
   - 适合快速验证算法

3. 多 GPU → Unified Memory + MemAdvise
   - 数据自动在多个 GPU 间迁移
   - 用 Advise 控制数据放置策略

4. GPU Direct Storage (GDS):
   - 数据直接从 NVMe SSD → GPU 显存, 不经过 CPU
   - 用于大数据集 (数据集 > CPU 内存)
   
5. GPU Direct RDMA:
   - 数据直接从网卡 → GPU 显存
   - 用于分布式训练
```


## 2.8 本章总结

```
Kernel Launch 全路径:
  CPU: 打包参数 → 写 Command Buffer → Doorbell → 立即返回 (~5μs)
  GPU: Command Processor → CTA Scheduler → SM Resource Check → Slot 分配
  SM:  寄存器分区 + Smem 划分 + Warp 初始化 + Barrier 初始化
  执行: I-Cache 取指 → Decode → Scoreboard 检查 → 发射 → 流水线执行

关键设计理念:
  异步执行: CPU 不等 GPU → launch 开销被掩盖
  Stream: 有序队列, 内顺序外并发 → 计算与传输重叠
  Grid-Stride Loop: 固定 Block 数, 每线程多元素 → 消除 Tail Effect
```


## 2.9 Q&A

### Q: Kernel Launch 开销 ~5μs 是哪里花的? 能减少吗?

```
~3-5μs: CPU 端 CUDA Runtime 的参数打包 + Driver 写 Command Buffer
~1-2μs: GPU 端命令解析 + CTA 分配 + SM 初始化

减少方法:
  CUDA Graph: 录制多个 kernel 的调用序列, 回放时只有 1 次 launch 开销 (~1μs)
  Persistent Kernel: 1 次 launch, 内部循环领取任务 → 0 额外开销
  合并 Kernel: 将多个小 kernel 融合成 1 个
```

### Q: Stream 0 为什么会和其他 Stream 隐式同步?

```
Legacy Default Stream (Stream 0) 的行为:
  它不是一个“普通 Stream” — 它是一个特殊的同步点。
  在 Stream 0 上发起的操作会等待所有其他 Stream 的前序操作完成,
  且其他 Stream 的后序操作会等待 Stream 0 完成。
  → Stream 0 会打破并发!

解决: 永远显式创建 Stream, 不用默认 Stream。
或者: 编译时加 --default-stream per-thread → 每个 CPU 线程有独立默认 Stream。
```

### 概念辨析: "Block" vs "CTA" vs "Workgroup"

```
Block:     CUDA 编程模型中的术语 (软件概念)
CTA:       NVIDIA 硬件文档中的术语 (Cooperative Thread Array)
Workgroup: OpenCL 中的等价术语

三者是同一个东西!不同社区/文档用不同名字。
在 NVIDIA 的 SASS 和硬件描述中用 CTA, 在 CUDA C++ 中用 Block。

ncu 中的 "Launched CTAs" = 你 launch 的 Block 总数。
```


## 2.10 练习题

配套代码在 [`theory/exercises/`](./exercises/) 目录下: [`ch02_ex1_blocksize.cu`](./exercises/ch02_ex1_blocksize.cu) / [`ch02_ex2_2d_index.cu`](./exercises/ch02_ex2_2d_index.cu) / [`ch02_ex3_async.cu`](./exercises/ch02_ex3_async.cu)

### 练习 1: blockSize 对性能的影响 [难度: ⭐]

```
打开 01_vector_add/vector_add.cu:

1. 将 blockSize 分别改为 32, 64, 128, 256, 512, 1024。
   每次重新编译运行, 观察:
   - gridSize 怎么变? (总线程数要 ≥ N, 所以 gridSize = ceil(N/blockSize))
   - 性能有变化吗? 哪个最快?

2. 试试 blockSize=100 (不是 32 的倍数)。能跑吗?
   (提示: 可以跑, 但最后一个 Warp 只有 100%32=4 个活跃线程, 其余 28 个空转)
   (配合理论: 本章 2.2 "blockSize 深度分析")
```

### 练习 2: 2D Grid 索引计算 [难度: ⭐⭐]

```
打开 02_matrix_mul/matmul.cu:

1. 在 matmul_naive 的 kernel 开头加一行:
   if (row == 0 && col == 0)
       printf("Block(%d,%d) Thread(%d,%d) -> row=%d col=%d\n",
              blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row, col);
   
   预测: 它会打印什么? row 和 col 分别是多少?
   (提示: row = blockIdx.y * TILE_SIZE + threadIdx.y,
         col = blockIdx.x * TILE_SIZE + threadIdx.x)

2. 思考: 为什么是 threadIdx.x 对应列 (col) 而不是行?
   如果反过来会怎样? (提示: 同一 Warp 的线程 threadIdx.x 连续,
   让它们访问连续的列 → 内存地址连续 → 合并访问! 见 Ch3.4)
```

### 练习 3: 异步执行观察 [难度: ⭐⭐]

```
写一个小程序:

kernel<<<grid, block>>>(d_data, N);
printf("CPU: kernel 已启动\n");
cudaDeviceSynchronize();
printf("CPU: kernel 已完成\n");

观察: 第一个 printf 是否在 GPU 计算完成前就打印了?
(是的! kernel launch 是异步的 — CPU 不等 GPU 执行完就继续。)

思考: 这意味着什么?
  - cudaMemcpy(D2H) 会隐式同步 (等 GPU 完成后才拷贝)
  - 连续启动多个 kernel 不会等前一个完成 (它们在 GPU 上排队)
  - 这就是为什么 launch 开销只有 ~5μs — CPU 只是往队列里塞了一个命令
(配合理论: 本章 2.4 "CUDA Stream 与异步执行" 和 2.0 "Launch 全路径")
```
