# 向量加法：从源码到 GPU 执行的完整旅程

本文档配合 `vector_add.cu` 阅读。解释这段代码从你按下回车编译，
到 GPU 上亿万个晶体管真正执行加法，中间经历的每一步。

> **软硬件映射总览**: Grid/Block/Thread 如何对应到 SM/Warp/CUDA Core，
> 以及 Warp Scheduler 的每周期调度行为，见 [`theory/software_hardware_mapping.md`](../theory/software_hardware_mapping.md)。
> 本文档聚焦于**向量加法这个具体程序**的编译和执行过程。


## 第一阶段：编译 — 源码变成 GPU 可执行程序

当你运行 `nvcc -O2 -o vector_add vector_add.cu` 时，发生了以下过程：

```
vector_add.cu (你写的源码，混合了 CPU 代码和 GPU 代码)
       │
       ▼
 ┌─── nvcc 前端 (cudafe++) ───┐
 │                             │
 │  把 .cu 文件拆成两部分:      │
 │  ┌───────────┐ ┌──────────┐│
 │  │ Host 代码 │ │Device代码││
 │  │ (main等)  │ │(__global__)│
 │  └─────┬─────┘ └─────┬────┘│
 └────────┼──────────────┼─────┘
          │              │
          ▼              ▼
   系统 C++ 编译器     CUDA 编译器
   (gcc/clang)        (cicc → ptxas)
          │              │
          │         ┌────┴────┐
          │         │  PTX    │  ← GPU 的"虚拟汇编" (类似 Java 字节码)
          │         │ (.ptx)  │    和具体 GPU 架构无关
          │         └────┬────┘    可以在运行时被 JIT 编译到任何架构
          │              │
          │              ▼
          │         ┌────────┐
          │         │  SASS  │  ← GPU 的"真实机器码" (类似 x86 汇编)
          │         │(.cubin)│    针对特定架构 (如 sm_70 = Volta)
          │         └────┬───┘    包含每条指令的具体编码
          │              │
          ▼              ▼
   ┌──────────────────────────┐
   │    Fatbinary (胖二进制)    │  ← 打包在一起: host 目标码 + cubin + PTX
   │  包含多种架构的 GPU 代码   │    运行时选择匹配的版本
   └────────────┬─────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │   最终可执行文件           │
   │   vector_add             │  ← 看起来是普通可执行文件
   │   (ELF 格式, Linux)       │    但内部嵌入了 GPU 二进制
   └──────────────────────────┘
```

### 关键细节：PTX 和 SASS 是什么？

```
PTX (Parallel Thread Execution):
  NVIDIA 定义的 GPU "虚拟指令集"。
  类比: Java 的字节码 → 不直接在硬件上跑，需要再编译一次。
  
  向量加法的 kernel 编译成 PTX 大概长这样:
  
  .visible .entry vector_add(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
  ) {
    .reg .f32 %f<4>;          // 声明浮点寄存器
    .reg .u32 %r<6>;          // 声明整数寄存器
    .reg .u64 %rd<8>;         // 声明 64-bit 寄存器
    
    mov.u32 %r1, %tid.x;              // r1 = threadIdx.x
    mov.u32 %r2, %ctaid.x;            // r2 = blockIdx.x
    mov.u32 %r3, %ntid.x;             // r3 = blockDim.x
    mad.lo.u32 %r4, %r2, %r3, %r1;   // r4 = blockIdx.x * blockDim.x + threadIdx.x
    
    setp.ge.u32 %p1, %r4, %r5;        // if (idx >= n)
    @%p1 bra EXIT;                     //   goto EXIT
    
    ld.global.f32 %f1, [%rd1];        // f1 = a[idx]
    ld.global.f32 %f2, [%rd2];        // f2 = b[idx]
    add.f32 %f3, %f1, %f2;            // f3 = a[idx] + b[idx]
    st.global.f32 [%rd3], %f3;        // c[idx] = f3
    
    EXIT: ret;
  }

SASS (Streaming ASSembler):
  GPU 的"真实机器指令"。每代 GPU 架构不同。
  PTX 被 ptxas 编译器翻译成 SASS, 做了指令调度、寄存器分配等优化。
  
  同一段 kernel 的 SASS (Volta sm_70) 大致是:
  
  /*0000*/  S2R R0, SR_TID.X ;          // R0 = threadIdx.x (读硬件特殊寄存器)
  /*0010*/  S2R R3, SR_CTAID.X ;        // R3 = blockIdx.x
  /*0020*/  IMAD R0, R3, c[0x0][0x0], R0 ; // R0 = R3 * blockDim + R0 (用常量)
  /*0030*/  ISETP.GE.AND P0, PT, R0, c[0x0][0x168], PT ; // P0 = (idx >= n)
  /*0040*/  @P0 EXIT ;                  // if (P0) 退出
  /*0050*/  LDG.E R2, [R4] ;            // R2 = a[idx] (从全局显存加载)
  /*0060*/  LDG.E R5, [R6] ;            // R5 = b[idx]
  /*0070*/  FADD R2, R2, R5 ;           // R2 = R2 + R5
  /*0080*/  STG.E [R8], R2 ;            // c[idx] = R2 (写回全局显存)
  /*0090*/  EXIT ;

  注意: SASS 指令还附带控制码 (Stall Count, Yield, Barrier 等),
  这些控制码告诉硬件调度器"下一条指令要等几个周期""可以切换 Warp 吗"。
  ptxas 编译器精心计算这些控制码 → 直接影响性能。
```

### 你可以自己看 PTX 和 SASS

```bash
# 生成 PTX (人类可读的 GPU 虚拟汇编)
nvcc -ptx vector_add.cu -o vector_add.ptx
cat vector_add.ptx  # 打开看看

# 生成并查看 SASS (真实机器码的反汇编)
nvcc -cubin -arch=sm_70 vector_add.cu -o vector_add.cubin
cuobjdump -sass vector_add.cubin

# 或者对编译好的可执行文件直接看 SASS
cuobjdump -sass vector_add
```


## 第二阶段：加载 — 可执行文件被操作系统和 CUDA Runtime 加载

```
当你运行 ./vector_add 时:

┌─ 操作系统 (Linux) ─────────────────────────────────────┐
│                                                         │
│  1. 加载 ELF 可执行文件到 CPU 内存                        │
│     ├── .text 段: CPU 的 main() 等函数的机器码            │
│     ├── .rodata 段: 常量数据                             │
│     └── .nv_fatbin 段: 嵌入的 GPU 二进制 (fatbinary)      │
│                                                         │
│  2. 动态链接器加载共享库:                                  │
│     ├── libcudart.so (CUDA Runtime)                      │
│     ├── libcuda.so (CUDA Driver)                         │
│     └── libc.so 等                                       │
│                                                         │
│  3. 跳转到 main() 开始执行                                │
└─────────────────────────────────────────────────────────┘

main() 执行过程中, CUDA Runtime 在幕后做的事:

┌─ CUDA Runtime (libcudart.so) ──────────────────────────┐
│                                                         │
│  4. 第一次调用任何 CUDA 函数时 (如 cudaMalloc):           │
│     ├── 初始化 CUDA Context (GPU 的"执行环境"):           │
│     │   ├── 建立 GPU 页表 (虚拟地址 → 物理地址映射)       │
│     │   ├── 分配 GPU 命令队列 (Command Buffer)           │
│     │   └── 设置默认 Stream                              │
│     │                                                   │
│     ├── 从 .nv_fatbin 中提取 GPU 二进制:                  │
│     │   ├── 如果有匹配当前 GPU 的 cubin → 直接用           │
│     │   └── 如果没有 → 找 PTX → JIT 编译成 SASS          │
│     │                                                   │
│     └── 将 kernel 代码加载到 GPU 的指令内存               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```


## 第三阶段：执行 — main() 中每一行代码在硬件上做了什么

### malloc + 初始化 (纯 CPU 操作)

```c
float *h_a = (float *)malloc(bytes);  // 在 CPU 内存分配 4MB
for (int i = 0; i < N; i++) h_a[i] = i;  // CPU 填入数据
```

```
硬件变化:
  ┌── CPU ────────────────────────────────┐
  │ malloc() → 操作系统分配 4MB 虚拟内存   │
  │         → 物理页面可能是懒分配的        │
  │            (第一次写入时才分配物理页)     │
  │                                        │
  │ for 循环写入 → CPU L1/L2 Cache 填充    │
  │             → 最终写回 CPU DRAM         │
  └────────────────────────────────────────┘
  
  ┌── GPU ────────────────────────────────┐
  │ (无变化, GPU 还不知道这件事)            │
  └────────────────────────────────────────┘
```

### cudaMalloc (GPU 显存分配)

```c
float *d_a;
cudaMalloc(&d_a, bytes);  // 在 GPU 显存分配 4MB
```

```
硬件变化:
  ┌── CPU ────────────────────────────────┐
  │ cudaMalloc() 调用 CUDA Driver          │
  │ Driver 在 GPU 的页表中分配虚拟地址      │
  │ 物理 HBM 页面可能是懒分配的             │
  │ 返回 d_a = 一个指向 GPU 显存的指针     │
  │ (CPU 不能解引用这个指针!)              │
  └────────────────────────────────────────┘
  
  ┌── GPU ────────────────────────────────┐
  │ GPU 的 Memory Controller 更新页表      │
  │ HBM 中预留了 4MB 的地址范围            │
  │ (但物理页面可能还未分配)               │
  └────────────────────────────────────────┘
```

### cudaMemcpy Host → Device (数据搬运)

```c
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
```

```
硬件变化 — 数据从 CPU 内存流向 GPU 显存:

  CPU 内存 (DRAM)
      │
      │ CPU DMA Engine 读取 h_a 的 4MB 数据
      │ (从 CPU DRAM → PCIe Root Complex)
      │
      ▼
  PCIe Gen4 x16 总线 (~32 GB/s)
      │
      │ 数据通过 PCIe 链路传输
      │ 每个 TLP (Transaction Layer Packet) 携带 128-256 字节有效载荷
      │ 4MB / 256B = ~16000 个 TLP
      │
      ▼
  GPU 的 PCIe 接口 (Host Interface)
      │
      │ GPU 的 Copy Engine (DMA 引擎) 接收数据
      │ Copy Engine 独立于计算引擎 → 可以和计算并行
      │
      ▼
  GPU Memory Controller
      │
      │ 将数据写入 HBM 的物理页面
      │ 通过 HBM 的 Channel/Bank 分散存储
      │
      ▼
  HBM (GPU 显存)
      d_a 指向的 4MB 区域现在有了和 h_a 一样的数据
  
  耗时: 4MB / 32 GB/s ≈ 0.125 ms (PCIe 传输)
  实际可能更长 (PCIe 协议开销 + 初始化延迟)
```

### kernel launch — 最关键的一步

```c
vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
```

```
这一行触发的完整硬件流程:

═══ CPU 端 (~5 μs) ═══════════════════════════════════════

  CPU 执行 CUDA Runtime 代码:
    1. 打包 Kernel 参数: {d_a, d_b, d_c, N} → 写入参数缓冲区
    2. 构造 Launch Descriptor:
       ├── kernel 函数地址 (GPU 端)
       ├── gridDim = {4096, 1, 1}
       ├── blockDim = {256, 1, 1}
       ├── shared memory = 0
       └── stream = default
    3. 将 Descriptor 写入 GPU 的 Command Buffer (通过 MMIO)
    4. 写 Doorbell Register 通知 GPU: "有新命令!"
    5. CPU 立即返回 ← 不等 GPU 执行完!

═══ GPU 端 ═══════════════════════════════════════════════

  ┌─ GPU Command Processor ─────────────────────────┐
  │ 检测到 Doorbell → 从 Command Buffer 读取命令      │
  │ 解析 Launch Descriptor                           │
  │ 参数写入 Constant Memory Bank 0 (CB0)             │
  │ 传给 CTA Scheduler (GigaThread Engine)           │
  └──────────────────────────┬──────────────────────┘
                             │
  ┌─ CTA Scheduler ─────────┴──────────────────────┐
  │ 知道要分配 4096 个 Block, 每个 256 线程            │
  │                                                   │
  │ 对每个 SM 检查资源:                                │
  │   寄存器够? Shared Memory 够? Warp 槽位够?        │
  │                                                   │
  │ 开始分配:                                          │
  │   Block 0 → SM 0                                  │
  │   Block 1 → SM 1                                  │
  │   ...                                             │
  │   Block 107 → SM 107                              │
  │   Block 108 → SM 0 (第二波, SM 0 可能已跑完 Block 0)│
  │   ...                                             │
  └────────────────────────────────────────────────────┘
                             │
  ┌─ SM 接收 Block ──────────┴─────────────────────┐
  │ 以 Block 0 分配到 SM 0 为例:                      │
  │                                                   │
  │ 1. 分配资源:                                       │
  │    ├── 256 个线程 = 8 个 Warp                      │
  │    ├── 8 × 48 寄存器 × 32 = 12288 个寄存器        │
  │    └── 0 字节 Shared Memory (本 kernel 不用)       │
  │                                                   │
  │ 2. 初始化 Warp:                                    │
  │    Warp 0 (Thread 0-31):                          │
  │      PC = kernel 入口地址                          │
  │      threadIdx.x = 0,1,2,...,31                    │
  │      blockIdx.x = 0                               │
  │    Warp 1 (Thread 32-63): threadIdx.x = 32-63     │
  │    ...                                            │
  │    Warp 7 (Thread 224-255)                        │
  │                                                   │
  │ 3. 所有 8 个 Warp 变为 "Eligible" (可被调度)       │
  └────────────────────────────────────────────────────┘
                             │
  ┌─ Warp Scheduler 开始调度 ─┴────────────────────┐
  │                                                  │
  │ 每个时钟周期, Warp Scheduler 做:                  │
  │   1. 查看所有 Eligible Warp                       │
  │   2. 选一个发射它的下一条指令                      │
  │                                                  │
  │ 以 Warp 0 为例, 逐指令执行:                       │
  └──────────────────────────────────────────────────┘
```

### 向量加法的 Warp 调度实景

```
以 SM 0 上 Block 0 (256 线程 = 8 Warp) 为例。
SM 0 的 PB 0 管理 Warp 0 和 Warp 4。

向量加法 kernel 的指令序列 (每 Warp 相同):
  指令 0: S2R R0, SR_TID.X        (读 threadIdx.x, 4 cyc)
  指令 1: S2R R3, SR_CTAID.X      (读 blockIdx.x, 4 cyc)
  指令 2: IMAD R0, R3, 256, R0    (idx = blockIdx*256 + threadIdx, 4 cyc)
  指令 3: ISETP.GE P0, R0, N      (边界检查, 4 cyc)
  指令 4: @P0 EXIT                 (条件退出)
  指令 5: LDG.E R2, [addr_a]      (加载 a[idx], ~500 cyc!)
  指令 6: LDG.E R5, [addr_b]      (加载 b[idx], ~500 cyc!)
  指令 7: FADD R2, R2, R5         (a+b, 但要等 LDG 回来!)
  指令 8: STG.E [addr_c], R2      (写 c[idx])
  指令 9: EXIT

PB 0 的 Warp Scheduler 调度时间轴:

Cycle    选中 Warp   执行的指令          Warp 0 状态      Warp 4 状态
─────    ─────────   ──────────          ──────────       ──────────
0        Warp 0      S2R (threadIdx)     Executing        Eligible
1        Warp 0      S2R (blockIdx)      Executing        Eligible
2        Warp 0      IMAD (计算 idx)     Executing        Eligible
3        Warp 0      ISETP (边界检查)    Executing        Eligible
4        Warp 0      LDG a[idx]          → Stall (等HBM)  Eligible
                     ↓ Scoreboard 标记 R2 pending
                     ↓ Warp 0 下一条 LDG 也依赖不了 R2, 但可以发射
5        Warp 0      LDG b[idx]          → Stall (等HBM)  Eligible
                     ↓ R2 和 R5 都 pending
                     ↓ 下一条 FADD 依赖 R2,R5 → Warp 0 彻底 Stall
6        Warp 4      S2R (threadIdx)     Stall-Long       Executing
7        Warp 4      S2R (blockIdx)      Stall-Long       Executing
8        Warp 4      IMAD                Stall-Long       Executing
9        Warp 4      ISETP               Stall-Long       Executing
10       Warp 4      LDG a[idx]          Stall-Long       → Stall
11       Warp 4      LDG b[idx]          Stall-Long       → Stall
12       (无 Eligible Warp!)              Stall-Long       Stall-Long
         → PB 0 空闲! 在等 HBM 数据!
         → 这就是为什么只有 2 Warp 时延迟无法完全隐藏
...
~500     Warp 0 的 LDG 数据回来     → Eligible!       Stall-Long
501      Warp 0      FADD (a+b)         Executing        Stall-Long
502      Warp 0      STG (写 c)         Executing        Stall-Long
503      Warp 0      EXIT               完成!            Stall-Long
~505     Warp 4 的数据也回来        (已完成)          → Eligible
506      Warp 4      FADD               (已完成)         Executing
507      Warp 4      STG                                  Executing
508      Warp 4      EXIT                                 完成!

总耗时 ≈ 508 cycles (per PB)
理想耗时 (如果零延迟): ~10 cycles × 2 Warp = 20 cycles
→ 效率 = 20/508 ≈ 4%!
→ 这就是向量加法是 Memory Bound 的本质: 99% 的时间在等 HBM!

如果 SM 上有 5 Block (每 PB 10 Warp):
  Warp 0 Stall 后, 还有 Warp 4,8,12,...36 可以执行
  10 个 Warp × ~6 条指令 × ~1 cyc = ~60 cycles 的有用工作
  vs HBM 延迟 ~500 cycles
  → 还是不够! 需要 ~500/6 ≈ 83 Warp 才能完全隐藏
  → 向量加法天然 Memory Bound, 再多 Warp 也无法突破带宽上限
```


### 逐指令在硬件上的执行

```
Warp 0 (Thread 0-31) 执行 kernel 代码:

指令 1: S2R R0, SR_TID.X
  含义: R0 = threadIdx.x
  硬件: 从硬件特殊寄存器读取 (初始化时已设置好)
  结果: Thread 0 的 R0=0, Thread 1 的 R0=1, ..., Thread 31 的 R0=31
  延迟: ~4 cycles

指令 2: S2R R3, SR_CTAID.X
  含义: R3 = blockIdx.x
  硬件: 读硬件特殊寄存器
  结果: Warp 0 的所有 32 个线程的 R3 都 = 0 (它们都在 Block 0)
  
指令 3: IMAD R0, R3, c[0x0][0x0], R0
  含义: R0 = R3 × blockDim.x + R0  (即 idx = blockIdx.x * 256 + threadIdx.x)
  硬件: c[0x0][0x0] 从 Constant Memory 读取 blockDim.x = 256
        Constant Memory 有专用缓存 → 32 线程读同一值 → 广播, 1 cycle
        IMAD = Integer Multiply-Add, 在 INT32 ALU 执行
  结果: Thread 0 的 R0=0, Thread 1 的 R0=1, ..., Thread 31 的 R0=31

指令 4: ISETP.GE P0, R0, c[0x0][0x168]
  含义: P0 = (idx >= n)
  硬件: c[0x0][0x168] 是 kernel 参数 n = 1048576, 从 Constant Memory 读取
        比较结果写入谓词寄存器 P0 (每线程 1 bit)
  结果: 所有 32 线程的 idx (0-31) 都 < n → P0 = false

指令 5: @P0 EXIT
  含义: if (P0) 退出
  硬件: P0 为 false → 不退出, 继续执行
        如果某些线程的 P0 为 true (最后一个 Block 的多余线程),
        它们会被标记为 inactive (Active Mask 中对应 bit 清零),
        不参与后续指令。

指令 6: LDG.E R2, [R4]    ← 加载 a[idx]
  含义: 从全局显存加载 a[idx] 到寄存器 R2
  硬件流程 (这是最复杂的一条指令!):
  
  ┌─ LD/ST Unit ────────────────────────────────────────┐
  │ 32 个线程各提供地址: a + 0*4, a + 1*4, ..., a + 31*4 │
  │ (连续地址! 每个相差 4 字节)                            │
  │                                                       │
  │ Coalescing Logic (合并逻辑):                           │
  │   32 × 4B = 128B, 全部落在 1 个 128B Cache Line 内     │
  │   → 合并成 1 次内存事务! (最优情况)                     │
  │                                                       │
  │ 查 L1 Cache:                                           │
  │   首次访问 → Miss                                      │
  │   分配 MSHR entry → 向 L2 发请求                       │
  │                                                       │
  │ Scoreboard 标记 R2 为 "pending" (等待数据)              │
  │ Warp 0 进入 Stall (不能执行下一条指令, 因为依赖 R2)     │
  │ → Warp Scheduler 切换到 Warp 1 继续执行其他 Warp!      │
  └───────────────────────────────────────────────────────┘
  
  数据在网络中的旅程:
  
  SM 的 L1 Cache                    Warp 0 在等
       │ (L1 Miss)                  Warp 1-7 在跑
       ▼                            (延迟隐藏!)
  MSHR → NoC (片上网络)
       │ ~30 cycles 传输
       ▼
  L2 Cache Slice
       │ (如果 L2 Hit) → 数据返回 → ~200 cycles 总延迟
       │ (如果 L2 Miss) ↓
       ▼
  Memory Controller
       │ 排队 + DRAM 访问
       ▼
  HBM (物理显存)
       │ Row Activate + Read → ~50ns
       ▼
  数据回传: HBM → MC → L2 → NoC → SM L1 → 寄存器 R2
  总延迟: ~500-800 cycles (~350-560 ns)
  
  在这 500 cycles 中, Warp Scheduler 一直在执行其他 Warp!
  等数据回来, R2 从 "pending" 变为 "ready", Warp 0 变回 Eligible。

指令 7: LDG.E R5, [R6]    ← 加载 b[idx]
  同上, 又是一次全局内存加载。
  如果和指令 6 是背靠背发射的 (ILP), 两次加载可以在飞行中重叠。

指令 8: FADD R2, R2, R5    ← a[idx] + b[idx]
  含义: R2 = R2 + R5 (浮点加法)
  硬件: 在 FP32 ALU 执行, 4 级流水线, 延迟 4 cycles
        32 个线程各自做自己的加法 (SIMT: 同一指令, 不同数据)
        FP32 Core 每 Processing Block 有 16 个, 一个 Warp 32 线程
        → 需要 2 个 cycle 完成 (前 16 线程 → 后 16 线程)

指令 9: STG.E [R8], R2    ← c[idx] = 结果
  含义: 将 R2 写入全局显存 c[idx]
  硬件: LD/ST Unit 将 32 个写请求合并成 1 个 128B 写事务
        写入不需要等完成 (fire-and-forget), Warp 可以继续执行下一条指令
        数据经过: SM → NoC → L2 (写入) → 最终回写 HBM

指令 10: EXIT
  含义: 这个 Warp 执行完毕
  硬件: Warp 0 的所有线程变为 inactive
        SM 检查: Block 0 的 8 个 Warp 都 EXIT 了吗?
        如果都退出了 → 释放 Block 0 的所有资源 (寄存器, 槽位)
        → CTA Scheduler 可以分配新 Block 到这个 SM
```


## 第四阶段：结果返回 CPU

```c
cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
```

```
这个调用会等待 GPU 上所有操作完成 (隐式同步), 然后:

  HBM (GPU 显存)
      │
      │ GPU Copy Engine 读取 d_c 的 4MB 数据
      │ (和计算引擎独立, 不影响其他 kernel)
      │
      ▼
  GPU PCIe 接口
      │
      │ 通过 PCIe Gen4 x16 传输
      │ ~32 GB/s → 4MB / 32GB/s ≈ 0.125 ms
      │
      ▼
  CPU PCIe Root Complex
      │
      │ DMA 写入 CPU 内存
      │
      ▼
  CPU 内存 (DRAM)
      h_c 现在有了 GPU 计算的结果
```


## 完整时间线 (所有阶段)

```
时间 ──────────────────────────────────────────────────────────►

CPU:  [malloc+init] [cudaMalloc] [Memcpy H2D] [launch] [等待] [Memcpy D2H] [验证]
                                     │           │   ↗
                                     │           │  │
GPU:                                 │      [Command] [CTA分配] [Warp执行...] [完成]
                                     │      Proc.
                                     │
PCIe:               ←──── H2D 传输 ────→                    ←── D2H 传输 ──→
                                    
                    │← ~0.125ms →│   │←5μs→│  │←── ~0.1ms ──→│ │← ~0.125ms →│

注意:
  kernel launch 只要 ~5μs (CPU 端), 但 GPU 实际执行要 ~0.1ms。
  CPU 在 launch 后立即返回, GPU 在后台执行。
  直到 cudaMemcpy(D2H) 时 CPU 才等待 GPU 完成 (隐式同步)。
```


## GPU 越界访问 — 为什么不报错而是静默损坏数据

```
在 CPU 上, 数组越界 = segfault (操作系统检测到访问了未映射的页)

在 GPU 上, 数组越界 ≠ 立即崩溃!
  GPU 没有操作系统来检测非法访问。
  越界写入的后果:
    1. 写到同一个 cudaMalloc 分配的块内 → 数据被悄无声息地破坏
       → 后续 kernel 读到错误数据 → 结果莫名其妙地不对
    2. 写到未分配区域 → 取决于 GPU 页表:
       a) 页表有映射 → 写到别人的显存 → 影响其他 tensor/kernel
       b) 页表无映射 → 写入被静默丢弃 (写黑洞)
    3. 越界读取 → 读到未初始化的显存数据 (垃圾值)

为什么特别危险?
  GPU 是异步的 → 写越界发生时, CPU 可能已经执行到很远的地方
  → cudaDeviceSynchronize() 也不会报错!
  → 只有 cuda-memcheck / compute-sanitizer 能检测

如何检测:
  compute-sanitizer --tool memcheck ./your_program

  compute-sanitizer 会:
    - 在每次全局内存访问前后插入检查
    - 检测越界读/写, 报告精确到哪行代码哪线程
    - 代价: kernel 运行慢 10-50×

  CUDA 11.2+ 推荐用 compute-sanitizer (替代旧的 cuda-memcheck)

最佳实践:
  1. 开发时: 用 compute-sanitizer 检查所有新 kernel
  2. 生产: 始终做好 idx < N 的边界检查
  3. 调试诡异结果时: 第一个怀疑就是越界访问
```


## 练习题

完成 `vector_add.cu` 的阅读后，可以在 [`exercises/`](./exercises/) 目录下做以下练习：

| 练习 | 公式 | 核心考点 |
|------|------|---------|
| [ex1_saxpy_level1.cu](./exercises/ex1_saxpy_level1.cu) | `y[i] = a*x[i] + y[i]` | 标量参数 + 原地修改（只填 kernel） |
| [ex1_saxpy_level2.cu](./exercises/ex1_saxpy_level2.cu) | 同上 | kernel + host 端全部自己写 |
| [ex2_relu_level1.cu](./exercises/ex2_relu_level1.cu) | `y[i] = max(x[i], 0)` | 单输入 + 条件判断（只填 kernel） |
| [ex2_relu_level2.cu](./exercises/ex2_relu_level2.cu) | 同上 | kernel + host 端全部自己写 |
| [ex3_fma_level1.cu](./exercises/ex3_fma_level1.cu) | `d[i] = a[i]*b[i] + c[i]` | 3 输入 1 输出（只填 kernel） |
| [ex3_fma_level2.cu](./exercises/ex3_fma_level2.cu) | 同上 | kernel + host 端全部自己写，管理 4 块 GPU 内存 |

编译方式（在 `exercises/` 目录下）：

```bash
nvcc -O2 -o ex1_saxpy_level1 ex1_saxpy_level1.cu
./ex1_saxpy_level1
```
