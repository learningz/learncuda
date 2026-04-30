# 第零章：术语与概念参考

**读完你能做什么**: 理解后续所有章节用到的术语和概念，不会被专业名词卡住。
**你需要先知道什么**: 完成了 [`tutorial.md`](../tutorial.md) Part 1 的动手练习，或有基本的 CUDA 编程经验。
**难度**: ⭐⭐ 进阶 (术语查阅) / ⭐⭐⭐ 专家 (深入理解)

> **给新手的建议**: 不需要从头到尾读完本章。推荐的用法是：
> 1. 先扫一遍 0.7 (GPU 专有术语) 和 0.12 (术语索引)，建立印象
> 2. 读后续章节遇到不懂的术语时回来查
> 3. 0.3-0.6 的计算机基础知识在你觉得"为什么是这样"的时候再深入
> 4. 半导体物理和数字逻辑见 `appendix_hardware.md` (可选)


## 0.3 存储器层级 (Memory Hierarchy)

### 为什么需要层级?

```
理想存储器: 无限大、无限快、零延迟。
现实: 又快又大的存储器不存在 (物理限制)。

根本矛盾:
  速度快 → 需要离处理器近 → 面积/容量有限 → 贵
  容量大 → 需要高密度存储 → 离处理器远 → 慢
  
  SRAM (静态RAM): 快 (~1ns), 但密度低 (~100 MB/cm²), 每 bit 6个晶体管
  DRAM (动态RAM): 慢 (~50ns), 但密度高 (~10 GB/cm²), 每 bit 1个晶体管+1个电容
  HBM:  DRAM 的变种, 通过堆叠提高带宽
  Flash: 极慢 (~μs), 但密度极高, 非易失
  
解决: 用层级结构, 越靠近处理器越快越小:

处理器内部:
  Register File:  SRAM, ~ns,    ~KB-MB
  L1 Cache:       SRAM, ~1-5ns,  ~KB-百KB
  
处理器外部:
  L2 Cache:       SRAM, ~5-20ns, ~MB-十MB
  Main Memory:    DRAM, ~50-100ns, ~GB-TB
  Storage:        Flash/HDD, ~μs-ms, ~TB-PB
```

### SRAM vs DRAM — 结构差异

```
SRAM (Static RAM):
  每个 bit 用 6 个晶体管 (6T SRAM cell):
  ┌─────────────────────┐
  │  VDD                │
  │   │      │          │
  │  PMOS   PMOS        │  4 个晶体管组成
  │   │      │          │  交叉耦合反相器
  │  ─┤├──┤├─            │  保持 0 或 1
  │   │      │          │
  │  NMOS   NMOS        │
  │   │      │          │
  │  GND                │
  │                     │
  │  + 2个 Access 晶体管  │  控制读/写
  └─────────────────────┘
  
  特点:
  - 不需要刷新, 只要供电就保持数据
  - 读写速度快 (亚纳秒级)
  - 面积大 (6T → 每 bit 很占面积)
  - 功耗: 静态漏电 + 读写功耗
  
  用于: Register File, L1/L2 Cache, Shared Memory

DRAM (Dynamic RAM):
  每个 bit 只用 1 个晶体管 + 1 个电容:
  ┌───────────┐
  │  Bit Line  │
  │     │      │
  │    NMOS    │
  │     │      │
  │  ┌──┴──┐   │
  │  │ Cap │   │  电容存储电荷 = 1 bit
  │  └──┬──┘   │  有电荷 = 1, 无电荷 = 0
  │     │      │
  │    GND     │
  └───────────┘
  
  特点:
  - 电容会漏电 → 必须定期刷新 (Refresh, 每 ~64ms)
  - 读是破坏性的 (读出电荷后需要写回)
  - 面积小 (1T1C → 每 bit 面积是 SRAM 的 1/4-1/6)
  - 读写慢 (~数十 ns), 但带宽可以做高 (宽总线)
  
  用于: GPU 显存 (HBM), CPU 主存 (DDR)
  
DRAM 的内部组织:
  Bank → Row → Column
  
  ┌─ DRAM Bank ──────────────────┐
  │                              │
  │  Row 0: [col0][col1]...[colN]│ ← 一行 ~1KB-2KB
  │  Row 1: [col0][col1]...[colN]│
  │  ...                         │
  │  Row M: [col0][col1]...[colN]│ ← 通常 ~32K-64K 行
  │                              │
  │  Row Buffer (Sense Amplifier)│ ← 当前打开的行的缓存
  │  [col0][col1]...[colN]       │   读写都在 Row Buffer 上操作
  └──────────────────────────────┘
  
  访问过程:
  1. ACTIVATE: 打开一行 (将 Row 的数据读入 Row Buffer) ~13ns (tRCD)
  2. READ/WRITE: 从 Row Buffer 读/写特定列 ~13ns (CL)
  3. PRECHARGE: 关闭当前行 (准备打开下一行) ~13ns (tRP)
  
  如果连续访问同一行 (Row Buffer Hit): 只需 READ ~13ns
  如果访问不同行 (Row Buffer Conflict): PRECHARGE + ACTIVATE + READ ~40ns
  → 这就是 "Row Conflict" 延迟的来源
```

### Cache 的工作原理

```
Cache 存储最近使用的数据, 利用局部性:
  时间局部性 (Temporal): 刚访问过的数据很可能再次被访问
  空间局部性 (Spatial): 刚访问地址附近的数据很可能被访问

Cache 的基本单位: Cache Line (缓存行)
  GPU L1 Cache Line: 128 bytes
  GPU L2 Cache Line: 128 bytes (分 4 个 32-byte Sector)
  CPU L1 Cache Line: 64 bytes (典型)

组织方式:
  Direct Mapped: 每个地址只能放在一个位置 → 简单但冲突多
  N-Way Set Associative: 每个地址有 N 个候选位置 → 冲突少
  Fully Associative: 可以放任何位置 → 最灵活但硬件复杂

  GPU L1: 通常 ~4-way set associative
  GPU L2: 通常 ~16-way set associative

替换策略:
  当 Cache 满了, 需要驱逐 (Evict) 一行:
  LRU (Least Recently Used): 驱逐最久没用的 → 硬件复杂
  Pseudo-LRU: LRU 的近似, 硬件简单
  Random: 随机驱逐 → 最简单
  
  GPU 通常用 Pseudo-LRU 或类似策略。

Write Policy:
  Write-Through: 写时同时更新 Cache 和下级存储 → 简单但带宽大
  Write-Back: 只写 Cache, 被驱逐时才写回下级 → 省带宽但复杂
  
  GPU L1: 通常 Write-Through (对全局内存) 或 Write-Evict
  GPU L2: Write-Back (最终写回 HBM)

Coherence (一致性):
  CPU 多核需要 Cache Coherence Protocol (如 MESI) 保证各核看到一致数据。
  GPU: 同一 SM 内通过 Shared Memory 显式管理;
       不同 SM 之间不保证 L1 一致性 (需要 __threadfence + L2 刷新)。
  → GPU 的弱一致性模型是有意为之: 简化硬件, 省面积。
```

### 虚拟内存与地址翻译

```
现代处理器 (包括 GPU) 使用虚拟地址:

为什么需要虚拟内存?
  1. 隔离: 不同进程/kernel 看到独立的地址空间
  2. 保护: 防止非法访问其他进程的内存
  3. 灵活: 物理内存不需要连续, 虚拟地址可以连续

地址翻译: 虚拟地址 (VA) → 物理地址 (PA)

Page Table (页表):
  将地址空间分成固定大小的 Page:
  ┌──────────────────────────────────────┐
  │ 虚拟地址 (48-bit, 以 4KB page 为例):  │
  │                                      │
  │ [VPN (36 bit)] [Page Offset (12 bit)]│
  │     │                      │         │
  │     │ 查 Page Table        │         │
  │     ▼                      │         │
  │ [PPN (36 bit)] [Page Offset (12 bit)]│
  │                                      │
  │ = 物理地址                            │
  └──────────────────────────────────────┘
  
  Page Table 本身存在内存中 (很大, 多级)。
  每次访存都要查 Page Table → 如果每次都查 → 极慢。

TLB (Translation Lookaside Buffer):
  Page Table 的硬件缓存。
  
  GPU TLB 层级:
  L1 TLB (per SM): ~32-128 entries, ~1 cycle 延迟
    命中 → 直接得到物理地址
    未命中 ↓
  L2 TLB (共享): ~几千 entries, ~几十 cycles
    命中 → 得到物理地址
    未命中 ↓
  Page Table Walk: 从内存中读 Page Table → ~几百到上千 cycles!
    → 这就是为什么 TLB Miss 极其昂贵
    → 使用 Large Pages (2MB instead of 4KB) 可以减少 TLB miss
       因为一个 TLB entry 覆盖更大范围

GPU 的页表:
  NVIDIA GPU 使用多级页表 (类似 CPU 的 x86-64 四级页表)
  Page Table 存在 GPU 显存中
  CUDA Unified Memory 的 Page Fault:
    GPU 访问不在显存中的页 → GPU MMU 触发 Page Fault
    → 通知 CPU Driver → 页面迁移 → 更新 Page Table → 重试
    整个过程 ~20-50 μs (非常慢!)
```


## 0.4 总线与互联 (Buses and Interconnect)

### PCIe — CPU 与 GPU 之间的桥梁

```
PCIe (Peripheral Component Interconnect Express):
  GPU 通过 PCIe 插槽连接到 CPU (除了 NVLink 直连的场景)。

物理层:
  PCIe 使用差分串行信号 (Differential Signaling):
  每个 Lane 有 2 对线 (TX差分对 + RX差分对)
  x16 = 16 个 Lane 并行
  
  PCIe 代际:
  ┌─────────────────────────────────────────────────┐
  │ 版本    │ 每Lane带宽  │ x16总带宽    │ 编码      │
  │ Gen 3  │ 8 GT/s     │ ~16 GB/s    │ 128b/130b │
  │ Gen 4  │ 16 GT/s    │ ~32 GB/s    │ 128b/130b │
  │ Gen 5  │ 32 GT/s    │ ~64 GB/s    │ 128b/130b │
  │ Gen 6  │ 64 GT/s    │ ~128 GB/s   │ 242b/256b │
  └─────────────────────────────────────────────────┘
  
  GT/s = GigaTransfers per second
  128b/130b 编码: 每 130 bit 中有 128 bit 是有效数据 (编码开销 ~1.5%)

BAR (Base Address Register):
  GPU 通过 PCIe BAR 向 CPU 暴露一段地址空间:
  BAR0: GPU 寄存器 (控制寄存器, Doorbell 等) → MMIO 访问
  BAR1: GPU 显存的一个窗口 → CPU 可以直接读写 GPU 显存
        (很慢, 走 PCIe → 通常只用于小量数据或调试)
  
  MMIO (Memory-Mapped IO):
  CPU 写 GPU 的某个地址 = 写 GPU 的控制寄存器
  例如: CPU 写 Doorbell Register → 通知 GPU 有新命令
  这就是 kernel launch 时 CPU 通知 GPU 的机制。

DMA (Direct Memory Access):
  数据传输不经过 CPU:
  cudaMemcpy(H2D): GPU 的 DMA Engine 从 CPU 内存读, 写入 GPU 显存
  cudaMemcpy(D2H): GPU 的 DMA Engine 从 GPU 显存读, 写入 CPU 内存
  
  GPU 通常有 2 个独立的 DMA Engine (Copy Engine):
  CE0: Host → Device (H2D 方向)
  CE1: Device → Host (D2H 方向)
  它们可以和 Compute Engine 同时工作 → 传输和计算重叠
```

### NVLink — GPU 间高速互联

```
NVLink 是 NVIDIA 专有的 GPU 间/GPU-CPU 间互联:

              NVLink
  ┌──────┐ ←════════→ ┌──────┐
  │ GPU 0│             │ GPU 1│
  └──────┘             └──────┘
  
  NVLink 3.0 (A100): 12 links × 50 GB/s = 600 GB/s (双向)
  NVLink 4.0 (H100): 18 links × 50 GB/s = 900 GB/s (双向)
  
  vs PCIe 4.0 x16: 32 GB/s → NVLink 快 ~20×!

NVSwitch:
  当 GPU 数量 > 2 时, 直接两两 NVLink 连接不够 (link 数有限)。
  NVSwitch 是一个独立的交换芯片:
  
  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
  │ GPU0 │   │ GPU1 │   │ GPU2 │   │ GPU3 │
  └──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘
     │          │          │          │
     └──────────┴──────────┴──────────┘
                    │
              ┌─────┴─────┐
              │  NVSwitch  │ ← 任意 GPU 对之间全带宽互联
              │ (独立芯片) │
              └───────────┘
  
  DGX A100: 6 个 NVSwitch 连接 8 个 A100
  → 任意两个 GPU 之间 600 GB/s 带宽 (全双工)

GPU Direct:
  GPU Direct RDMA: 网卡 (NIC) 直接读写 GPU 显存, 不经过 CPU
  GPU Direct Storage: NVMe SSD 直接读写 GPU 显存
  GPU Direct P2P: GPU 之间直接通过 NVLink/PCIe 传数据
  
  这些技术的核心: 绕过 CPU, 减少数据搬运延迟
```


## 0.5 处理器微架构基础

### 指令集架构 (ISA) vs 微架构 (Microarchitecture)

```
ISA (Instruction Set Architecture):
  定义了软件可见的接口: 有哪些指令、寄存器、寻址模式。
  同一 ISA 可以有不同的硬件实现。
  
  CPU: x86-64, ARM, RISC-V → 公开标准, 任何人可以实现
  GPU: NVIDIA PTX (虚拟ISA) + SASS (真实ISA) → NVIDIA 私有

  PTX (Parallel Thread Execution):
    NVIDIA 定义的虚拟指令集。
    和具体 GPU 架构无关。
    可以被 JIT 编译成任何架构的 SASS。
    保证向前兼容: 旧 PTX 在新 GPU 上能跑。
    
  SASS (Streaming ASSembler):
    具体 GPU 架构的真实机器码。
    sm_70 (Volta) 的 SASS 和 sm_80 (Ampere) 不同。
    NVIDIA 不公开 SASS 的完整规范, 但可以反汇编查看。

微架构 (Microarchitecture):
  ISA 的具体硬件实现方式。
  "Ampere" 是微架构名, SM_80 是 ISA 版本。
  同一 ISA 版本也可能有不同的微架构变体
  (如 GA100 / GA102 都是 Ampere, 但 SM 结构不同)。
```

### 指令执行模式

```
CPU: 乱序执行 (Out-of-Order Execution)
  指令可以不按程序顺序执行, 只要结果正确 (数据依赖得到满足)。
  硬件:
  - Reorder Buffer (ROB): 几百 entry, 跟踪指令状态
  - Reservation Station: 等待操作数就绪
  - Rename Register File: 消除名称依赖 (WAW, WAR)
  
  这些硬件极其复杂, 占大量面积和功耗。
  好处: 单线程性能极高 (自动发现并行性)。

GPU: 顺序发射 (In-Order Issue), 多线程隐藏延迟
  每个 Warp 内的指令按程序顺序发射 (不乱序!)。
  但多个 Warp 之间可以交错执行。
  
  简单得多的硬件:
  - 没有 ROB (不需要, 因为不乱序)
  - 没有 Rename (寄存器直接按编号分配)
  - 只需要 Scoreboard (追踪数据依赖)
  
  延迟隐藏不靠硬件复杂度, 靠线程数量。
  面积省下来全部做计算单元和寄存器。
```


## 0.6 并行计算基本概念

### 并行的层次

```
1. Bit-Level Parallelism (位级并行):
   32-bit 加法器同时处理 32 个 bit → 一条指令
   这在现代处理器中是最基础的, 无需程序员关心。

2. Instruction-Level Parallelism (ILP, 指令级并行):
   同时执行多条独立指令。
   CPU: 超标量 (Superscalar) → 每周期发射多条指令
   GPU: 同一 Warp 的连续独立指令可以背靠背发射

3. Data-Level Parallelism (DLP, 数据级并行):
   同一操作应用于多个数据元素。
   CPU: SIMD (SSE/AVX) → 一条指令处理 4-16 个 float
   GPU: SIMT → 一条指令处理 32 个线程 (Warp)

4. Thread-Level Parallelism (TLP, 线程级并行):
   多个独立线程同时执行。
   CPU: 多核, 每核独立线程 → ~8-64 线程
   GPU: 数千个线程同时在 SM 上 → ~100,000+ 线程

5. Task-Level Parallelism:
   不同任务 (kernel) 同时执行。
   GPU: 多 Stream 并发, MPS (Multi-Process Service)
```

### Amdahl's Law — 并行的理论极限

```
加速比 S = 1 / ((1-P) + P/N)

P = 可并行部分的比例
N = 并行处理器数量

例: 程序 95% 可并行 (P=0.95):
  N=10:    S = 1/(0.05 + 0.095) = 6.9×
  N=100:   S = 1/(0.05 + 0.0095) = 16.8×
  N=1000:  S = 1/(0.05 + 0.00095) = 19.6×
  N=∞:     S = 1/0.05 = 20× ← 理论极限!

  即使有无限多的处理器, 5% 的串行部分限制了最大加速比为 20×!

对 GPU 的启示:
  1. Kernel 中的串行部分 (如全局归约的最后一步) 限制了可扩展性
  2. 小 kernel 的 launch 开销是 "串行" 的 → 影响很大
  3. 即使 GPU 有上万个核心, 如果算法不够并行, 加速有限
```

### 常用并行原语

```
Map:    f([a,b,c]) → [f(a), f(b), f(c)]      完美并行
Reduce: sum([a,b,c,d]) → a+b+c+d              需要通信 (树形归约)
Scan:   prefix_sum([a,b,c,d]) → [a, a+b, a+b+c, a+b+c+d]  有顺序依赖
Gather: output[i] = input[index[i]]            随机读
Scatter: output[index[i]] = input[i]           随机写 (可能冲突)
Stencil: output[i] = f(input[i-1], input[i], input[i+1])  邻域访问

GPU 的优势:
  Map → 完美映射到 GPU (每线程一个元素)
  Reduce → 需要技巧 (Warp Shuffle + Shared Memory)
  GEMM → GPU 的甜点 (高算术强度 + 规则访问)

GPU 的弱势:
  分支密集的代码 (Warp Divergence)
  随机内存访问 (合并不了)
  串行依赖链 (单线程性能差)
  递归/深度嵌套 (栈空间有限)
```


## 0.7 GPU 专有硬件术语

```
SM (Streaming Multiprocessor):
  GPU 的基本计算单元。一个 GPU 有多个 SM (A100: 108个)。
  类比 CPU 的 "核心", 但比 CPU 核心简单得多, 数量多得多。
  一个 SM 可以同时运行数十个 Warp (数千个线程)。

Processing Block / Sub-partition:
  SM 内部的子单元。Ampere SM 有 4 个 Processing Block。
  每个 PB 有自己的:
  - Warp Scheduler (1个)
  - FP32 Core (16个)
  - FP32/INT Core (16个)
  - Tensor Core (1个)
  - LD/ST Unit (4个)
  - SFU (4个)
  - Register File Partition (16384 × 32-bit)

CUDA Core:
  NVIDIA 的市场术语, 指一个 FP32 ALU (算术逻辑单元)。
  一个 CUDA Core 每周期处理一个线程的一条 FP32 指令。
  A100: 64 CUDA Core / SM × 108 SM = 6912 CUDA Core
  注意: "CUDA Core" 不是独立的处理器, 它没有自己的 PC 或调度逻辑。
  它只是 SM 内的一个执行管道中的功能单元。

Tensor Core:
  专用矩阵乘累加单元。一条指令完成一个小矩阵乘 (如 16×8×16)。
  性能比 CUDA Core 做等量矩阵乘快 ~16×。

SFU (Special Function Unit):
  执行超越函数: sin, cos, exp, log, rsqrt 等。
  比 FP32 Core 慢 ~4× (吞吐量)。

LD/ST Unit (Load/Store Unit):
  执行内存访问: 地址生成、合并、cache 查询。

Warp:
  32 个线程组成的执行单位。GPU 不调度单个线程, 而是调度 Warp。
  同一 Warp 的 32 个线程在同一时刻执行同一条指令 (SIMT)。

Lane:
  Warp 内的线程编号 (0-31)。Lane ID = threadIdx.x % 32 (1D block 中)。

Occupancy:
  SM 上实际驻留的 Warp 数 / SM 最大可驻留 Warp 数。
  反映延迟隐藏能力。100% 不一定最优。

Scoreboard:
  硬件表, 追踪哪些寄存器有尚未完成的写入。
  Warp 的下一条指令的源寄存器有 pending write → Warp stall。

MSHR (Miss Status Holding Register):
  L1 Cache 中跟踪在飞内存请求的硬件表。
  合并对同一 cache line 的多个请求。容量有限 (~48-64 entries/SM)。

Barrier:
  硬件同步机制。__syncthreads() 编译为 BAR.SYNC 指令。
  每个 Block 最多 16 个 barrier, 每个有参与线程计数器。

CTA (Cooperative Thread Array):
  NVIDIA 对 "Thread Block" 的硬件术语。CTA = Block。

GPC (Graphics Processing Cluster):
  包含多个 TPC 的物理单元。共享时钟域和部分互联。

TPC (Texture Processing Cluster):
  包含 2 个 SM + 纹理单元的物理单元。

GigaThread Engine:
  全局 CTA 调度器。负责将 Block 分配到 SM。也叫 CTA Scheduler。
```


## 0.8 CUDA 软件栈

```
从上到下的软件层级:

┌─────────────────────────────────┐
│ Python (PyTorch / JAX / etc.)   │ ← 用户代码
├─────────────────────────────────┤
│ CUDA Libraries                  │
│ cuBLAS, cuDNN, cuFFT, NCCL, etc│ ← 高性能数学库
├─────────────────────────────────┤
│ CUDA Runtime API (libcudart)    │ ← cudaMalloc, cudaMemcpy, <<<>>>
├─────────────────────────────────┤
│ CUDA Driver API (libcuda)       │ ← cuCtxCreate, cuModuleLoad, cuLaunchKernel
├─────────────────────────────────┤
│ Kernel Mode Driver (nvidia.ko)  │ ← 内核驱动, 管理硬件资源
├─────────────────────────────────┤
│ GPU Hardware                    │
└─────────────────────────────────┘

CUDA Runtime API vs Driver API:
  Runtime: 更高层, 更易用。大多数人用的是这个。
    cudaMalloc, cudaMemcpy, kernel<<<>>>, cudaDeviceSynchronize
    自动管理 Context (每个 GPU 一个默认 Context)。
    
  Driver: 更底层, 更灵活。用于需要精确控制的场景。
    cuCtxCreate, cuModuleLoad, cuLaunchKernel
    需要手动管理 Context、Module、Function。
    PyTorch/TensorFlow 的后端用 Driver API。

CUDA Context:
  类似 CPU 的 "进程"。封装了:
  - GPU 上的地址空间 (Page Table)
  - 已加载的 Module (kernel 二进制)
  - 已分配的内存
  - 已创建的 Stream 和 Event
  一个 GPU 可以有多个 Context, 但同一时刻只有一个 active。
  Context Switch 有开销 (~μs级)。

CUDA Module:
  一个编译好的 GPU 二进制 (cubin 或 fatbinary)。
  包含一个或多个 kernel 函数。

CUDA Stream:
  GPU 上命令的有序队列。同一 Stream 内顺序执行, 不同 Stream 可并发。
  底层: 一个 Ring Buffer, CPU 端写入, GPU 端读取。

CUDA Event:
  Stream 中的时间标记。可以用于:
  - 精确计时 (cudaEventElapsedTime)
  - 跨 Stream 同步 (cudaStreamWaitEvent)
  - 查询操作是否完成 (cudaEventQuery)

nvcc 编译流程:
  .cu 源码
    │ nvcc 前端: 分离 host code 和 device code
    ├── Host code → 系统 C++ 编译器 (gcc/clang/MSVC)
    └── Device code → cicc (CUDA 前端) → PTX → ptxas → SASS (cubin)
  
  fatbinary: 将多个架构的 cubin 和 PTX 打包在一起。
  运行时: 选择匹配的 cubin, 或 JIT 编译 PTX。
```


## 0.9 浮点数与数值表示

### IEEE 754 浮点标准

```
浮点数 = (-1)^sign × 1.mantissa × 2^(exponent - bias)

FP64 (double):
  1 sign + 11 exponent + 52 mantissa = 64 bits
  Bias = 1023
  范围: ~±1.8 × 10^308
  精度: ~15-16 位有效十进制数字
  机器精度 (ε): 2^-52 ≈ 2.22 × 10^-16

FP32 (float):
  1 sign + 8 exponent + 23 mantissa = 32 bits
  Bias = 127
  范围: ~±3.4 × 10^38
  精度: ~7 位有效十进制数字
  机器精度 (ε): 2^-23 ≈ 1.19 × 10^-7

FP16 (half):
  1 sign + 5 exponent + 10 mantissa = 16 bits
  Bias = 15
  范围: ~±65504
  精度: ~3-4 位有效十进制数字
  机器精度 (ε): 2^-10 ≈ 9.77 × 10^-4

BF16 (bfloat16):
  1 sign + 8 exponent + 7 mantissa = 16 bits
  Bias = 127
  范围: 和 FP32 相同! (8 bit exponent)
  精度: ~2-3 位有效十进制数字 (比 FP16 差)
  优势: FP32 截断高16位即可得到 BF16 → 转换免费

特殊值:
  +0 / -0:    exponent=0, mantissa=0 (IEEE 有正零和负零)
  +∞ / -∞:   exponent=全1, mantissa=0
  NaN:        exponent=全1, mantissa≠0 (Not a Number)
  Subnormal:  exponent=0, mantissa≠0 → 可表示非常小的数 (精度降低)

FMA (Fused Multiply-Add):
  a × b + c, 只做一次舍入 (而不是乘和加各舍入一次)。
  → 比分开的 MUL+ADD 更精确 (只有 1 ULP 误差而不是 2 ULP)
  → 也更快 (1 条指令而不是 2 条)
  GPU 的 FFMA 指令就是 FMA。
```

### 整数表示

```
无符号整数 (Unsigned):
  N bit → 范围 [0, 2^N - 1]
  uint32: [0, 4294967295]

有符号整数 (Two's Complement, 补码):
  N bit → 范围 [-2^(N-1), 2^(N-1) - 1]
  int32: [-2147483648, 2147483647]
  最高位是符号位: 0=正, 1=负
  -x 的表示 = ~x + 1 (按位取反再加1)

定点数 (Fixed Point):
  将整数的一部分位解释为小数。
  例: Q8.8 格式: 8 bit 整数 + 8 bit 小数 = 16 bit
  值 = 原始整数 / 256
  用于某些低精度推理场景。

量化整数 (Quantized Integer):
  INT8 / INT4 量化:
  float_value = int_value × scale + zero_point
  scale 和 zero_point 是浮点数, per-tensor 或 per-channel 存储。
  GPU Tensor Core 直接支持 INT8/INT4 矩阵乘。
```


## 0.10 性能分析术语

```
Latency (延迟):
  完成一个操作所需的时间。
  单位: cycles, ns, μs, ms
  例: 全局内存加载延迟 = ~400-800 cycles ≈ 300-600 ns

Throughput (吞吐):
  单位时间内完成的操作数。
  单位: ops/sec, bytes/sec, FLOPS
  例: A100 FP32 吞吐 = 19.5 TFLOPS

Bandwidth (带宽):
  单位时间内传输的数据量。
  单位: bytes/sec, GB/s, TB/s
  例: A100 HBM 带宽 = 2039 GB/s
  
  理论带宽 vs 有效带宽:
    理论: 物理极限 (时钟 × 数据宽度 × DDR)
    有效: 实际程序测到的 (总是 < 理论, 因为各种开销)
    效率 = 有效/理论 → 80%+ 算优秀

FLOPS (Floating Point Operations Per Second):
  每秒浮点运算次数。
  TFLOPS = 10^12 FLOPS
  
  FMA 算 1 FLOP 还是 2 FLOP?
  工业惯例: FMA = 2 FLOP (一个乘法 + 一个加法)
  所以 A100 FP32: 19.5 TFLOPS 指的是 9.75T 条 FMA/s × 2

Arithmetic Intensity (算术强度, AI):
  FLOP / Byte (每传输一字节数据做多少浮点运算)
  决定 kernel 是 Memory Bound 还是 Compute Bound。
  
  Ridge Point = 峰值算力 / 峰值带宽
  A100 FP32: 19.5 TFLOPS / 2039 GB/s ≈ 9.6 FLOP/Byte
  AI < 9.6 → Memory Bound
  AI > 9.6 → Compute Bound

Memory Bound:
  性能被内存带宽限制, 计算单元有闲置。
  大多数 AI 算子 (elementwise, reduce, softmax, layernorm) 属于此类。
  优化方向: 减少访存量 (融合, tiling), 提高合并度, 向量化。

Compute Bound:
  性能被计算单元限制, 内存带宽有富余。
  大矩阵乘法, 大卷积属于此类。
  优化方向: Tensor Core, ILP, 减少无效计算。

Latency Bound:
  性能被单个长延迟操作限制 (如串行依赖链)。
  少见, 通常是算法设计问题。

Occupancy:
  实际驻留 Warp / SM 最大 Warp 容量。
  影响延迟隐藏能力, 但不是越高越好。
  
Warp Stall Reasons (ncu 中的关键指标):
  Long Scoreboard:  等全局内存 → Memory Bound 的信号
  Short Scoreboard: 等计算完成 → 依赖链/ILP 不足
  Math Throttle:    计算管道满 → Compute Bound
  MIO Throttle:     内存指令队列满 → MSHR 耗尽
  Barrier:          等 __syncthreads → 同步开销
  Not Selected:     就绪但没被选 → 好事, 说明并行度充足
```


## 0.11 深度学习相关术语

```
Tensor (张量):
  多维数组。深度学习中所有数据都是 Tensor。
  shape = [N, C, H, W]: Batch × Channel × Height × Width (图像)
  shape = [B, S, D]: Batch × Sequence × Dimension (NLP)
  GPU 上连续存储 (contiguous), 按行主序 (row-major)。

算子 (Operator / Op):
  计算图中的一个节点。接收输入 Tensor, 输出 Tensor。
  例: MatMul, Conv2d, ReLU, Softmax, LayerNorm

前向传播 (Forward Pass):
  输入 → 逐层计算 → 输出 + Loss
  每层保存中间结果 (激活值) 给反向用

反向传播 (Backward Pass):
  从 Loss 开始, 逐层计算梯度 (链式法则)
  需要前向保存的激活值来计算梯度
  → 这就是为什么训练比推理用更多显存

梯度 (Gradient):
  Loss 对参数的偏导数。用于更新参数 (如 SGD: w -= lr × grad)。

Loss Scaling:
  FP16 训练时, 梯度可能太小变成 0。
  先放大 Loss → 梯度也放大 → 更新前缩小回来。
  动态 Loss Scaling: 自动调整缩放因子。

混合精度 (Mixed Precision):
  FP16 做计算 (快) + FP32 做累加和权重更新 (准)。
  Tensor Core 天然支持: FP16 输入, FP32 累加器。

算子融合 (Operator Fusion):
  将多个小算子合并成一个大 kernel → 减少中间数据的内存读写。
  例: MatMul + Bias + ReLU → 1 个 fused kernel
  PyTorch torch.compile / Triton 自动做融合。

激活检查点 (Activation Checkpointing):
  不保存所有前向激活值, 反向时重新计算。
  显存减少 ~O(√N) 层, 计算增加 ~33%。

模型并行 (Model Parallelism):
  Tensor Parallel: 将一个大矩阵切分到多个 GPU
  Pipeline Parallel: 不同层放在不同 GPU, 流水线执行
  
数据并行 (Data Parallelism):
  每个 GPU 有完整模型, 各自处理不同 batch, 然后同步梯度 (AllReduce)。

AllReduce:
  分布式训练中的核心通信操作。
  每个 GPU 有本地梯度, AllReduce 后每个 GPU 得到全局梯度之和。
  通过 NCCL (NVIDIA Collective Communication Library) 实现。
  使用 NVLink / NVSwitch / IB 等高速互联。
```


## 0.12 术语速查索引 (按字母排序)

```
术语                    简要定义                                    详见
─────                  ────────                                   ────
ALU                    算术逻辑单元, 做加减乘除等                     0.2
Amdahl's Law           并行加速的理论上限公式                        0.6
Bank (DRAM)            DRAM 内部的独立存储单元                      0.3
Bank (Shared Mem)      Shared Memory 的 32 个并行端口               Ch3
BAR                    PCIe Base Address Register                 0.4
Barrier                硬件同步屏障 (__syncthreads)                 0.7
BF16                   Brain Float 16, 8bit exp + 7bit mantissa    0.9
Cache Line             缓存最小传输单位 (GPU: 128 bytes)             0.3
Coalescing             将多线程内存请求合并为少量事务                  Ch1
Compute Bound          性能受计算能力限制                           0.10
Constant Memory        64KB 只读内存, 有专用缓存                    Ch3
Context (CUDA)         GPU 上的执行环境 (地址空间/模块/分配)          0.8
CTA                    Cooperative Thread Array = Block             0.7
CUDA Core              NVIDIA 对 FP32 ALU 的市场术语                0.7
DMA                    直接内存访问, 不经过处理器                     0.4
DRAM                   动态随机存储器 (显存/主存的基础)               0.3
ECC                    Error Correcting Code, 内存纠错              Ch1
FinFET                 鳍式场效应管, 22nm 以下标准晶体管结构           0.1
FMA                    Fused Multiply-Add, 一条指令做 a×b+c          0.9
FLOPS                  每秒浮点运算次数                             0.10
FP16/FP32/FP64         IEEE 半精度/单精度/双精度浮点                  0.9
GPC                    Graphics Processing Cluster                 0.7
Grid                   一次 kernel launch 的所有 Block 的集合        Ch2
HBM                    High Bandwidth Memory, 堆叠式高带宽显存       Ch1
I-Cache                指令缓存                                    Ch1
ILP                    Instruction-Level Parallelism               0.6
ISA                    Instruction Set Architecture                0.5
L1/L2 Cache            一级/二级缓存                                0.3
Lane                   Warp 内的线程编号 (0-31)                     0.7
LD/ST Unit             Load/Store 执行单元                          0.7
Memory Bound           性能受内存带宽限制                           0.10
MMIO                   Memory-Mapped IO                            0.4
MOSFET                 金属氧化物半导体场效应管                       0.1
MSHR                   Miss Status Holding Register                0.7
NaN                    Not a Number (浮点特殊值)                    0.9
NoC                    Network-on-Chip, 片上互联网络                 Ch1
NVLink                 GPU 间高速互联 (600-900 GB/s)                0.4
NVSwitch               NVLink 交换芯片                             0.4
Occupancy              SM 的 Warp 占用率                           0.7
Page Table             虚拟→物理地址映射表                          0.3
PCIe                   CPU-GPU 互联总线                            0.4
Pipeline               流水线, 多阶段重叠执行                       0.2
PTX                    Parallel Thread Execution, GPU 虚拟指令集     0.5
Register File          寄存器文件, SM 上最快的存储                    Ch1
Roofline Model         性能分析模型 (算力 vs 带宽)                   Ch3
Row Buffer             DRAM Bank 内打开行的缓存                     0.3
SASS                   GPU 真实机器指令集                           0.5
Scoreboard             硬件依赖追踪表                               0.7
Sector                 L2 Cache Line 的 32-byte 子单位              Ch3
SFU                    Special Function Unit (sin/cos/exp)          0.7
Shared Memory          Block 内线程共享的片上 SRAM                   Ch3
SIMD                   Single Instruction Multiple Data (CPU)       0.6
SIMT                   Single Instruction Multiple Threads (GPU)    Ch2
SM                     Streaming Multiprocessor                    0.7
SRAM                   静态随机存储器 (缓存/寄存器的基础)             0.3
Stream (CUDA)          GPU 命令有序队列                             0.8
Subnormal              非常小的浮点数 (exp=0)                       0.9
Tensor Core            矩阵乘累加专用单元                           0.7
TDP                    Thermal Design Power, 最大散热功率            0.1
TLB                    Translation Lookaside Buffer, 页表缓存       0.3
TLP                    Thread-Level Parallelism                    0.6
TMA                    Tensor Memory Accelerator (Hopper)           Ch6
TPC                    Texture Processing Cluster                  0.7
TSV                    Through-Silicon Via, 硅通孔 (HBM 互联)       Ch1
Warp                   32 线程的执行单位                            0.7
Warp Divergence        同一 Warp 内线程走不同分支                    Ch4
WMMA                   Warp Matrix Multiply-Accumulate API          Ch6
```


## 0.13 本章总结

```
本章建立了从晶体管到 GPU 软件栈的完整知识链:

物理层:   晶体管 → 逻辑门 → 功能单元 (ALU/FMA/SFU) → 存储单元 (SRAM/DRAM)
组织层:   Register → Cache → Memory → Storage (层级结构)
连接层:   PCIe / NVLink / NoC (片内/片间/系统级互联)
架构层:   ISA (PTX/SASS) → 微架构 (SM/Warp Scheduler/Scoreboard)
编程层:   CUDA Runtime/Driver → Stream/Event/Graph
数值层:   FP64/FP32/FP16/BF16/INT8 的表示、范围、精度

核心思想:
  1. 存储器层级的矛盾: 快↔大 不可兼得 → 层级结构 + 局部性利用
  2. CPU vs GPU 的本质: 晶体管预算分配不同 → 低延迟 vs 高吞吐
  3. GPU 延迟隐藏: 不减少延迟, 而是用海量线程让延迟"消失"
  4. 并行的天花板: Amdahl 定律 → 串行部分决定上限
```


## 0.14 Q&A — 常见疑问与概念辨析

### Q1: "7nm" 是指晶体管真的只有 7 纳米吗?

```
不是。现代工艺节点名称是市场命名, 和实际物理尺寸脱节。

TSMC 7nm 的实际 Gate Pitch (栅极间距) ≈ 54nm, Fin Pitch ≈ 30nm。
真正衡量工艺先进程度的是逻辑密度 (MTr/mm²):
  TSMC 7nm: 65.5 MTr/mm²
  TSMC 5nm: 113.9 MTr/mm²
  密度提升 ~1.74× (接近理论 2×, 因为面积缩小)

为什么不直接叫密度? 因为 "7nm" 比 "65.5 MTr/mm²" 更好卖。
```

### Q2: SRAM 为什么比 DRAM 快那么多? 既然快, 为什么不全用 SRAM?

```
SRAM 快的本质原因:
  读操作只需要感应交叉耦合反相器的状态 → 信号强, 几乎瞬时
  不需要给电容充放电 → 无 RC 延迟

DRAM 慢的原因:
  读操作需要将电容上微小的电荷放大 (Sense Amplifier)
  电容电荷很少 (~30 fC), 信号/噪声比低 → 放大需要时间
  读后必须回写 (破坏性读) → 额外延迟

不全用 SRAM 的原因: 面积和成本
  SRAM 6T cell: ~120 F² (F = 最小特征尺寸)
  DRAM 1T1C cell: ~20 F²
  → SRAM 面积是 DRAM 的 6× → 同等容量成本 6×+
  
  假设 A100 的 80GB 显存全用 SRAM:
  以 7nm 的 SRAM 密度 (~25 Mb/mm²):
  80GB = 640 Gb → 640000/25 = 25600 mm² ≈ 31 个 A100 die 的面积!
  → 物理上不可能
```

### Q3: "带宽" 和 "吞吐" 有什么区别?

```
带宽 (Bandwidth): 通常指数据传输速率 (bytes/sec)
  "HBM 带宽 2039 GB/s" = 每秒最多传输 2039 GB 数据

吞吐 (Throughput): 更通用, 指单位时间完成的操作量
  可以是 bytes/sec (此时 = 带宽)
  也可以是 FLOPS (浮点运算/秒)
  也可以是 ops/sec (指令/秒)

在 GPU 领域:
  "内存带宽" = bytes/sec (HBM 的传输能力)
  "计算吞吐" = FLOPS (ALU 的计算能力)
  "指令吞吐" = instructions/cycle/SM (调度器的发射能力)

易混淆点:
  "这个 kernel 的带宽利用率是 85%"
  → 指的是有效数据传输速率 / HBM 峰值带宽 = 85%
  不是指 "85% 的数据是有用的" (那叫 Global Load Efficiency)
```

### Q4: 延迟 (Latency) 和延迟隐藏 (Latency Hiding) 不是矛盾的吗?

```
不矛盾。延迟隐藏不减少延迟, 它减少的是延迟对吞吐的影响。

类比: 餐厅厨师
  延迟: 一道菜从开始做到端上桌 = 30 分钟 (不变)
  无隐藏: 厨师做完一道才开始下一道 → 30 min/道
  有隐藏: 厨师同时处理 10 道菜, 交错操作 → 每 3 min 出一道

  每道菜还是要 30 分钟 (延迟不变),
  但餐厅整体每 3 分钟出一道 (吞吐提高 10×)。

GPU 的情况:
  一次全局内存加载延迟 = 500 cycles (不变)
  1 个 Warp: 发起加载, 等 500 cycles, 处理, 再加载 → 吞吐低
  20 个 Warp: Warp 0 加载等待 → 切 Warp 1 → 切 Warp 2 → ...
              500 cycles 后 Warp 0 数据到 → 立即处理 → 吞吐高

  延迟还是 500 cycles, 但每个 cycle 都有 Warp 在执行 → 吞吐接近峰值。
  "隐藏" 的意思是: 延迟的代价被其他工作掩盖了, 计算单元不闲着。
```

### Q5: FP16 和 BF16 到底该用哪个?

```
取决于场景:

FP16:
  精度高 (~3-4 位有效数字) ← 10 bit mantissa
  范围小 (±65504) ← 5 bit exponent
  需要 Loss Scaling (梯度可能下溢为 0)
  适用: 推理 (精度重要), CV 任务 (值范围可控)

BF16:
  精度低 (~2-3 位有效数字) ← 7 bit mantissa
  范围大 (和 FP32 相同) ← 8 bit exponent
  通常不需要 Loss Scaling (范围够大, 不溢出)
  FP32→BF16 转换零开销 (截断低 16 位)
  适用: 训练 (范围重要), NLP/LLM (某些层输出范围大)

大模型训练 (GPT/LLaMA 级):
  BF16 已成为事实标准 (方便, 不用折腾 Loss Scaling)
  Ampere/Hopper 对 BF16 的 Tensor Core 支持完善

推理 (部署):
  FP16 或 INT8/INT4 更常见 (极致压缩模型体积和带宽)
  BF16 推理也可以但没有 FP16 普及

易混淆:
  "FP16 更精确所以更好" → 不一定! BF16 的范围优势在训练中更关键。
  梯度值跨越 10^-8 到 10^3: FP16 的 ±65504 范围可能不够, BF16 没问题。
```

### Q6: GPU 的 "核心数" 和 CPU 能直接比较吗?

```
绝对不能。

CPU 核心: 一个完整的处理器
  有自己的指令获取/解码/分支预测/乱序引擎/多级缓存
  可以独立运行一个完整的操作系统线程
  例: Intel i9-13900K 有 24 核 → 能跑 24 个独立程序

GPU "CUDA Core": 一个浮点计算管道
  没有自己的 PC, 不能独立获取指令
  共享 Warp Scheduler, 不能独立调度
  只能执行 SM 分配给它的一条指令的一个线程部分
  例: A100 有 6912 CUDA Core → 不能跑 6912 个程序

正确的类比:
  GPU SM ≈ CPU 核心 (各自是独立的调度单位)
  A100: 108 SM → 类比 108 个 "核心" (但每个核心比 CPU 弱得多)
  
更准确的说法:
  GPU 是一个 108 核的处理器, 每核有 64 个 ALU 和 64 个并发线程槽。
  CPU 是一个 24 核的处理器, 每核有 ~8 个 ALU 和 2 个并发线程槽。
  
  GPU 总 ALU: 108 × 64 = 6912 → 大规模并行简单运算
  CPU 总 ALU: 24 × 8 = 192 → 少量但极其灵活的运算
```
