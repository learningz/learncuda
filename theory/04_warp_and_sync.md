# 第四章：Warp 深入 — 分支分歧、原语、同步与内存序

**难度**: ⭐⭐ 进阶 (4.1-4.4) / ⭐⭐⭐ 专家 (4.5-4.8)
**前置知识**: 第1章的 Warp 概念; 第2章的 SIMT 模型
**读完你能做什么**: 避免 Warp Divergence; 使用 Shuffle/Vote 写高效归约; 正确做 Block/Grid 同步
**配套代码**: `03_reduce/` (Warp Shuffle reduce)
**新手建议**: 4.1 (分支分歧) 和 4.2 (Warp Shuffle) 是必读。4.6 (内存一致性) 在你做跨 Block 通信时再读

> **回顾**: 在 [`tutorial.md`](../tutorial.md) Part 3 中你已经学过 Warp = 32 个线程为一组执行。
> 本章深入讲解 Warp 的行为细节。如果你对 Warp 还没有概念，请先回顾 [`tutorial.md`](../tutorial.md) Part 3。
>
> **本章首次出现的术语**:
> - **Warp Divergence (分支分歧)**: 同一 Warp 中的线程走了不同的 if/else 分支 → 性能下降
> - **Active Mask**: 一个 32-bit 位掩码，标记 Warp 中哪些线程当前"活跃"（参与执行）
> - **Predicate Register**: GPU 的条件标志寄存器，每个线程有自己的 (类似 CPU 的 flag)
> - **SASS**: GPU 的真实机器指令 (汇编语言)，类似 CPU 的 x86 汇编
> - **Warp Shuffle**: 同一 Warp 内的线程直接交换寄存器中的值，不需要经过 Shared Memory
> - **Barrier**: 同步屏障，所有线程到达后才能继续（`__syncthreads()` 就是一个 Barrier）
> - **原子操作 (Atomic)**: 保证多线程同时修改同一地址时不出错的特殊读-改-写操作

## 4.1 Warp Divergence — 从 SASS 指令级别理解

> **动手实验**: 运行 `11_warp_divergence/warp_divergence.cu` 亲眼看到分歧的性能影响!
> ```bash
> cd 11_warp_divergence && nvcc -O2 -o warp_divergence warp_divergence.cu && ./warp_divergence
> ```
> 对比 "无分歧"、"50% 分歧"、"无分支算术" 三种情况的耗时。

### 硬件实现机制

一个 Warp 的 32 个线程共用一套执行单元。遇到条件分支时，
硬件使用 **Active Mask** 和 **Predicate Register** 来控制哪些线程参与执行。

```
分支指令在 SASS 级别的真实行为:

源码:
    if (threadIdx.x < 16) {
        a[idx] = foo(x);
    } else {
        a[idx] = bar(x);
    }

编译后的 SASS (简化):
    ISETP.LT P0, PT, R_tidx, 16, PT;   // P0 = (threadIdx.x < 16)
    @P0  BRA  LABEL_THEN;               // predicate 为真 → 跳到 THEN
    // --- ELSE 路径 ---
    ...bar(x)...                         // Active Mask = ~P0 (线程 16-31)
    BRA LABEL_END;
    LABEL_THEN:
    // --- THEN 路径 ---
    ...foo(x)...                         // Active Mask = P0 (线程 0-15)
    LABEL_END:
    // 重新汇合: Active Mask = 0xFFFFFFFF

关键: 两条路径都在同一组执行单元上串行执行。
被 mask 掉的线程的 ALU 计算正常进行但结果被丢弃,
被 mask 掉的线程的 store 指令不会写入内存。
```

### 汇合点 (Reconvergence Point)

```
Pre-Volta (Pascal 及更早):
  使用硬件栈管理汇合:
  ┌────────────────────────────────┐
  │ Convergence Stack (每 Warp)     │
  │ ┌─────────────────────────────┐│
  │ │ Top: Reconvergence PC, Mask ││ ← 分支时压栈
  │ │ ...                         ││
  │ └─────────────────────────────┘│
  └────────────────────────────────┘
  
  遇到分支:
    1. 计算汇合点 PC (通常是 if-else 后的第一条指令)
    2. 将 {汇合点 PC, 全量 Mask} 压入栈
    3. 先执行 THEN 路径 (Active Mask = 满足条件的线程)
    4. THEN 路径执行完 → 弹栈 → 切换到 ELSE 路径
    5. ELSE 路径执行完 → 弹栈 → 到达汇合点 → 全量 Mask 恢复
  
  问题: 嵌套分支 → 栈深度增加; 循环中的分支 → 锁步限制

Volta+ (独立线程调度):
  没有硬件栈。每个线程有独立的 PC。
  Warp Scheduler 使用 "Convergence Optimizer" 硬件:
    - 动态检测哪些线程在同一 PC 位置
    - 将它们组合成一个 "子 Warp" 执行
    - 在分支后尝试让线程尽快重新汇合
  
  这允许了 Pre-Volta 不可能的模式:
    - Warp 内的生产者-消费者 (线程 0 产出数据, 线程 1 消费)
    - Warp 内的互斥锁 (需要 __syncwarp 配合)
    - 不同线程执行不同的循环迭代次数
```

### 分支分歧的精确性能模型

```
分歧的代价不是简单的 "两条路径串行":

Case 1: 短分支 (< ~7 条指令)
  编译器自动用谓词化 (Predication):
  @P0  FFMA R4, R0, R2, R4;    // 有谓词 P0 的 FMA
  @!P0 FFMA R4, R0, R3, R4;    // 有谓词 !P0 的 FMA
  
  两条指令都执行, 但只有一条写入结果。
  代价: 2 条指令而不是 1 条, 但没有跳转开销。
  → 分歧代价 ≈ 2× 指令数 (不是真正的分支跳转)

Case 2: 长分支 (真正的跳转指令)
  代价 = time(path_A) + time(path_B) + 跳转开销
  
  跳转开销:
  - 直接跳转 (BRA): ~5 cycles (Instruction Cache 可能命中)
  - 如果跳转目标不在 I-Cache → I-Cache miss → ~几十 cycles
  
Case 3: 不均匀的分支
  31 个线程走 A (3 cycles), 1 个线程走 B (100 cycles):
  总代价 = 3 + 100 = 103 cycles
  → 即使只有 1 个线程走了慢路径, 整个 Warp 都要等!
  
  这在以下场景中很常见:
  - 边界处理 (boundary checking)
  - 稀疏数据 (某些线程遇到 NaN/特殊值需要额外处理)
  - 早退 (early exit) 不生效 — 一个线程没退出 → 整个 Warp 继续

Case 4: 循环中的分歧
  for (int i = 0; i < array[threadIdx.x]; i++) {
      // 每个线程的循环次数不同!
      compute();
  }
  
  如果线程 0 循环 100 次, 线程 31 循环 1 次:
  Pre-Volta: 所有线程锁步, 执行 100 次. 线程 31 在后 99 次中被 mask.
  Volta+: 线程 31 可以先完成, 但它仍然不能执行下一条 Warp 指令
          (需要等到所有线程汇合). 实际改善有限.
  → 最坏情况: Warp 执行时间 = 最慢线程的时间
```

### 分歧优化: 完整技术清单

```cuda
// 技巧 1: 让分支对齐 Warp 边界
// 优化前: Warp 内分歧
if (threadIdx.x < threshold) { path_A(); } else { path_B(); }

// 优化后: 不同 Warp 走不同分支, Warp 内无分歧
int warp_id = threadIdx.x / 32;
if (warp_id < threshold / 32) { path_A(); } else { path_B(); }


// 技巧 2: 用算术代替分支 (Branchless)
// 适用于结果是两个值之一的情况

// 分支版:
float y = (x > 0) ? x : 0;
// SASS: ISETP + BRA + ... (跳转指令)

// 无分支版:
float y = fmaxf(x, 0.0f);     // 编译为 FMNMX 指令, 1 cycle, 无分支
// 或
float y = x * (float)(x > 0);  // ISETP + I2F + FMUL, 3 cycles 但无跳转

// 更复杂的无分支选择:
float y = (cond) ? val_a : val_b;
// 无分支: SEL 指令 (SASS 层面)
// 或手动: float y = val_b + (float)(cond) * (val_a - val_b);


// 技巧 3: 数据重排 (Data Reorganization)
// 如果分支取决于数据, 预排序数据让相邻线程走同一分支

// 优化前: 随机分布的正负数
//   [+, -, +, -, +, -, ...]  → 50% 分歧
// 优化后: 先对数据按正负分组排序
//   [+, +, +, ..., -, -, -]  → 只有 1 个 Warp 有分歧 (正负交界处)

// 适用场景: 稀疏矩阵处理、粒子模拟中的类型判断


// 技巧 4: Warp 内紧凑 (Warp Compaction)
// 将需要执行特殊路径的线程"打包"到少数几个 Warp 中
unsigned active = __ballot_sync(0xffffffff, needs_special_path);
int count = __popc(active);
int prefix = __popc(active & ((1u << lane_id) - 1));

// 将需要特殊处理的元素紧凑写入 buffer
if (needs_special_path) {
    special_buffer[base + prefix] = my_data;
}
// 稍后用专门的 kernel 处理 special_buffer
// 避免在主 kernel 中产生分歧


// 技巧 5: 利用 __any_sync / __all_sync 提前退出
// 如果 Warp 中所有线程都不需要特殊处理 → 跳过整个分支
if (__any_sync(0xffffffff, needs_special)) {
    // 只有当至少一个线程需要时才执行
    if (needs_special) {
        do_special_work();
    }
}
// 大多数 Warp 完全跳过这个 if 块 → 无分歧
```


## 4.2 Warp Shuffle — 完整机制与高级模式

> 分支分歧告诉你 Warp 的"限制"——同一 Warp 不能同时做不同的事。
> Warp Shuffle 则展示 Warp 的"超能力"——32 个线程可以直接交换寄存器中的值，
> 不需要 Shared Memory，不需要 __syncthreads()。
> 这是写高性能 Reduce、Softmax、LayerNorm 的核心工具。

### 硬件实现

```
Warp Shuffle 的物理路径:
  源线程的寄存器 → Warp 内部的 Crossbar 网络 → 目标线程的寄存器

不经过 Shared Memory!
延迟: 在 Scoreboard 中标记为 ~1-2 cycles (极快)

底层 SASS 指令:
  SHFL.BFLY R4, R0, 0x1, 0x1f;    // butterfly (XOR) shuffle
  SHFL.DOWN R4, R0, 0x10, 0x1f;   // shift down by 16
  SHFL.UP   R4, R0, 0x1, 0x0;     // shift up by 1
  SHFL.IDX  R4, R0, R2, 0x1f;     // indexed shuffle (从 R2 指定的 lane 读)

参数 0x1f = 31: "clamp" 值, 当 source lane ≥ 32 或 < 0 时的行为。
  对于 __shfl_down_sync: 如果 src_lane > 31, 返回调用者自己的值
  对于 __shfl_up_sync: 如果 src_lane < 0, 返回调用者自己的值
```

### Shuffle 的 width 参数 — 子 Warp 操作

```cuda
// __shfl_down_sync 有一个隐藏的 width 参数 (默认 32):
T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = warpSize);

// width 将 Warp 切成多个独立的段, 每段独立 shuffle:
// width=16: lane 0-15 为一段, lane 16-31 为一段
// width=8: lane 0-7, 8-15, 16-23, 24-31 各为一段

// 用途: 在一个 Warp 内同时做多个独立的归约!
// 例: 一个 Warp 同时处理 4 行, 每行 8 个元素
__device__ float warp_reduce_sum_width8(float val) {
    val += __shfl_down_sync(0xffffffff, val, 4, 8);
    val += __shfl_down_sync(0xffffffff, val, 2, 8);
    val += __shfl_down_sync(0xffffffff, val, 1, 8);
    return val;  // lane 0, 8, 16, 24 各自持有自己段的 sum
}

// 这在 Softmax (每行独立) 中非常有用:
// 如果行长度 ≤ 8, 一个 Warp 可以同时处理 4 行!
```

### 高级 Shuffle 模式

```cuda
// 模式 1: Warp 级前缀和 (Prefix Sum / Scan)
__device__ float warp_prefix_sum(float val) {
    // Hillis-Steele 并行前缀和
    float tmp;
    tmp = __shfl_up_sync(0xffffffff, val, 1);
    if (lane_id >= 1) val += tmp;
    tmp = __shfl_up_sync(0xffffffff, val, 2);
    if (lane_id >= 2) val += tmp;
    tmp = __shfl_up_sync(0xffffffff, val, 4);
    if (lane_id >= 4) val += tmp;
    tmp = __shfl_up_sync(0xffffffff, val, 8);
    if (lane_id >= 8) val += tmp;
    tmp = __shfl_up_sync(0xffffffff, val, 16);
    if (lane_id >= 16) val += tmp;
    return val;  // inclusive prefix sum
    // lane 0: a0
    // lane 1: a0+a1
    // lane 2: a0+a1+a2
    // ...
    // lane 31: a0+a1+...+a31
}

// 注意: 上面有分支! 但因为条件基于 lane_id (常量),
// 编译器会用谓词化处理, 实际不会产生分歧。


// 模式 2: Warp 级排序 (Bitonic Sort)
__device__ void warp_bitonic_sort(int &val) {
    // 5 阶段的 Bitonic 排序 (32 个元素)
    for (int k = 2; k <= 32; k *= 2) {
        for (int j = k/2; j >= 1; j /= 2) {
            int partner = lane_id ^ j;  // XOR 找到交换伙伴
            int partner_val = __shfl_xor_sync(0xffffffff, val, j);
            
            // 确定升序还是降序
            bool ascending = ((lane_id & k) == 0);
            if (ascending) {
                if (lane_id < partner) val = min(val, partner_val);
                else val = max(val, partner_val);
            } else {
                if (lane_id < partner) val = max(val, partner_val);
                else val = min(val, partner_val);
            }
        }
    }
}
// 32 个元素的排序在一个 Warp 内完成, 不用 Shared Memory!


// 模式 3: Warp 级矩阵转置
// 4×8 → 8×4 转置 (用于数据布局转换)
__device__ void warp_transpose_4x8(float &val) {
    // lane 0-7: row 0, lane 8-15: row 1, lane 16-23: row 2, lane 24-31: row 3
    // 目标: lane 0-3: col 0, lane 4-7: col 1, ...
    
    // 使用 __shfl_sync 重新排列:
    int src_row = lane_id / 8;
    int src_col = lane_id % 8;
    int dst_lane = src_col * 4 + src_row;
    val = __shfl_sync(0xffffffff, val, dst_lane);
}


// 模式 4: Warp 级广播与聚集
// 广播: 一个线程的值发给所有线程
float broadcast = __shfl_sync(0xffffffff, val, 5);  // 广播 lane 5 的值

// 分段广播: 每段的 lane 0 广播给本段
float seg_broadcast = __shfl_sync(0xffffffff, val, 
    (lane_id / segment_size) * segment_size);

// 聚集: 每个线程从不同源获取数据 (类似查表)
int index = compute_source_lane();  // 每个线程算出自己要从哪个 lane 取
float gathered = __shfl_sync(0xffffffff, val, index);
```

### Warp Vote — 深入与应用

```cuda
// __ballot_sync 返回位掩码, 可以做很多位运算:

unsigned mask = __ballot_sync(0xffffffff, predicate);

// 1. 计算满足条件的线程数:
int count = __popc(mask);

// 2. 计算当前线程在满足条件的线程中的排名 (exclusive prefix popcount):
int rank = __popc(mask & ((1u << lane_id) - 1));
// 如果 mask = 0b10110100 (lane 2,4,5,7 满足):
//   lane 2: rank = 0 (前面没有满足的)
//   lane 4: rank = 1
//   lane 5: rank = 2
//   lane 7: rank = 3

// 这就是 "Stream Compaction" 的 Warp 级实现!
// 将满足条件的元素紧凑打包:
if (predicate) {
    compact_output[base + rank] = my_value;
}

// 3. 找到第一个/最后一个满足条件的 lane:
int first = __ffs(mask) - 1;    // Find First Set (从 LSB)
int last = 31 - __clz(mask);    // Count Leading Zeros

// 4. 提取特定位 (条件选择某个线程的结果):
bool lane5_result = (mask >> 5) & 1;

// 5. Warp 级条件提前退出:
if (__all_sync(0xffffffff, done)) return;
// 只有当所有线程都完成时才退出
// 单个线程不能提前 return (会破坏 Warp 汇合)

// __match_any_sync / __match_all_sync (Volta+):
// 找到 Warp 中哪些线程持有相同的值
unsigned matching = __match_any_sync(0xffffffff, my_key);
// matching 的每个 bit 代表一个线程
// 如果 lane 0, 5, 12 的 my_key 相同, 这三个线程的 matching 中 bit 0,5,12 为 1

// 应用: Warp 内的分组操作 (如 group-by)
```


## 4.3 同步机制 — 从硬件到语义

> Warp Shuffle 是 Warp 内部的通信。但如果需要不同 Warp 之间（同一 Block 内）
> 通信呢？就需要 __syncthreads()——Block 级的同步屏障。
> 更大范围的同步（跨 Block、跨 GPU）有 Cooperative Groups。
> 本节从硬件层面讲清楚这些同步操作到底做了什么。

### __syncthreads() 的硬件实现

```
__syncthreads() 编译为 SASS 指令 BAR.SYNC:

BAR.SYNC 0;    // 屏障编号 0, 等待全部线程

硬件实现:
1. SM 内有专用的 Barrier 硬件单元
2. 每个 Block 最多可以用 16 个不同的 barrier (编号 0-15)
3. 每个 barrier 维护一个计数器:
   - 初始化为 Block 的线程数
   - 每个 Warp 到达时, 计数器减去该 Warp 的活跃线程数
   - 计数器归零 → 所有 Warp 被释放

BAR.SYNC vs BAR.ARRIVE + BAR.WAIT (分离式屏障):
  BAR.SYNC 0;   // arrive + wait 合一 (阻塞)
  
  BAR.ARRIVE 0; // 只通知到达, 不等待 (非阻塞)
  ... 做一些不依赖其他线程的工作 ...
  BAR.WAIT 0;   // 等待所有线程都 arrive
  
  后者允许在等待期间做有用工作 → 减少停顿

__syncthreads() 的代价:
  如果所有 Warp 同时到达: ~几 cycles (硬件计数器操作)
  如果 Warp 到达时间不均匀: 先到达的 Warp stall (Barrier stall)
    → 在 ncu 中表现为 "Stall Barrier"
  
  实测: __syncthreads() 本身 ~20-30 cycles (考虑 Warp 不均匀性)
```

### __syncthreads_count / _and / _or — 带返回值的同步

```cuda
// 同步 + 统计:
int count = __syncthreads_count(predicate);
// 等同于: __syncthreads(); 然后 block-level reduce of predicate
// 但只需要一次同步操作!

// 同步 + 全量判断:
int all = __syncthreads_and(predicate);  // 所有线程的 predicate 都为真?
int any = __syncthreads_or(predicate);   // 任何线程的 predicate 为真?

// 用途: 迭代收敛判断
while (true) {
    float diff = compute_iteration();
    if (__syncthreads_and(diff < epsilon)) break;
    // 只有当所有线程都收敛时才退出
}
```

### 内存栅栏 (Memory Fence) — 比同步更底层

```
__syncthreads() 包含隐式的内存栅栏。
但有时你只需要内存序而不需要执行同步:

__threadfence_block():
  保证当前线程对 Shared Memory 和 Global Memory 的写入
  对同一 Block 的其他线程可见。
  不等待其他线程。

__threadfence():
  保证当前线程对 Global Memory 的写入
  对所有 SM 上的所有线程可见。
  不等待其他线程。

__threadfence_system():
  保证写入对 CPU 和其他 GPU 也可见 (跨 PCIe/NVLink)。
```

```cuda
// 经典用法: Lock-free 的 Block 间通信 (Global Reduce)
__device__ unsigned int block_counter = 0;  // 原子计数器
__device__ float partial_sums[MAX_BLOCKS];

__global__ void global_reduce(float *data, float *result, int n) {
    // 1. Block 内部归约 (常规方法)
    float block_sum = block_reduce_sum(data, n);
    
    if (threadIdx.x == 0) {
        // 2. 写入部分结果
        partial_sums[blockIdx.x] = block_sum;
        
        // 3. 内存栅栏: 确保 partial_sums 写入对其他 Block 可见
        __threadfence();
        
        // 4. 原子递增计数器
        unsigned int old = atomicInc(&block_counter, gridDim.x - 1);
        
        // 5. 最后一个到达的 Block 负责最终归约
        if (old == gridDim.x - 1) {
            // 我是最后一个! 此时所有 Block 的 partial_sums 都已可见
            float total = 0;
            for (int i = 0; i < gridDim.x; i++) {
                total += partial_sums[i];
            }
            *result = total;
            block_counter = 0;  // 重置
        }
    }
}
// 注意: 这个模式需要非常小心! __threadfence() 只保证可见性顺序,
// 不保证写入时间。在高竞争场景下可能有微妙的正确性问题。
```

### Cooperative Groups — 完整 API

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel() {
    // ---- 预定义的分组 ----
    
    // Thread Block 分组 (等价于 __syncthreads 的作用域)
    cg::thread_block block = cg::this_thread_block();
    block.sync();  // == __syncthreads()
    int tid = block.thread_rank();  // == threadIdx.x (对 1D block)
    int size = block.size();        // == blockDim.x * blockDim.y * blockDim.z
    
    // Warp 分组 (Tiled Partition)
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    warp.sync();  // == __syncwarp()
    int lane = warp.thread_rank();
    float sum = cg::reduce(warp, val, cg::plus<float>());  // Warp 级归约!
    
    // 子 Warp 分组 (如 16 个线程为一组)
    cg::thread_block_tile<16> half_warp = cg::tiled_partition<16>(block);
    float half_sum = cg::reduce(half_warp, val, cg::plus<float>());
    
    // ---- Grid 级同步 ----
    cg::grid_group grid = cg::this_grid();
    grid.sync();  // 所有 Block 同步!
    // 约束: 必须用 cudaLaunchCooperativeKernel 启动
    // 约束: gridDim 必须 ≤ 设备支持的最大活跃 Block 数
    //       (所有 Block 必须能同时驻留, 否则死锁)
    
    // ---- Cluster 分组 (Hopper) ----
    // cg::cluster_group cluster = cg::this_cluster();
    // cluster.sync();
    // 同一 Cluster 内的 Block 可以直接访问彼此的 Shared Memory
    
    // ---- 动态分组 (Coalesced Group) ----
    // 在分支后, 获取当前 active 的线程组
    if (threadIdx.x % 3 == 0) {
        cg::coalesced_group active = cg::coalesced_threads();
        // active 只包含进入这个 if 分支的线程
        int rank = active.thread_rank();  // 在 active 组中的排名
        float sum = cg::reduce(active, val, cg::plus<float>());
    }
}
```

### Cooperative Groups 的 reduce / scan 内建

```cuda
// CG 提供了高效的组内归约和扫描:

cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

// 归约 (所有线程得到结果):
float sum = cg::reduce(warp, val, cg::plus<float>());
float max = cg::reduce(warp, val, cg::greater<float>());
float min = cg::reduce(warp, val, cg::less<float>());
int   bor = cg::reduce(warp, ival, cg::bit_or<int>());

// 扫描 (前缀和):
float prefix = cg::inclusive_scan(warp, val, cg::plus<float>());
float excl   = cg::exclusive_scan(warp, val, cg::plus<float>());

// 这些函数内部使用最优的 shuffle 指令实现
// 比手写 shuffle 更简洁, 且编译器可以进一步优化
```


## 4.4 原子操作 — 从硬件到无锁算法

> 同步保证线程"在同一时刻看到一致的数据"。
> 但如果多个线程需要"修改同一个地址"（如多个 Block 往全局计数器累加），
> 仅靠同步不够——需要原子操作保证"读-改-写"是不可分割的。
> 原子操作很强大但也很贵，本节教你什么时候用、什么时候应该避免。

### 原子操作的硬件实现

```
全局内存原子操作:
  1. 请求发送到 L2 Cache (所有原子操作在 L2 级别执行)
  2. L2 的 Atomic Unit 锁定包含目标地址的 Cache Line
  3. 执行 Read-Modify-Write (原子地)
  4. 解锁 Cache Line
  5. 返回旧值 (如果是 atomicXxx 返回值版本)
  
  延迟: ~100-1000 cycles (取决于竞争程度)
  
  如果多个 Warp 原子操作同一地址:
    请求在 L2 排队 → 串行化
    N 个线程竞争 → ~N × 单次延迟

Shared Memory 原子操作:
  直接在 Shared Memory 的 Bank 级别执行
  延迟: ~10-20 cycles (无竞争), 竞争时串行化
  没有 L2 的参与 → 比 Global 原子快很多

Warp 级原子优化 (硬件自动):
  如果同一 Warp 的多个线程 atomicAdd 到同一地址:
  Volta+ 硬件会自动在 Warp 内先合并:
    32 个 atomicAdd → Warp 内先求和 → 1 次原子操作
  → 32× 减少 L2 原子请求!
  
  但这只对 atomicAdd 有效, 其他原子操作不自动合并。
```

### 各精度的原子操作支持

```
操作              int32  int64  float  double  half  half2
atomicAdd         CC1.0  CC6.0  CC2.0  CC6.0   CC7.0  CC6.0*
atomicSub         CC1.0  —      —      —       —      —
atomicMin/Max     CC1.0  CC3.5  —      —       —      —
atomicExch        CC1.0  CC3.5  CC2.0  CC6.0   CC7.0  —
atomicCAS         CC1.0  CC3.5  CC2.0* CC6.0*  —      —
atomicAnd/Or/Xor  CC1.0  CC3.5  —      —       —      —

* float CAS 需要用 int CAS + 位转换:
__device__ float atomicMinFloat(float *addr, float val) {
    int *addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
// 这个 CAS 循环模式可以实现任意的原子 read-modify-write
```

### 减少原子操作竞争的模式

```cuda
// 模式 1: 分层归约 (最重要的模式)
__global__ void histogram(const int *data, int *hist, int n) {
    __shared__ int local_hist[NUM_BINS];
    
    // 初始化局部直方图
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x)
        local_hist[i] = 0;
    __syncthreads();
    
    // 在 Shared Memory 中累加 (原子操作在 SMEM 上快得多)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&local_hist[data[i]], 1);
    __syncthreads();
    
    // 从 Shared Memory 原子写入 Global Memory (每 bin 只有 1 次全局原子!)
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x)
        atomicAdd(&hist[i], local_hist[i]);
}

// 模式 2: Warp 级预聚合
__global__ void reduce_atomic(float *data, float *result, int n) {
    float val = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        val += data[i];
    
    // Warp 内归约 (无原子操作)
    val = warp_reduce_sum(val);
    
    // 每 Warp 只有 lane 0 做一次全局原子操作
    if ((threadIdx.x % 32) == 0)
        atomicAdd(result, val);
    // 竞争减少 32×!
}

// 模式 3: 分 bin 写入 (如果输出地址分散)
// 将全局输出分成多个 bin, 每个 bin 有独立计数器
// 避免所有线程挤在同一个地址上
```


## 4.5 Occupancy — 从数学到实践

### 精确的 Occupancy 计算

```
给定:
  GPU: Ampere (SM_80)
  SM 限制: 2048 max threads, 64 max warps, 32 max blocks, 65536 registers, 164KB smem
  Kernel 参数: blockDim=256, 寄存器/线程=48, shared mem/block=16384

Step 1: Block 大小限制
  每 Block 线程数 = 256 → 8 Warps/Block
  SM 最多 32 Blocks → 32 × 256 = 8192 threads (超过 2048 上限)
  → 受限于 2048/256 = 8 Blocks/SM

Step 2: 寄存器限制
  每 Block 寄存器 = 256 × 48 = 12288
  (实际会向上取整到分配粒度 = 256 寄存器为单位)
  → 12288 → 取整到 12288 (已对齐)
  SM 总寄存器 = 65536
  → 65536 / 12288 = 5.33 → floor = 5 Blocks/SM

Step 3: Shared Memory 限制
  每 Block = 16384 字节 (16KB)
  (向上取整到分配粒度 = 128 字节)
  → 16384 / 128 = 128 个 chunk → 16384 字节
  SM 总 Shared Memory = 164KB (假设全部配为 Shared Mem)
  → 164*1024 / 16384 = 10.25 → floor = 10 Blocks/SM

Step 4: 取最小值
  Thread 限制: 8 Blocks
  Register 限制: 5 Blocks    ← 瓶颈!
  Shared Mem 限制: 10 Blocks
  Block 数限制: 32 Blocks
  
  → 最终: 5 Blocks/SM
  → 5 × 256 = 1280 threads = 40 Warps
  → Occupancy = 40 / 64 = 62.5%
```

### 寄存器分配的粒度细节

```
寄存器分配粒度 (因架构而异):

Volta/Turing: 以 Warp 为单位, 以 256 个寄存器为粒度
  即: 每 Warp 的寄存器 = ceil(寄存器数/线程 × 32 / 256) × 256
  
  例: 每线程 33 个寄存器
  实际分配 = ceil(33 × 32 / 256) × 256 = ceil(4.125) × 256 = 5 × 256 = 1280 / Warp
  每线程实际消耗 = 1280 / 32 = 40 个寄存器 (多了 7 个的浪费!)

Ampere: 同样 256 寄存器粒度

→ 这意味着 "减少 1 个寄存器" 不一定有用,
   只有跨过分配粒度的边界时才会真正释放资源。

查看方法:
  nvcc --ptxas-options=-v 报告的是逻辑寄存器数
  实际分配 = 按上述粒度向上取整
```

### Occupancy 与性能的非线性关系

```
性能
  │
  │                    ┌──── 拐点: 足够隐藏延迟
  │                   ╱ 
  │                  ╱
  │                 ╱       ┌── 更高 Occupancy 几乎无提升
  │                ╱       ╱    (甚至可能因 cache thrashing 变慢)
  │               ╱       ╱
  │              ╱       ╱
  │             ╱  ─────╱───────────────
  │            ╱  
  │           ╱
  │          ╱
  │         ╱
  │        ╱ ← 低 Occupancy: 性能随 Occupancy 快速提升
  │       ╱    (因为 Warp 太少, 无法隐藏延迟)
  │      ╱
  └─────╱──────────────────────── Occupancy (%)
       25%    50%    75%   100%

经验规则:
- Memory Bound kernel: 通常 50% Occupancy 就足够了
- Compute Bound kernel: 25-37.5% 可能就够了 (更多寄存器 = 更少 spilling)
- 混合型: 看 ncu 的 warp stall 分析, 如果 "Not Selected" 占比高 → Occupancy 过剩

著名的反例:
  cuBLAS 的 GEMM 在某些配置下只用 25% Occupancy:
  - 每线程 255 个寄存器 (几乎上限!)
  - 巨大的 Shared Memory tile
  - 但 Tensor Core 的计算密度足够高, 不需要很多 Warp 来隐藏延迟
  → 低 Occupancy 但是 85%+ 的硬件效率!
```

### 动态调整 Occupancy 的技术

```cuda
// 技术 1: 用 Shared Memory 调节 Occupancy
// 如果寄存器是瓶颈, 可以用 Shared Memory 代替部分寄存器存储
__global__ void kernel() {
    __shared__ float staging[BLOCK_SIZE * EXTRA_VARS];
    // 将一些不常用的变量存到 Shared Memory → 释放寄存器 → 更高 Occupancy
}

// 技术 2: 动态选择 launch 配置
int bestBlockSize, minGridSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, kernel);

// 或者手动搜索:
float bestTime = INFINITY;
int bestConfig = 0;
for (int bs = 128; bs <= 1024; bs += 128) {
    float time = benchmark(kernel, bs);
    if (time < bestTime) {
        bestTime = time;
        bestConfig = bs;
    }
}

// 技术 3: __launch_bounds__ 精确控制
__global__ void __launch_bounds__(256, 4) kernel() {
    // maxThreadsPerBlock = 256
    // minBlocksPerSM = 4
    // 编译器保证: 每线程寄存器 ≤ 65536 / (256 × 4) = 64
}

// 如果不指定 __launch_bounds__:
// 编译器默认优化为单 Block/SM 配置 → 可能用很多寄存器
// 显式指定后, 编译器知道需要容纳 4 个 Block → 限制寄存器
```


## 4.6 内存一致性模型 (Memory Consistency)

### GPU 的弱内存模型

```
GPU 使用 Relaxed Consistency Model (弱一致性):
  一个线程的写入不保证对其他线程立即可见!
  写入可能停留在 L1 Cache / Store Buffer 中。

这意味着:

线程 A (Block 0):        线程 B (Block 1):
  data[0] = 42;           while (flag == 0);  // 忙等
  flag = 1;               x = data[0];         // x 可能不是 42!

原因:
  - 线程 A 对 data[0] 的写入可能还在 L1 Cache 中
  - flag 的写入可能先于 data[0] 到达 L2
  - 线程 B 看到 flag=1 后读 data[0], 可能读到旧值

修复:
线程 A:                   线程 B:
  data[0] = 42;
  __threadfence();         // 确保 data[0] 对所有线程可见
  atomicExch(&flag, 1);   while (atomicAdd(&flag, 0) == 0);
                           x = data[0];  // 现在一定是 42

规则:
  1. 同一 Warp 内: 天然可见 (共享寄存器文件)
  2. 同一 Block 内: __syncthreads() 保证可见性
  3. 跨 Block: __threadfence() + 原子操作 保证可见性
  4. 跨 GPU/CPU: __threadfence_system() + 原子操作
```

### Acquire-Release 语义 (CUDA 11+)

```cuda
// CUDA 11+ 支持 C++ 内存序 (通过 cuda::atomic):
#include <cuda/atomic>

cuda::atomic<int, cuda::thread_scope_device> flag(0);
cuda::atomic<int, cuda::thread_scope_device> data_ready(0);

// 生产者:
data[0] = 42;
flag.store(1, cuda::memory_order_release);
// release: 保证之前的所有写入在 flag=1 之前对其他线程可见

// 消费者:
while (flag.load(cuda::memory_order_acquire) == 0);
// acquire: 保证之后的所有读取看到 flag 修改之前的所有写入
int x = data[0];  // 保证看到 42

// Thread Scope:
//   cuda::thread_scope_thread:   线程内 (无意义)
//   cuda::thread_scope_block:    Block 内 (类似 __threadfence_block)
//   cuda::thread_scope_device:   设备内 (类似 __threadfence)
//   cuda::thread_scope_system:   跨设备 (类似 __threadfence_system)
```

### volatile 关键字的作用

```cuda
// volatile 告诉编译器: 每次都必须真正读/写内存, 不能用寄存器中的缓存副本
__device__ volatile int shared_flag;

// 不用 volatile:
while (shared_flag == 0);
// 编译器可能优化为: R0 = shared_flag; while (R0 == 0); → 死循环!
// (R0 被缓存了, 永远看不到其他线程的更新)

// 用 volatile:
while (shared_flag == 0);
// 每次循环真正从内存读取

// 注意: volatile 不能替代原子操作或内存栅栏!
// volatile 只保证编译器不缓存, 不保证硬件的写入可见性
// 正确做法: 用原子操作 + __threadfence
```


## 4.7 Warp 汇合算法 — 编译器如何决定汇合点

### Pre-Volta: 立即后支配 (Immediate Post-Dominator)

```
编译器分析控制流图 (CFG), 找到每个分支的
"立即后支配节点" (Immediate Post-Dominator, IPDOM):
从分支的两条路径都必须经过的第一个点。

例:
         ┌── if ──┐
         │        │
       then     else
         │        │
         └── ◆ ──┘  ← IPDOM: 汇合点
             │
           next

编译器在 IPDOM 处插入 SSY (Set Synchronization Point) 标记,
硬件的收敛栈 (Convergence Stack) 在该处恢复全量 Active Mask。

问题: 循环中的分支
  while (cond) {
      if (divergent) { break; }
      body();
  }
  // break 提前退出的线程在 while 之后等待
  // 但 while 的 IPDOM 是循环之后 → 即使只有 1 个线程还在循环
  // 其他 31 个线程都在等 → 极大浪费
```

### Volta+: 基于调度器的动态汇合

```
Volta 取消了硬件收敛栈, 改用调度器的动态策略:

每个线程有独立的 PC (程序计数器) 和 Active 状态。
Warp Scheduler 每周期:
  1. 收集所有 Active 线程的 PC 值
  2. 找到 PC 相同的线程集合 (称为 Convergence Group)
  3. 选择最大的 Convergence Group 执行
  4. (或者选择 PC 值最小的 → 优先推进落后的线程)

这意味着:
- 不同线程可以在不同代码位置
- 硬件自动找到可以一起执行的线程
- 无需编译器精确计算汇合点

SASS 中的相关指令 (Volta+):
  BSSY B0, target;    // Begin SYnchrony: 声明一个汇合区域
  BSYNC B0;           // 在汇合点等待所有参与线程
  
  这些指令给调度器提供 hint, 帮助更快汇合,
  但不像 Pre-Volta 的 SSY 那样强制。

实际影响:
  循环中 break 的场景 Volta+ 可以更好处理:
  break 的线程立即变为 inactive, 不再占用调度资源
  而不是在循环外空转等待 (Pre-Volta 的行为)
```

### __syncwarp() 在 Volta+ 中的必要性

```
Pre-Volta: 同一 Warp 的线程锁步执行, 隐式同步。
  val = smem[lane ^ 1];  // 读邻居的值 → 锁步保证邻居已写入

Volta+: 独立线程调度, 不保证锁步!
  线程 0 可能已执行到第 10 行, 线程 1 还在第 5 行。
  
  必须显式同步:
  smem[lane] = my_val;
  __syncwarp();             // 确保所有 lane 都完成了写入
  float neighbor = smem[lane ^ 1];  // 现在安全
  
  __syncwarp() 编译为:
  BAR.SYNC 0, 0x1f;  // 或 WARPSYNC 指令
  
  它和 __syncthreads() 的区别:
  __syncthreads(): Block 内所有线程同步 (跨 Warp)
  __syncwarp():    只同步当前 Warp 的 32 个线程 (极轻量)
```


## 4.8 原子操作的 SASS 编码与硬件路径

### 全局内存原子操作的 SASS

```
atomicAdd(&global_ptr[idx], val);

编译为 SASS:
  RED.E.ADD.F32.FTZ.RN [R2], R0;   // Reduce (原子归约)
  // RED = Atomic Reduce, 不返回旧值
  // 如果需要返回旧值:
  ATOM.E.ADD.F32.FTZ.RN R4, [R2], R0;  // Atomic, R4 = 旧值

硬件路径:
  1. LD/ST Unit 发出原子请求 (包含地址 + 操作数 + 操作类型)
  2. 请求通过 NoC 到达 L2 Cache
  3. L2 的 Atomic Unit 执行:
     a. 锁定包含目标地址的 Sector
     b. 读取当前值
     c. 执行运算 (如 ADD)
     d. 写入新值
     e. 解锁 Sector
     f. 如果是 ATOM (需要返回旧值), 将旧值返回给 SM
  
  延迟: ~100 cycles (无竞争) → ~1000+ cycles (高竞争)
  
  竞争时的串行化:
    如果 N 个请求同时到达 L2 的同一 Sector:
    它们在 L2 的 Atomic Unit 排队串行执行。
    延迟 ≈ N × 单次延迟

Shared Memory 原子操作:
  ATOMS.ADD R4, [R2], R0;  // Shared Memory Atomic Add
  
  在 Shared Memory 的 Bank 级别执行, 不经过 L2。
  延迟: ~20 cycles (无竞争)
  同一 Bank 的竞争: 串行化 (和 Bank Conflict 类似)
```

### Warp 级原子合并 (Hardware Coalescing)

```
Ampere+ 对 atomicAdd 有硬件级合并优化:

场景: 一个 Warp 的 32 个线程都对同一地址 atomicAdd

Pre-Ampere:
  32 个独立的 L2 原子请求 → 串行执行 → ~3200 cycles

Ampere+:
  硬件检测到同一 Warp 的多个 atomicAdd 指向同一地址:
  1. 在 Warp 内先做归约 (类似 warp_reduce_sum)
  2. 只发 1 个原子请求到 L2 (值 = 32 个值之和)
  3. 延迟 ≈ ~100 cycles (只有 1 次 L2 原子)

  加速: 32× 减少 L2 压力!

  但这个优化有限制:
  - 只对 atomicAdd 有效 (不对 atomicMin/Max/CAS)
  - 只对同一 Warp 内的请求合并 (跨 Warp 不合并)
  - 如果 32 个线程的目标地址不全相同, 则按地址分组合并

在 SASS 中可以看到:
  RED.E.ADD.F32.FTZ.RN [R2], R0;  // Warp-level coalesced reduce
  vs
  ATOM.E.ADD.F32.FTZ.RN R4, [R2], R0;  // 需要旧值 → 不能合并!
  
  → RED 指令可以被硬件合并, ATOM 不行 (因为每个线程需要不同的旧值)
  → 如果不需要旧值, 用 RED (atomicAdd 不使用返回值时编译器自动选择)
```


## 4.9 本章总结

```
Warp 是 GPU 的灵魂: 所有调度、执行、同步都围绕 Warp 进行。

分支分歧: 同一 Warp 内走不同分支 → 串行执行两条路径
  解决: 对齐 Warp 边界 / 无分支算术 / 数据重排
  Volta+: 独立线程调度 + BSSY/BSYNC 动态汇合

Warp 原语:
  Shuffle: 线程间寄存器直接交换 (~1 cyc, 不经 SMEM)
  Vote: ballot/any/all → 快速位操作
  应用: 归约, 前缀和, 排序, 转置, 紧凑

同步:
  __syncthreads() → BAR.SYNC (硬件 Barrier, Block 级)
  __syncwarp() → Warp 级, Volta+ 必须显式使用
  Cooperative Groups → Grid/Cluster 级同步

原子操作:
  在 L2 Atomic Unit 执行, 高竞争时串行化
  RED vs ATOM: RED 可被硬件合并 (Ampere+), ATOM 不行
  分层归约: Warp → Block → Grid → 最少化原子操作次数

内存一致性: GPU 弱一致性 → 写入不保证立即可见
  __threadfence() + 原子操作 = 跨 Block 通信的正确姿势
```


## 4.10 Q&A

### Q: __syncthreads() 放在 if 分支里为什么死锁, 但 __syncwarp() 不会?

```
__syncthreads() = BAR.SYNC: 硬件计数器等待 Block 内所有线程到达。
如果部分线程永远不执行这行 → 计数器永远不归零 → 死锁。

if (threadIdx.x < 128) {
    __syncthreads();  // Thread 128-255 永远不到达 → 死锁!
}

__syncwarp() = Warp 内 32 线程同步:
因为同一 Warp 的 32 线程必须执行相同的控制流 (即使被 mask),
所以 __syncwarp() 永远会被所有 active 线程执行到。
但要注意传入正确的 mask (Volta+ 要求显式指定哪些线程参与)。
```

### Q: volatile 能替代 __threadfence() 吗?

```
不能! 它们解决的是不同层面的问题:

volatile: 告诉编译器 “不要缓存这个变量到寄存器, 每次都从内存读”。
  解决的是: 编译器优化导致看不到更新 (软件层面)。

__threadfence(): 告诉硬件 “确保我之前的写入对其他 SM 可见”。
  解决的是: 硬件缓存/写缓冲导致其他 SM 看不到更新 (硬件层面)。

两者都需要! volatile 防编译器优化, __threadfence 防硬件重排序。
但现代 CUDA 推荐用 cuda::atomic 代替这两者 (更安全更清晰)。
```

### 概念辨析: Warp Divergence 的代价不总是 2×

```
常见误解: “分支分歧 = 性能减半”

实际代价取决于:
  1. 两条路径的长度: 如果 path_A = 100 cyc, path_B = 1 cyc
     分歧代价 = 100 + 1 = 101 cyc (vs 无分歧 100 cyc) → 只多 1%
  
  2. 短分支被谓词化: 编译器可能用谓词指令替代跳转
     @P0 FADD R0, R1, R2;  → 所有线程都执行, 但只有 P0=true 的写入
     这根本不是“分歧”, 而是 “预测执行” → 代价很小
  
  3. 分歧只在 Warp 内部产生, 不同 Warp 走不同分支完全没代价
```


## 4.11 练习题

配套代码在 [`theory/exercises/`](./exercises/) 目录下: [`ch04_ex1_divergence.cu`](./exercises/ch04_ex1_divergence.cu) / [`ch04_ex2_shuffle.cu`](./exercises/ch04_ex2_shuffle.cu) / [`ch04_ex3_deadlock.cu`](./exercises/ch04_ex3_deadlock.cu)

### 练习 1: 分支分歧探索 [难度: ⭐⭐]

```
打开 11_warp_divergence/warp_divergence.cu:

1. 把 "50% 分歧" 的条件从 threadIdx.x % 2 改成 threadIdx.x % 4。
   预测: 性能会变差还是变好? 为什么?
   (提示: %4 意味着 4 条路径 → 分歧更严重?
    但实际上编译器可能将短分支谓词化, 试试看!)

2. 把 if/else 的两条路径加长 — 每条路径做 10 次乘法而不是 1 次。
   这时分歧的代价更明显吗?
   (理论预测: 路径越长, 分歧串行化的代价越大)
```

### 练习 2: Warp Shuffle 归约 [难度: ⭐⭐⭐]

```
打开 03_reduce/reduce.cu:

1. 将 V2 (Warp Shuffle) 中的 __shfl_down_sync 换成 __shfl_xor_sync。
   提示: __shfl_xor_sync(0xffffffff, val, 16) 等价于
         "lane i 和 lane i^16 交换并相加"
   这就是“蝶形归约”(butterfly reduce)。
   验证结果是否一致。

2. 写一个 warp_reduce_max 函数 (将加法换成 fmaxf)。
   用它来实现 "Warp 内求最大值"。
   (配合理论: 本章 4.2 节 "Warp Shuffle 实现归约")
```

### 练习 3: 手写 __syncthreads 死锁 [难度: ⭐]

```
写一个小 kernel, 故意在 if 分支里放 __syncthreads():

__global__ void deadlock_demo() {
    if (threadIdx.x < 128) {
        __syncthreads();  // 只有一半线程到达!
    }
}

编译并运行。观察: 程序会怎样? (挂起/超时/报错?)
然后把 __syncthreads 移到 if 外面, 确认程序正常结束。
(配合理论: 本章 4.3 节 "__syncthreads 的关键规则")
```
