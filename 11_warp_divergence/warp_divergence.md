# Warp Divergence：分支指令在硬件上的真实行为

配合 `warp_divergence.cu` 阅读。

**难度**: ⭐⭐ 进阶
**前置知识**: Warp 执行模型（[tutorial Part 3](../tutorial.md#part-3-warp-线程调度和-shuffle)）
**读完你能做什么**: 理解 SIMT 下分支指令的硬件行为（Active Mask / 谓词执行），能写出避免 Warp 分歧的代码


## 什么是 Warp Divergence (分支分歧)

### GPU 怎么执行 if/else

先回忆一个关键事实：GPU 以 **Warp (32 线程)** 为单位执行指令。
同一 Warp 的 32 个线程在同一时刻执行**同一条指令**——只是各自操作不同的数据。

当代码遇到 if/else 时会发生什么？

```cuda
if (threadIdx.x < 16) {
    path_A();    // 前 16 个线程走这里
} else {
    path_B();    // 后 16 个线程走这里
}
```

在一个 Warp 中，Thread 0-15 满足条件，Thread 16-31 不满足。
但 GPU 不能让一半线程执行 path_A、另一半同时执行 path_B——
因为 32 个线程共享同一个指令指针，只能执行同一条指令！

```
GPU 实际的执行顺序:

第 1 阶段: 执行 path_A 的指令
  Thread 0-15:  执行! (活跃)
  Thread 16-31: 空等  (被屏蔽, 什么都不做)

第 2 阶段: 执行 path_B 的指令
  Thread 0-15:  空等  (被屏蔽)
  Thread 16-31: 执行! (活跃)

→ 两条路径串行执行!
→ 如果 path_A 和 path_B 各需要 T 时间, 总时间 = 2T (而不是 T)
→ 性能减半!
```

### 分歧的严重程度取决于 Warp 内的分布

关键点：**分歧只在同一个 Warp 内部才有影响。**

```
不同 Warp 走不同分支 → 完全没问题!
  Warp 0 的 32 线程全走 path_A → Warp 0 只执行 A → 无分歧
  Warp 1 的 32 线程全走 path_B → Warp 1 只执行 B → 无分歧
  → 每个 Warp 只执行一条路径 → 性能正常!

同一 Warp 走不同分支 → 分歧!
  Warp 0 中: Thread 0-15 走 A, Thread 16-31 走 B → 分歧!
  → 这个 Warp 必须串行执行两条路径 → 性能减半!
```

### 怎么在代码中避免

对比 `warp_divergence.cu` 中的两种写法：

```cuda
// 写法 1: 以线程为边界 → 每个 Warp 内部都有分歧
if (threadIdx.x % 2 == 0) {   // 奇偶线程走不同路径
    output[idx] = val * val;    // Thread 0,2,4...走这里
} else {
    output[idx] = val * 0.5f;   // Thread 1,3,5...走这里
}
// Warp 0: Thread 0 走 if, Thread 1 走 else, Thread 2 走 if...
// → 每个 Warp 都分歧 → 全部慢一倍

// 写法 2: 以 Warp 为边界 → 无分歧
int warp_id = threadIdx.x / 32;
if (warp_id % 2 == 0) {       // 整个 Warp 走同一路径
    output[idx] = val * val;    // Warp 0,2,4...走这里
} else {
    output[idx] = val * 0.5f;   // Warp 1,3,5...走这里
}
// Warp 0 的 32 线程全走 if → 无分歧!
// Warp 1 的 32 线程全走 else → 无分歧!
// → 性能正常

// 写法 3: 无分支算术 → 完全消除分支指令
float mask = (threadIdx.x % 2 == 0) ? 1.0f : 0.0f;
output[idx] = mask * val * val + (1.0f - mask) * val * 0.5f;
// 没有 if/else, 所有线程走同一条指令序列
// → 零分歧! 但代价是: 所有线程都要做两条路径的计算
// → 只有当两条路径很短时才值得
```


## 无分歧 vs 有分歧的 SASS 对比

### 无分歧 (以 Warp 为边界判断)

```cuda
if (warp_id % 2 == 0) { output[idx] = val * val + val; }
else                   { output[idx] = val * 0.5f - 1.0f; }
```

```
SASS (简化):
  ISETP.EQ P0, R_warp_id_mod2, 0 ;   // P0 = (warp_id % 2 == 0)
  @P0 BRA PATH_A ;
  // PATH_B:
  FMUL R2, R0, 0.5 ;
  FADD R2, R2, -1.0 ;
  BRA END ;
  PATH_A:
  FFMA R2, R0, R0, R0 ;    // val*val+val (FMA)
  END:
  STG.E [R4], R2 ;

硬件行为:
  Warp 0 (全是 warp_id=0): P0 全 true → 全走 PATH_A → 无分歧!
  Warp 1 (全是 warp_id=1): P0 全 false → 全走 PATH_B → 无分歧!
  
  每个 Warp 内 32 线程走同一路径 → 正常速度执行。
  不同 Warp 走不同路径 → 完全没问题 (它们本来就是独立调度的)。
```


### 50% 分歧 (奇偶线程判断)

```cuda
if (threadIdx.x % 2 == 0) { output[idx] = val * val + val; }
else                       { output[idx] = val * 0.5f - 1.0f; }
```

```
SASS 行为 (分歧!):
  ISETP.EQ P0, R_tid_mod2, 0 ;
  
  同一 Warp 内:
    Thread 0:  P0 = true  → 想走 PATH_A
    Thread 1:  P0 = false → 想走 PATH_B
    Thread 2:  P0 = true  → 想走 PATH_A
    ...
    → 16 个线程要 A, 16 个要 B → 分歧!

  GPU 硬件的处理方式 (Pre-Volta):
    Step 1: 执行 PATH_A, Active Mask = 0x55555555 (偶数线程)
            Thread 0,2,4,...30 的 FMA 正常计算
            Thread 1,3,5,...31 被 mask, ALU 空转
    
    Step 2: 执行 PATH_B, Active Mask = 0xAAAAAAAA (奇数线程)
            Thread 1,3,5,...31 的 FMUL+FADD 正常计算
            Thread 0,2,4,...30 被 mask, ALU 空转
    
    Step 3: 汇合, Active Mask = 0xFFFFFFFF (全部恢复)
  
  代价: PATH_A 和 PATH_B 串行执行!
    时间 = time(A) + time(B), 而不是 max(time(A), time(B))
    如果两条路径等长 → 性能减半!

### Volta+ 的变化: Independent Thread Scheduling (ITS)

**关键**: 上面描述的 Pre-Volta 行为在现代 GPU 上已经变了!
从 Volta (SM 7.0) 开始, NVIDIA 引入了 **Independent Thread Scheduling** —
每个线程有自己的程序计数器 (PC) 和调用栈, 不再强制锁步执行。

```
Volta+ 的硬件行为 (以同样的奇偶分歧为例):

  同一 Warp 内 Thread 0 (P0=true) 和 Thread 1 (P0=false):
    Pre-Volta: 整个 Warp 先执行 PATH_A, 再执行 PATH_B (串行)
    Volta+:    两个线程可以交错执行! 调度器可以在指令级别交替发射

  具体来说:
    Cycle 0: Thread 0 发射 PATH_A 的第 1 条指令 (FFMA)
    Cycle 1: Thread 1 发射 PATH_B 的第 1 条指令 (FMUL)
             ↑ 不需要等 Thread 0 走完 PATH_A!
    Cycle 2: Thread 0 发射 PATH_A 的第 2 条指令
    ...

  这意味着:
    - 短分支 (1-3 条指令): 开销几乎为零 — 编译器通常直接用谓词消除
    - 中等分支 (4-10 条指令): 开销比 Pre-Volta 小得多, 交错执行掩盖了部分延迟
    - 长分支 (10+ 条指令): 两条路径仍然要串行完成, 只是交错粒度更细

  ITS 的关键限制 — reconvergence 不再自动发生!
    Pre-Volta: 分支结束后, 硬件自动将所有线程的 Active Mask 恢复为全 1
    Volta+:    分支结束后, 线程**不会自动汇合**!
              必须显式调用 __syncwarp() 才能让 Warp 重新同步。
              不调用 __syncwarp() → 线程继续独立执行 → 后续代码可能读到未同步的数据!
```

**`__syncwarp()` — Volta+ 上的正确性必需**:

```cuda
// Volta+ 上的正确写法:
if (threadIdx.x < 16) {
    path_A();
} else {
    path_B();
}
__syncwarp();  // ← 必须! 让走了不同路径的线程重新同步

// 然后才能安全地做 Warp Shuffle 或访问 Shared Memory:
float val = __shfl_down_sync(0xffffffff, my_val, 1);  // 安全: 所有线程已同步
```

```
如果不加 __syncwarp():
  Thread 0 (走了 path_A, 3 条指令) 已经执行到 __shfl_down_sync
  Thread 16 (走了 path_B, 5 条指令) 还在执行 path_B 的最后一条
  → __shfl_down_sync 等不到 Thread 16 → deadlock 或读到未定义值!
  → 而且只在某些 GPU 上出错, 换个 GPU 就正常 → 最难调试的 bug!

经验法则 (Volta+):
  1. 分支后如果要使用 Warp Shuffle → 必须先 __syncwarp()
  2. 分支后如果要访问 Shared Memory (同一 Warp 内不同线程写入) → 先 __syncwarp()
  3. 如果分支以 Warp 为边界 (所有 32 线程走同一条路径) → 不需要 __syncwarp()
  4. 如果不确定 → 加上 __syncwarp() (开销<10 cycles, 比 debug 一周便宜得多)
```

### 循环中的分歧 (最坏情况)

```
// 每线程循环次数不同 → 典型的"最坏分歧"
for (int i = threadIdx.x; i < N; i += 32) {
    process(data[i]);  // Thread 0 循环 100 次, Thread 31 循环 1 次
}
// Thread 31 早就完成了, 但必须等 Thread 0 做完所有 100 次!
// → 31 个线程空转 99 次迭代的时间!

SASS 级别: 循环的 back-edge (跳回循环头) 会被 Active Mask 控制。
只有当所有线程都到达循环出口时, 整个 Warp 才能继续执行后面的代码。
→ 循环次数由"最慢的线程"决定。
→ 这就是为什么 if (idx < N) 的保护下, N 不是 32 的倍数时会有尾端损耗。
```

**Active Mask 的十六进制解读**:
```
0x55555555 = 0101 0101 0101 ... (32位)
  → bit 0=1 (Thread 0 活跃), bit 1=0 (Thread 1 屏蔽), bit 2=1, bit 3=0...
  → 偶数线程活跃, 奇数线程屏蔽

0xAAAAAAAA = 1010 1010 1010 ... (32位)
  → bit 0=0 (Thread 0 屏蔽), bit 1=1 (Thread 1 活跃), bit 2=0, bit 3=1...
  → 奇数线程活跃, 偶数线程屏蔽

0xFFFFFFFF = 全部 32 线程活跃
0x0000FFFF = 只有低 16 线程活跃 (Thread 0-15)
```
```


### 无分支算术 (编译器优化)

```cuda
output[idx] = (threadIdx.x % 2 == 0) ? path_a : path_b;
```

```
编译器可能优化为谓词指令 (无分支跳转!):

  FFMA R2, R0, R0, R0 ;       // 先算 path_a (全 32 线程都算)
  FMUL R3, R0, 0.5 ;          // 先算 path_b (全 32 线程都算)
  FADD R3, R3, -1.0 ;
  @P0 MOV R4, R2 ;            // 如果 P0=true: 用 path_a
  @!P0 MOV R4, R3 ;           // 如果 P0=false: 用 path_b
  STG.E [R6], R4 ;

  两条路径都执行了! 但没有分支跳转 → 没有 Active Mask 切换!
  只是最后用谓词选择 (SEL/MOV) 哪个结果写入。
  
  代价: 计算了两条路径 (有冗余), 但避免了分歧的串行化开销。
  当路径很短时 (1-2 条指令), 这比真正的分歧更快。
  当路径很长时, 冗余计算的代价超过分歧 → 编译器会选择真正的分支。
```


## 什么时候分歧代价大, 什么时候小

```
┌───────────────────────────┬──────────────────────────────┐
│ 场景                      │ 分歧代价                      │
├───────────────────────────┼──────────────────────────────┤
│ 两条路径都只有 1-3 条指令  │ 几乎为零 (编译器谓词化)       │
│ 两条路径各 10+ 条指令      │ 接近 2× 减速                 │
│ 一条路径 100 条, 另一条 1  │ 不是减半, 而是慢 101/100 ≈ 1% │
│ 分歧以 Warp 为边界        │ 零代价 (不同 Warp 独立调度)   │
│ 循环中的分歧 (不同迭代数)  │ 最坏: 最慢线程决定整个 Warp    │
└───────────────────────────┴──────────────────────────────┘
```


## 练习题

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_warp_uniform_level1.cu](./exercises/ex1_warp_uniform_level1.cu) | 用 Warp-uniform 条件消除分歧 | 让分支以 Warp 为单位变化（只填 kernel） |

```bash
nvcc -O2 --extended-lambda -o ex1_warp_uniform_level1 ex1_warp_uniform_level1.cu
```
