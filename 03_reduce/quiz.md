# 自测题: 并行归约 (Reduce)

配合 [`03_reduce/reduce.cu`](./reduce.cu) 和 [`theory/04_warp_and_sync.md`](../theory/04_warp_and_sync.md)

---

**Q1: V0 (交错归约) 为什么有 Warp Divergence?**

A) 因为用了 Shared Memory
B) 因为 stride 从 1 开始, 同一 Warp 内奇偶线程走不同分支
C) 因为没有用 `__syncthreads()`
D) 因为 blockSize 不是 32 的倍数

<details><summary>答案</summary>
**B**。V0 中 `if (tid % (2*stride) == 0)` 导致 stride=1 时, tid 0, 2, 4, 6... 走 if, tid 1, 3, 5, 7... 跳过 → 同一 Warp 内执行不同路径 → 串行执行 → 性能减半。
</details>

---

**Q2: V1 为什么没有 Divergence?**

A) 因为用了 Warp Shuffle
B) 因为 stride 从 blockDim/2 开始递减, 前半部分线程全部工作
C) 因为每个 Warp 只包含连续 32 个线程
D) 因为减少了 `__syncthreads()`

<details><summary>答案</summary>
**B**。V1 的 `if (tid < stride)` 使得 stride=128 时线程 0-127 (Warp 0-3) 全部工作, 线程 128-255 (Warp 4-7) 全部不工作。整个 Warp 走同一分支 → 无分歧。
</details>

---

**Q3: V3 (Warp Shuffle) 的 `__shfl_down_sync(0xffffffff, val, offset)` 做了什么?**

A) 把 val 存到 Shared Memory
B) 从编号比自己大 offset 的线程读取 val, 不经过任何内存
C) 同步同 Warp 的所有线程
D) 把 val 广播给所有 Warp

<details><summary>答案</summary>
**B**。`__shfl_down_sync` 是寄存器级通信: 从同一个 Warp 内编号更大的 lane 直接读取寄存器值, 延迟 ~1 cycle, 不需要 `__syncthreads()`。
</details>

---

**Q4: 第一个参数 `0xffffffff` 是什么意思?**

A) 值 4294967295
B) 32 位全 1 的掩码 → 所有 32 个 lane 都参与 Shuffle
C) Shared Memory 的起始地址
D) 目标 lane 的编号

<details><summary>答案</summary>
**B**。这是 participant mask: 每位对应一个 lane(线程), 1=参与。`0xffffffff` = 全部 32 位为 1 = 全部 32 个 lane 参与。如果写成 `0x0000ffff` 则只有 lane 0-15 参与。
</details>

---

**Q5: V3 比 V1 快, 主要原因是什么?**

A) 用了更多 Shared Memory
B) `__syncthreads()` 从 8 次降到 1 次 + Warp 内归约零同步
C) 用了 float4 向量化
D) 减少了 gridSize

<details><summary>答案</summary>
**B**。V1 每轮归约后都需要 `__syncthreads()`, 总共 8 次。V3 的 Warp 内归约 (32→1) 全用 Shuffle, 零 `__syncthreads()`。只有 Warp 间需要 1 次 `__syncthreads()`。
</details>

---

**Q6: V4 (ILP) 的 4 路循环展开为什么有效?**

A) 每条加载指令的数据量大了 4 倍
B) 4 条独立的 LDG 背靠背发射 → 等第 1 条的延迟时, 后面 3 条已经在路上了
C) 编译器自动优化了
D) 减少了 Shared Memory 的使用

<details><summary>答案</summary>
**B**。`LDG` (Global Load) 延迟 ~500 cycles。串行: 500 + 等 + 500 + 等 = 2000 cycles。4 路展开: 4 条 LDG 连续发射 → 总等待 ~500 cycles (4 条重叠) → 平均每元素 ~125 cycles。
</details>

---

**Q7: V5 (float4) 为什么比 V4 的 4 路展开更快?**

A) 每次加载更多字节
B) 1 条 LDG.128 替代 4 条 LDG.32, 指令数减少 → LD/ST 流水线压力降低
C) 绕过了 L1 Cache
D) 用了 Tensor Core

<details><summary>答案</summary>
**B**。float4 和 4 路展开传输的数据量相同 (都是 16 bytes), 但 float4 用 1 条向量化指令替代 4 条标量指令 → 指令发射数减 4 倍 → LD/ST pipeline 的占用率降低 → 可以同时处理更多 Warp 的请求。
</details>

---

**Q8: V6 (atomicAdd) 单 kernel 版本的缺点是什么?**

A) 不能用 float4
B) `atomicAdd` 在输出地址上有序列化 → 所有 Block 同时写同一地址 → 等锁
C) Shared Memory 不够用
D) 编译不了

<details><summary>答案</summary>
**B**。所有 Block 的 `atomicAdd` 都指向同一个 `output`, 硬件需要序列化这些操作。Block 数少时 (256 个) 影响不大, Block 数多时 (16384 个) 成为瓶颈。V6 通常需要先 `cudaMemset` 清空 output。
</details>

---

**计分**: 每题 1 分, 满分 8 分
- 7-8: 你已经完全理解了 Reduce 优化路径, 可以继续学习 Softmax/LayerNorm
- 5-6: 基本理解, 建议重读 V3 (Warp Shuffle) 和 V4 (ILP) 部分
- <5: 建议重新跑 `./03_reduce/reduce`, 逐版对比输出
