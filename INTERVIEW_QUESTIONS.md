# CUDA 面试题

每章 2-4 题，覆盖核心概念。答案在折叠区，先自己想再看答案。

---

## 01 vector_add — 第一个 CUDA 程序

**Q1.1** 下面这段 kernel launch 有什么问题？

```cuda
vector_add<<<1, 1024>>>(d_a, d_b, d_c, N);  // N = 1000000
```

<details>
<summary>答案</summary>

只有 1024 个线程，但 N=1,000,000 个元素。每个线程只处理了 1 个元素，剩余 999,000+ 个元素没被处理。应该用 Grid-Stride Loop 或增加 grid 数量：

```cuda
int grid = (N + 255) / 256;
vector_add<<<grid, 256>>>(d_a, d_b, d_c, N);
```
</details>

**Q1.2** 下面代码的输出是什么？为什么？

```cuda
float *d_a;
cudaMalloc(&d_a, N * sizeof(float));
// 忘了 cudaMemcpy
vector_add<<<grid, block>>>(d_a, d_b, d_c, N);
```

<details>
<summary>答案</summary>

输出是随机值（垃圾数据）。`cudaMalloc` 分配显存但**不会自动清零**，`d_a` 里的数据是显存上次被使用后残留的值。需要用 `cudaMemcpy` 把 host 数据拷过去，或者用 `cudaMemset` 清零。
</details>

**Q1.3** 同一个 kernel，下面两种写法哪个快？为什么？

```cuda
// A: 大 grid, 小 block
kernel<<<1024, 64>>>(data, N);

// B: 小 grid, 大 block
kernel<<<64, 1024>>>(data, N);
```

<details>
<summary>答案</summary>

通常 **B 更快**。原因：
- Launch overhead 正比于 grid size（每个 Block 需要 SM 分配资源），A 有 1024 个 Block，B 只有 64 个
- 大 Block 内线程更多 → 同一 SM 上能隐藏更多延迟
- 但 Block 太大（1024）可能导致寄存器/Shared Memory 不够 → Occupancy 下降
- 一般来说 blockDim=256 是安全默认，具体要看你 kernel 的资源用量
</details>

---

## 02 matrix_mul — Shared Memory Tiling

**Q2.1** 为什么 Tiled 矩阵乘法需要**两个** `__syncthreads()`，而很多人只写了一个？

<details>
<summary>答案</summary>

第一个 `__syncthreads()`：保证所有线程都写完了 Shared Memory 之后才能开始读（写→读 barrier）。

第二个 `__syncthreads()`：保证所有线程都算完了当前 tile 之后才能覆盖 Shared Memory 写下一轮的数据（读→写 barrier）。

如果只有第一个：某 Warp 算得快，已经进入下一轮循环开始往 As 写新数据，但另一个 Warp 还在读旧数据 → 读到混合的新旧数据 → 结果错误。
</details>

**Q2.2** Tiled 矩阵乘法中，`As[threadIdx.y][threadIdx.x]` 写成 `As[threadIdx.x][threadIdx.y]` 会怎样？

<details>
<summary>答案</summary>

这是两种不同的映射方式：

- `As[threadIdx.y][threadIdx.x]`（正确）：`threadIdx.x` 连续→同一 Warp 写连续的列→合并写入。读的时候 `As[threadIdx.y][i]`→同一 Warp 的 threadIdx.y 不同→访问不同 Bank→无冲突。

- `As[threadIdx.x][threadIdx.y]`（错误）：写的时候 threadIdx.x 变化→跨行写→跨步访问→写入不合并。而且 `As[i][threadIdx.x]` 读的时候同一 Warp 的 threadIdx.x 连续→都读同一列→可能 Bank Conflict。

性能差距 ~2-5×。
</details>

**Q2.3** TILE_SIZE 从 16 改成 32（blockDim 从 256 变成 1024）会有什么影响？

<details>
<summary>答案</summary>

好处：
- 每个 tile 更大 → K 维度循环次数减半 → `__syncthreads()` 次数减半
- 数据复用更高（同一行 32 个线程复用 vs 16 个）

代价：
- Shared Memory 需要 `32×32×4B×2 = 8KB`（vs 原来的 2KB）
- 寄存器压力增大：sum 要累加 32 次而非 16 次
- blockDim=1024 → 每个 SM 最多放 1-2 个 Block → Occupancy 下降
- Ampere 上 SMEM 上限 100KB → 8KB 没问题
- 但 32×32=1024 线程 → 8KB SMEM / 1024 线程 = 8B/线程 → 寄存器可能成为瓶颈

实际效果取决于矩阵大小和 GPU 架构，需要用 ncu 实测。
</details>

---

## 03 reduce — 并行归约

**Q3.1** 下面两种归约顺序，哪种更好？为什么？

```cuda
// A: 相邻归约 (stride loop)
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) sdata[tid] += sdata[tid + s];
    __syncthreads();
}

// B: 交错归约 (stride loop, reverse)
for (int s = blockDim.x/2; s > 0; s /= 2) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

<details>
<summary>答案</summary>

**B 更好**。两个原因：

1. **线程利用率**：B 中第 1 轮有 blockDim/2 个线程干活，A 中第 1 轮只有 blockDim/2 个线程干活（一样）。但 B 中活跃线程是前 s 个（同一 Warp），A 中活跃线程是 `tid % (2*s) == 0` 的那些，分散在不同 Warp → Warp 利用率低 → 更多 `__syncthreads()` 开销。

2. **Bank Conflict**：B 中 `sdata[tid]` 和 `sdata[tid+s]` 随着 s 减小，地址越来越近，Bank Conflict 有规律（2-way, 4-way...）。A 中 stride 从 1 开始增大，冲突模式复杂。

综合来看 B 快 2-3×。
</details>

**Q3.2** `__shfl_down_sync(0xffffffff, val, offset)` 的 mask 参数 `0xffffffff` 是什么意思？写成 `0x0000ffff` 会怎样？

<details>
<summary>答案</summary>

`0xffffffff` = 32 个 1，表示所有 32 个 lane 都参与 shuffle。

`0x0000ffff` = 只有低 16 个 lane 参与（lane 0-15）。

如果写成 `0x0000ffff`：
- lane 16-31 不参与，它们不会提供自己的值
- `__shfl_down_sync(0x0000ffff, val, 8)` → lane 0 试图从 lane 8 拿值（lane 8 < 16，参与，OK）
- 但如果 mask 暗示某些 lane 不参与，而实际上它们还活着 → **未定义行为**，实际表现可能是 hang 或读到垃圾值

关键规则：mask 必须**精确匹配**实际活跃的 lane 集合。
</details>

---

## 04 pytorch_extension — PyTorch 接入

**Q4.1** PyTorch 自定义 CUDA 算子需要哪三段代码？各自做什么？

<details>
<summary>答案</summary>

1. **CUDA kernel** (`.cu`)：在 GPU 上跑的计算逻辑。输入 `float*`，输出 `float*`，纯 C 风格。

2. **C++ binding** (`.cu`，同一个文件)：桥接 PyTorch Tensor 和 CUDA kernel。
   - 校验输入（`TORCH_CHECK(x.is_cuda())`）
   - 分配输出（`torch::empty_like(x)`）
   - 调 kernel（`my_kernel<<<grid, block>>>(x.data_ptr<float>(), ...)`）
   - 用 `PYBIND11_MODULE` 暴露给 Python

3. **Python autograd.Function** (`.py`)：
   - `forward()` 调 C++ binding，`ctx.save_for_backward()` 保存给反向传播
   - `backward()` 计算梯度
   - 让算子能嵌入 autograd 计算图
</details>

**Q4.2** 为什么调用 kernel 前要检查 `x.is_contiguous()`？

<details>
<summary>答案</summary>

PyTorch Tensor 可能是非连续的（如 `x.transpose(0,1)` 或 `x[::2]` 的切片）。非连续 tensor 的元素在内存中不是紧密排列的，`data_ptr<float>()` 返回的是首地址，但按 row-major 的索引计算会访问到错误的位置。

连续 tensor：`x[i*stride + j]` 的物理地址 = base + (i*stride + j)*4
非连续 tensor（如转置）：逻辑上 `x[i][j]` 但物理存储不连续，索引公式不适用

解决办法：调 `.contiguous()` 或手动处理 stride。
</details>

---

## 05 bank_conflict — Bank Conflict

**Q5.1** 什么是 Shared Memory Bank Conflict？32 个线程访问 stride=2（每隔一个 float）的 Shared Memory 数组会发生几路冲突？

<details>
<summary>答案</summary>

Shared Memory 有 32 个 Bank，每个 Bank 宽 4 字节。同一 Warp 的 32 个线程如果多个线程访问同一个 Bank 的不同地址 → 请求被串行化 → 这就是 Bank Conflict。

stride=2 访问：
- Thread 0 → Bank[0]，Thread 1 → Bank[2]，Thread 2 → Bank[4]，...
- Thread 16 → Bank[32] = Bank[0]（32 Bank 循环）
- Thread 16 和 Thread 0 冲突！同样，Thread 17 和 Thread 1 冲突
- → **2 路冲突**（每个 Bank 被 2 个线程访问）

公式：冲突度 = 32 / gcd(32, stride)
- stride=1: 32/gcd(32,1)=32/1=32 → **无冲突**（每 Bank 1 线程）
- stride=2: 32/gcd(32,2)=32/2=16 → 2 路冲突
- stride=32: 32/gcd(32,32)=32/32=1 → 32 路冲突（所有线程抢同一个 Bank！）
</details>

**Q5.2** stride=33 访问 Shared Memory，冲突度是多少？这和直觉有什么不同？

<details>
<summary>答案</summary>

stride=33：32 / gcd(32, 33) = 32 / 1 = 32 → **无冲突**！

因为 33 和 32 互质（gcd=1），33 % 32 = 1，所以实际 Bank 模式等同于 stride=1（无冲突）。这和直觉相反——看起来"跨步很大"应该很不连续，但对 Bank 来说是循环的（32 个 Bank），跨 33 和跨 1 的 Bank 模式一样。

这就是为什么有时故意加 padding（如 `[32][33]` 代替 `[32][32]`）来消除 Bank Conflict。
</details>

---

## 06 coalescing — 合并访问

**Q6.1** 矩阵以 row-major 存储，kernel 中让 `threadIdx.x` 对应矩阵的行维度（M），问 HBM 访问效率如何？

```cuda
int row = threadIdx.x;  // 相邻线程对应相邻行!
int col = blockIdx.x;
output[row * N + col] = input[row * N + col];
```

<details>
<summary>答案</summary>

效率极差（~3%）。

- Thread 0 读 `input[0*N + col]`（地址 = base + col*4）
- Thread 1 读 `input[1*N + col]`（地址 = base + (N+col)*4）
- → 地址间隔 N×4 字节

同一 Warp 的 32 个线程跨了 32 个不同的 Cache Line → 32 次内存事务 → 4096B 传输只用到 128B → 效率 3%。

正确做法：`threadIdx.x` 对应列维度（N），这样相邻线程的地址间隔 4 字节，全部在同一 Cache Line 内。
</details>

**Q6.2** AoS (Array of Structures) 和 SoA (Structure of Arrays) 有什么区别？为什么 GPU 上 SoA 更好？

<details>
<summary>答案</summary>

```c
// AoS: 每个粒子的 (x,y,z) 连续存储
struct Particle { float x, y, z; };
Particle particles[N];  // [x0,y0,z0, x1,y1,z1, ...]

// SoA: 所有 x 连续，所有 y 连续，所有 z 连续
float x[N], y[N], z[N];  // [x0,x1,..., y0,y1,..., z0,z1,...]
```

只读 x 坐标时：
- **AoS**：Thread i 读 `particles[i].x` → 地址间隔 12B → 32 线程跨 3 个 Cache Line → 效率 ~33%。而且 y 和 z 也被传了但没用到。
- **SoA**：Thread i 读 `x[i]` → 地址连续 → 1 个 Cache Line → 效率 100%。

GPU 上 SoA 更好的原因是**合并访问**——连续线程需要访问连续的内存地址。
</details>

---

## 07 softmax — Softmax 三版本

**Q7.1** 为什么要先减去 max，不能直接算 `exp(x_i) / sum(exp(x))`？

<details>
<summary>答案</summary>

数值稳定性。`exp(89)` 就已经超过 float32 的最大值（~3.4×10³⁸），结果变成 +Inf。`+Inf / +Inf = NaN`，整个输出报废。

减 max 后：最大项变成 `exp(0) = 1`，其他项 ≤ 1，永远不会溢出。数学上等价（分子分母同时乘了 `exp(-max)`）。

这不是优化，是必须。
</details>

**Q7.2** Online Softmax 中，当 max 更新时，旧 sum 怎么修正？写出公式并解释。

<details>
<summary>答案</summary>

```cuda
local_sum = local_sum * expf(old_max - local_max) + expf(val - local_max);
```

修正因子 `exp(old_max - new_max)`：
- 旧 sum 的每一项 `exp(x_i - old_max)` 需要变成 `exp(x_i - new_max)`
- 即乘以 `exp(old_max - new_max)`（因为 `exp(x_i - new_max) = exp(x_i - old_max) × exp(old_max - new_max)`）
- 所有旧项都乘以同一个因子 → 整个 sum 乘一次即可

注意方向：`old_max - new_max ≤ 0`（max 只增不减），所以因子 ≤ 1，不会溢出。
</details>

**Q7.3** V3 (Warp-level) 为什么只适用于 N ≤ 32？N=64 怎么办？

<details>
<summary>答案</summary>

V3 中每个 lane 持有一行的一个元素，一个 Warp 只有 32 个线程，所以一行最多 32 个元素。

N=64 时的修改：每个线程处理 `ceil(64/32) = 2` 个元素。用 Grid-Stride Loop 或让每线程持有 2 个元素依次处理。但这样就退化成和 V1/V2 类似的模式了，只不过仍然零 Shared Memory：

```cuda
float val1 = (lane < cols) ? x[lane] : -INFINITY;
float val2 = (lane + 32 < cols) ? x[lane + 32] : -INFINITY;
float m1 = warp_reduce_max(val1);
float m2 = warp_reduce_max(val2);
float m = fmaxf(m1, m2);
```
</details>

---

## 08 ncu_profiling — 性能分析

**Q8.1** ncu 报告中 SOL Memory% = 25%, SOL Compute% = 20%。这说明什么？瓶颈在哪？

<details>
<summary>答案</summary>

两者都低（< 30%），不是典型的 Memory Bound 也不是 Compute Bound。可能是**延迟瓶颈（Latency Bound）**或**发射瓶颈（Issue Bound）**。

需要看 Warp Stall Reasons：
- **Long Scoreboard** 主导 → Memory Latency Bound（等数据但没占满带宽）→ 增加 Occupancy 或用 ILP 让更多请求同时在飞
- **Short Scoreboard / Not Selected** 主导 → 计算依赖链太长或调度拥塞 → 打破依赖链（ILP）或减少寄存器压力
- **Barrier** 主导 → `__syncthreads()` 开销大 → 用 Shuffle 替代或调整 Block 大小
- **MIO Throttle** → 内存请求队列满了 → 减少同时在飞的内存请求
</details>

**Q8.2** ncu 的 `--set full` vs `--set basic` 有什么区别？日常开发用哪个？

<details>
<summary>答案</summary>

- `--set basic`（默认）：~5 遍重放。覆盖 SOL、Occupancy、基本内存指标。适合日常开发快速定位瓶颈类型。
- `--set full`：~10 遍重放。额外覆盖详细内存分析、Warp State、Scheduler、指令统计。适合深入分析"为什么慢"。

日常开发：先用 `--set basic` 看 SOL 决定是 Memory/Compute Bound，如果不清楚再看 `--set full`。
</details>

---

## 09 register_tiling — GEMM 寄存器分块

**Q9.1** 什么叫 Register Blocking（寄存器分块）？为什么比只用 Shared Memory Tiling 更快？

<details>
<summary>答案</summary>

Shared Memory Tiling 中，每个线程计算 C 的 1 个元素，内循环每轮从 SMEM 读 1 个 A + 1 个 B → 2 次 LDS 指令 → 1 次 FMA。LDS:FMA = 2:1。

Register Blocking：每个线程计算 C 的一个小 tile（如 4×4），把 A 和 B 的一部分存在寄存器中。内循环每轮从 SMEM 读 1 个 A + 1 个 B → 2 次 LDS，但可以做 4×4 = 16 次 FMA。LDS:FMA = 2:16 = 1:8。

大幅减少 Shared Memory 读取次数 → 更多时间花在计算上。

代价：消耗更多寄存器（4×4 tile → ~32+ 寄存器/线程）。如果寄存器不够 → 寄存器溢出到 L1 → 反而变慢。
</details>

---

## 10 fused_kernel — 算子融合

**Q10.1** 为什么把两个 Memory Bound kernel 融合成一个能显著加速？

<details>
<summary>答案</summary>

两个 Memory Bound kernel 各跑一次：
```
Kernel 1: 读 A（N次），写 B（N次）       → 2N 显存操作
Kernel 2: 读 B（N次），写 C（N次）       → 2N 显存操作
总计: 4N 显存操作
```

融合成一个 kernel：
```
Fused: 读 A（N次），计算，写 C（N次）     → 2N 显存操作
```

省掉了一半的显存操作（中间的 B 不需要写回 HBM，直接在寄存器/Shared Memory 中传递给下一步）。

对于 Memory Bound 算子（算术强度低），减少显存操作 = 直接加速。融合节省的带宽可以达到 1.5-2× 的端到端加速。
</details>

**Q10.2** 使用 `float4` 向量化加载的前提条件是什么？

<details>
<summary>答案</summary>

1. **地址 16 字节对齐**：`float4 = 16 bytes`，`LDG.128` 要求 16B 对齐。`cudaMalloc` 默认 256B 对齐，满足条件。如果指针偏移不是 4 的倍数，需要先处理不对齐的头部。
2. **N 是 4 的倍数**（或处理尾部）：否则会越界读。
3. **指针 cast**：`reinterpret_cast<const float4*>(ptr)`。

不对齐的后果：`LDG.128` 变成多次 `LDG.32` → 向量化的好处全没了。
</details>

---

## 11 warp_divergence — Warp 分歧

**Q11.1** 同一 Warp 内有 16 个线程走 if 分支、16 个线程走 else 分支。性能损失是多少？

<details>
<summary>答案</summary>

理想情况（无分歧）：32 线程同时完成 → 1 拍。
有分歧：Warp 先执行 if 分支（16 线程活跃，16 线程 masked off），再执行 else 分支（反过来）→ 2 拍。

理论上 50% 效率损失。但实际上：
- 两个分支的指令数不同，取决于更长的那个
- 分支后通常会"收敛"回来（Warp 重新汇聚）
- 编译器可能用 `predicated execution` 优化简单 if-else（两条指令都发射但只写回一条的结果）

真正严重的是循环次数不同的分歧：`for (int i = 0; i < thread_dependent_N; i++)` → 不同线程退出循环的时间不同 → 早退出的被 stall → 最慢的线程决定整体时间。
</details>

**Q11.2** 什么情况下 if-else 分支**不**会导致性能损失？

<details>
<summary>答案</summary>

当分支条件和 threadIdx 的 Warp 边界对齐时。即同一 Warp 内所有 32 个线程走同一个分支。

例如：`if (threadIdx.x / 32 == 0)` → 第一个 Warp 全部走 true，其他 Warp 全部走 false → 零分歧。

编译器优化：简单 if-else 可能被编译为 `predicated execution`。例如 `val = (cond) ? a : b` 可能编译为两条指令都执行，用 `@P0` 条件寄存器决定哪条写回 → 实际效果和分歧类似但无 warp 串行化。
</details>

---

## 12 layernorm_project — 综合实战

**Q12.1** LayerNorm 中 `mean` 和 `rstd` 为什么必须保存起来给 backward 用？

<details>
<summary>答案</summary>

Backward 公式：
```
dx[i] = rstd * gamma[i] * (dy[i] - mean_dy - xhat[i] * mean_dy_xhat)
```

需要 `rstd` 和能重算的 `xhat[i] = (x[i] - mean) * rstd`。如果不保存 `mean` 和 `rstd`：
- 需要从 x 重新计算 mean 和 var → 一次额外的数据遍历（Memory Bound → 慢）
- 或者保存整个 y（N 个 float），但 mean/rstd 只有 1 个 float/行

保存 mean 和 rstd（每行 2 个 float）比保存 y（每行 cols 个 float）显存占用小得多，也比重算快。这是典型的"save for backward"权衡。
</details>

**Q12.2** LayerNorm 的 backward 中，dgamma 和 dbeta 为什么要用 `atomicAdd`？有没有不用 atomicAdd 的方案？

<details>
<summary>答案</summary>

```
dgamma[i] = Σ_rows dy[row][i] * xhat[row][i]
```

gamma 是 `[cols]`，但输入是 `[rows, cols]`。每行（一个 Block）都贡献对 `dgamma[i]` 的梯度。多个 Block 同时写同一个 `dgamma[i]` → 数据竞争 → 需要 `atomicAdd`。

不用 atomicAdd 的方案（两阶段归约）：
1. 每个 Block 先在 Shared Memory 中归约本 Block 对 dgamma 的贡献
2. 然后用一次 `atomicAdd` 累加（而不是每行每列都 atomicAdd）

这样 atomicAdd 次数从 `rows × cols` 降为 `gridDim × cols`，大幅减少原子操作开销。但代码复杂度增加。对于 LayerNorm 这种 Memory Bound 算子，atomicAdd 的额外开销不大（< 5%）。
</details>

**Q12.3** 为什么 LayerNorm 每个 Block 处理一行（`<<<rows, 256>>>`），而不是用多个 Block 处理同一行？

<details>
<summary>答案</summary>

因为每行的 mean 和 variance 需要看到该行的**所有元素**。如果多个 Block 处理同一行：
- 每个 Block 只能看到该行的一部分
- 需要跨 Block 通信才能得到整行的 mean 和 var
- GPU 上没有跨 Block 的高效通信机制（`cooperative_groups::grid_group` 有但开销大）

一个 Block 处理一行：Block 内通过 Shared Memory + Shuffle 高效通信。这种设计利用了"行之间天然并行、行内需要通信"的特性。

如果 cols 非常大（如 > 4096），单 Block 256 线程不够，可以用 cooperative groups 或改成无归一化统计量的 LayerNorm 变体。
</details>

---

## 13 flash_attention — 简化版 FlashAttention

**Q13.1** 标准 Attention 的显存瓶颈在哪？FlashAttention 怎么解决？

<details>
<summary>答案</summary>

标准 Attention：
```
S = Q × K^T          [N, N] ← N×N 的中间矩阵，N=2048 时 = 16MB FP32
P = softmax(S)       [N, N] ← 又一个 N×N
O = P × V            [N, d]
```

N×N 矩阵的显存和带宽是瓶颈：N=2048 时 S 和 P 各 16MB，需要从 HBM 写→读，对于长序列 N=8192 → 256MB。

FlashAttention：把 Q、K、V 分块，逐块计算 softmax 的 online 版本，不把完整的 S 和 P 写到 HBM。数据一直在 SRAM（Shared Memory）中处理完才写回。

核心技术：Tiling（分块）+ Online Softmax（逐块更新 max/sum）+ Recomputation（反向传播时重算中间值而非从 HBM 读）。
</details>

---

## 14 im2col_conv — im2col 卷积

**Q14.1** im2col 把输入 `[C, H, W]` 展开成 `[H_out×W_out, C×Kh×Kw]`。如果 C=64, H=W=224, Kh=Kw=3, pad=1，列矩阵有多大？这合理吗？

<details>
<summary>答案</summary>

```
H_out = 224 + 2×1 - 3 + 1 = 224
W_out = 224
总元素 = 224 × 224 × 64 × 3 × 3 = 28,901,376 float ≈ 110 MB
输入 = 64 × 224 × 224 = 3,211,264 float ≈ 12.3 MB
膨胀率 ≈ 9×
```

这其实很大（110MB vs 输入 12.3MB）。对于大图像，im2col 的额外内存成为负担。这也是为什么 cuDNN 对不同的卷积配置用不同的实现（Winograd、FFT、直接卷积）——不是所有情况都适合 im2col。
</details>

---

## 15 wmma_gemm — Tensor Core WMMA

**Q15.1** WMMA 的 A 矩阵是 row_major，B 矩阵必须是 col_major。如果你的 B 是 row_major 存储的，怎么处理？

<details>
<summary>答案</summary>

选项 1：**Host 端转置 B**（简单）。在 host 端把 B 转置存储，GPU 上直接当 col_major 用。代价：转置本身有开销，但对推理来说只做一次。

选项 2：**调整 load 参数**（需要理解 stride）。`load_matrix_sync` 的 leading dimension 参数可以处理 stride 不等于矩阵宽度的情况。如果 B 是 row_major `[N, K]`，可以声明 B fragment 为 row_major 然后 `load_matrix_sync(b_frag, B + warpN*16, K)` 从转置视角加载。

选项 3：**直接在 kernel 里用 Shared Memory 转置**。先 `load_matrix_sync` 到 Shared Memory，转置，再让 Tensor Core 以 col_major 视角读。这是 cuBLAS 的做法。
</details>

**Q15.2** 为什么 WMMA GEMM 用 FP16 输入 + FP32 累加，而不是全 FP16？

<details>
<summary>答案</summary>

FP16 尾数只有 10 bit（~3 位十进制有效数字）。如果累加器也用 FP16：
- 每次 `sum += a*b` 时，FP16 只有 10 bit 精度
- 累加 16 次 → 小项可能被舍入到 0（如 `1.0 + 0.0001 = 1.0` in FP16）

Tensor Core 硬件设计：乘法 FP16×FP16 → 结果自动提升为 FP32，累加也用 FP32。最后写回才可选 FP16 或 FP32。

这实现了"混合精度"的关键好处：输入用 FP16（省带宽）、计算内部用 FP32（保精度）。
</details>

---

## 16 streams — CUDA Stream

**Q16.1** 下面代码中，Kernel A 和 Kernel B 是并行执行的吗？

```cuda
kernel_A<<<grid, block>>>(d_a);
kernel_B<<<grid, block>>>(d_b);
```

<details>
<summary>答案</summary>

**不是**。没有指定 stream，都默认使用 stream 0（default stream）。default stream 是同步的——后面的 kernel 必须等前面的 kernel 完成。

要并行：
```cuda
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);
kernel_A<<<grid, block, 0, s1>>>(d_a);
kernel_B<<<grid, block, 0, s2>>>(d_b);
```

但实际是否真的并行取决于：SM 资源是否够同时跑两个 kernel、两个 kernel 各自的资源需求、kernel 是否够大值得并行。
</details>

**Q16.2** Pinned Memory (page-locked) 和普通 Pageable Memory 在 `cudaMemcpy` 中有什么区别？

<details>
<summary>答案</summary>

普通 Pageable Memory（`malloc`）：OS 可能随时把页面换出到磁盘。cudaMemcpy 时驱动需要先 pin 住这些页面，DMA 引擎才能直接访问 → 多了一次拷贝（先到 staging buffer）。

Pinned Memory（`cudaMallocHost`）：页面被锁在物理内存中，DMA 可以直接访问 → 省了一次拷贝 → 带宽约 2×。

代价：Pinned Memory 不能用太多，会挤占 OS 可用物理内存，导致系统变慢。

最佳实践：用 Pinned Memory 做 streaming 的 staging buffer（要反复传输的数据），普通数据用 Pageable Memory。
</details>

---

## 17 mixed_precision — 混合精度

**Q17.1** BF16 vs FP16：为什么 BF16 不需要 Loss Scaling 而 FP16 需要？

<details>
<summary>答案</summary>

关键是**指数范围**：

- FP16：5 bit 指数 → 范围 ±65504。梯度 < 2⁻¹⁴ ≈ 6e-5 称为"下溢"（underflow），小于最小正数的值变成 0。很多梯度确实很小（如 1e-7）→ 被截断为 0 → Loss Scaling 把梯度放大到 FP16 能表示的范围。

- BF16：8 bit 指数（和 FP32 一样）→ 范围 ±3.4×10³⁸。小梯度如 1e-7 完全在范围内（BF16 最小正数 ≈ 1.2e-38）。不需要 Loss Scaling。

代价：BF16 尾数只有 7 bit（vs FP16 的 10 bit）→ 精度更差。但训练中精度损失的负面影响远小于梯度下溢，所以 Ampere+ 上 BF16 更受欢迎。

BF16 和 FP32 互转：截断 FP32 的高 16 位即可 → 硬件成本极低。
</details>

**Q17.2** 混合精度训练中，"Master Weights"是什么？为什么要有它？

<details>
<summary>答案</summary>

Master Weights = 优化器维护的 FP32 全精度权重副本。

```
前向/反向：用 FP16 权重 → 快、省显存
梯度：FP16 格式（Loss Scaling 后）
更新：梯度转 FP32，更新 FP32 Master Weights
下一次迭代：FP32 Master Weights 转 FP16 → 前向
```

为什么要有 Master Weights？
- FP16 精度不够累加微小梯度更新：`weight_fp16 += lr * grad`，如果 `lr * grad` < weight_fp16 的最低有效位 → 更新丢失
- FP32 Master Weights 保留全精度，确保小的更新不会丢失

这就是"混合精度"的含义：关键状态用 FP32 保精度，计算用 FP16 加速。
</details>

---

## 跨章节综合题

**Z.1** 一个 kernel 在 ncu 中显示 Memory Bound（SOL Memory% = 80%），你已经做了 float4 向量化。还有哪些手段可以继续优化？

<details>
<summary>答案</summary>

1. **Shared Memory Tiling**（减少同一数据的重复 HBM 读）
2. **增大 blockDim**（更多 Warp 隐藏延迟 → 让内存请求队列保持满）
3. **算子融合**（消除中间 tensor 的读写）
4. **换 FP16**（数据量减半，带宽需求减半）
5. **使用 Read-Only Cache（__ldg）** 替代普通 LDG（如果数据是只读的）
6. **调整 grid size**（让所有 SM 同时忙碌）
7. **cp.async**（异步拷贝，让计算和访存重叠）
8. **检查是否可以用 Tensor Core**（如果涉及矩阵乘，直接跳到 312 TFLOPS）

优化的天花板是 Roofline 模型——如果已经是 Memory Bound，最终能加速的上限 = 当前带宽 / 峰值带宽。如果已到 80%，最多还有 ~25% 的带宽优化空间。
</details>

**Z.2** 设计一个完整的 CUDA 算子开发 + 验证流程。

<details>
<summary>答案</summary>

1. **CPU 参考实现**：用最简单的循环实现，验证正确性
2. **朴素 CUDA 版**：1 线程 1 输出，不优化
3. **正确性验证**：对比 CPU 和 GPU 输出（max error < 1e-5）
4. **性能基线**：cudaEvent 计时 → 计算有效带宽/GFLOPS
5. **ncu 分析**：`--set basic` → 确定瓶颈类型（Memory/Compute Bound）
6. **针对性优化**：
   - Memory Bound → 合并访问 + SMEM tiling + 向量化 + 融合
   - Compute Bound → ILP + 寄存器 tiling + Tensor Core
   - Latency Bound → 提高 Occupancy + 更多 Warp
7. **每次修改后重新验证正确性**（compute-sanitizer）
8. **对比 PyTorch 原生实现**（如果适用）→ 你的算子应该比原生快或至少不慢
9. **梯度检查**（如果是训练算子）→ `torch.autograd.gradcheck`
10. **大矩阵测试**（小矩阵 overhead 大，掩盖真实性能）
</details>

**Z.3** 一个矩阵乘法 kernel，M=N=K=1024，为什么 FP32 Tiled CUDA Core 版本跑出 200 GFLOPS（A100 FP32 peak = 19.5 TFLOPS），而 FP16 WMMA 版本能跑到 8 TFLOPS？

<details>
<summary>答案</summary>

FP32 CUDA Core (200 GFLOPS = 1% peak)：
- 主要原因：1024×1024 矩阵的 GEMM 算术强度约 170 FLOP/Byte，这在 A100 Ridge Point (~9.7) 的右边 → 理论上是 Compute Bound。但 1% peak 说明代码有问题：
- 可能原因：没有 Shared Memory tiling → 每个元素被从 HBM 读了 1024 次 → 变成了 Memory Bound → 带宽限制了性能。
- 即使加了 Tiling，CUDA Core 本身的峰值也只有 19.5 TFLOPS（所有 64 个 CUDA Core/SM 同时跑 FMA）。
- 实际能跑到 ~10-15 TFLOPS 已经很好了。

FP16 WMMA (8 TFLOPS = ~2.6% of 312 TFLOPS FP16 peak)：
- Tensor Core 专用硬件，单指令 4096 FLOP → 计算不再是瓶颈。
- 但这个版本直接从 HBM 加载（无 SMEM 预取）→ 仍然受到带宽限制。
- 加上 Shared Memory staging + double buffering → 可接近 peak。
- CUTLASS 能做到 > 80% peak → ~250 TFLOPS。
</details>
