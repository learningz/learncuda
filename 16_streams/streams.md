# CUDA Stream 与异步执行：GPU 并行的第二层含义

配合 `streams.cu` 阅读。


## "异步"到底是什么意思

```
CUDA 的异步有两个层面:

层面 1: CPU 和 GPU 异步 (你已经知道的)
  kernel<<<...>>>() 调用立即返回给 CPU，GPU 在后台执行。
  CPU 可以继续做其他事情（准备下一批数据、处理逻辑等）。
  只有 cudaDeviceSynchronize() 才会让 CPU 等 GPU 完成。

层面 2: GPU 内部不同引擎之间异步 (本课的重点)
  GPU 有 3 个独立的硬件引擎:
  
  ┌────────────────────────────────────────────────────┐
  │  Copy Engine (H2D)  │  用于 Host → Device 传输     │
  │  Copy Engine (D2H)  │  用于 Device → Host 传输     │
  │  Compute Engine     │  用于 kernel 执行             │
  └────────────────────────────────────────────────────┘
  
  这三个引擎可以同时工作!
  但默认情况下（单 Stream），所有操作排成一队串行执行，浪费了并行能力。
  多 Stream 就是让不同引擎同时忙起来。
```


## Stream 是什么

```
Stream = GPU 上的一个操作队列 (FIFO)

同一 Stream 内: 操作严格按提交顺序执行 (保证依赖关系)
不同 Stream 间: 操作可以并行执行 (没有顺序保证)

类比:
  单 Stream = 一条车道，所有车排队通过
  多 Stream = 多条车道，不同车道的车可以并行通过
  
  但同一条车道内的车必须保持前后顺序 (不能超车)
```


## 为什么多 Stream 能加速

```
单 Stream 的时间线:
  时间 → ─────────────────────────────────────────────
  Copy H2D:  [===chunk 0===][===chunk 1===]
  Compute:                                  [===chunk 0===][===chunk 1===]
  Copy D2H:                                                                [===0===][===1===]
  总时间 = T_H2D + T_compute + T_D2H

2 Stream (分块传输) 的时间线:
  时间 → ─────────────────────────────────────────────
  Copy H2D:  [==chunk 0==][==chunk 1==]
  Compute:                [==chunk 0==][==chunk 1==]    ← 和 H2D_1 重叠!
  Copy D2H:                            [==chunk 0==][==chunk 1==]
  总时间 ≈ T_H2D/2 + T_compute + T_D2H/2  (传输被部分遮盖)

4 Stream 的时间线:
  时间 → ─────────────────────────────────────────────
  Copy H2D:  [=0=][=1=][=2=][=3=]
  Compute:       [=0=][=1=][=2=][=3=]     ← 几乎完全重叠!
  Copy D2H:          [=0=][=1=][=2=][=3=]
  总时间 ≈ max(T_compute, T_transfer)  (传输几乎完全被遮盖)

极限:
  当 Stream 数量足够多时，传输时间被完全隐藏在计算时间内。
  总时间趋近于 max(T_compute, T_total_transfer)。
  继续增加 Stream 没有收益（硬件引擎数量有限，通常 H2D 和 D2H 各 1 个）。
```


## Pinned Memory 为什么重要

```
想让 Host↔Device 传输稳定地实现真正异步、并和 kernel 重叠，
最常见也最可靠的前提是使用 pinned host memory。
GPU 的 DMA 引擎需要直接访问 Host 内存的物理地址。

Pageable Memory (普通 malloc):
  OS 随时可能把这块内存交换到磁盘 (page fault)
  → 物理地址不稳定
  → DMA 不能直接长期依赖这块内存
  → CUDA 驱动通常会先拷到内部的 staging buffer (pinned)
  → 因而 `cudaMemcpyAsync` 往往会退化或至少失去理想的重叠效果
  
  路径: malloc buffer → [CPU 拷贝] → staging buffer → [DMA] → GPU
                         ↑ 这一步阻塞了!

Pinned Memory (cudaMallocHost):
  OS 保证这块内存锁定在物理内存中 (永不 page out)
  → 物理地址稳定
  → DMA 可以直接传输
  → 无需 staging buffer，可以真正异步
  
  路径: pinned buffer → [DMA 直传] → GPU
                         ↑ 异步! CPU 立即返回

典型加速: Pinned 比 Pageable 快 1.5-2x (因为省了一次拷贝)

注意事项:
  - Pinned Memory 会占用物理 RAM，不能被 OS 回收
  - 分配过多会导致系统内存紧张
  - 建议: 只对频繁传输的 buffer 使用 (如训练的 data loader)
  - cudaMallocHost 分配，cudaFreeHost 释放 (不是 free!)
```


## cudaEvent 计时的工作原理

```
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);   // 在 stream 时间线上插入一个"时间戳"
my_kernel<<<...>>>(...);
cudaEventRecord(stop, stream);    // 在 kernel 完成后插入另一个"时间戳"

cudaEventSynchronize(stop);       // CPU 等待 stop 事件完成
cudaEventElapsedTime(&ms, start, stop);  // 计算两个时间戳之间的 GPU 时间

为什么比 CPU 计时 (clock_gettime / std::chrono) 更准确?
  
  CPU 计时: 包含了 kernel launch 开销 (~5μs) + CPU-GPU 同步延迟
  GPU Event: 直接在 GPU 时间线上打点，精度 ~0.5μs，不受 CPU 影响
  
  对于短 kernel (< 100μs)，CPU 计时的误差可能比 kernel 本身还大!
  → 永远用 cudaEvent 计时 CUDA 程序。
```


## 在深度学习框架中的应用

```
PyTorch 内部大量使用多 Stream:
  - 默认计算 Stream: 所有 forward/backward kernel
  - 默认拷贝 Stream: DataLoader 的 pin_memory=True + 异步传输
  - NCCL Stream: 多 GPU 通信 (AllReduce 梯度)

典型训练循环的 Stream 重叠:
  Compute Stream:  [forward_batch_N ][backward_batch_N ][optimizer_step_N ]
  Copy Stream:     [                 ][load_batch_N+1   ][                  ]
  NCCL Stream:     [                 ][allreduce_grad_N ][                  ]
  
  → 数据加载、梯度通信都和计算重叠，总时间只取决于最慢的那个

PyTorch DataLoader 优化:
  DataLoader(pin_memory=True, num_workers=4)
  
  pin_memory=True:
    DataLoader 把数据 batch 放在 pinned memory 中
    → to('cuda') 时可以用异步传输
    → 传输和上一个 batch 的计算重叠
  
  如果不开 pin_memory:
    每次 to('cuda') 都要先拷到 staging buffer → 更慢且不能重叠
```


## 练习题

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_dual_stream_level1.cu](./exercises/ex1_dual_stream_level1.cu) | 双 Stream 流水线 | 手写 cudaMemcpyAsync + 多 Stream launch（只填 host 端） |

```bash
nvcc -O2 -o ex1_dual_stream_level1 ex1_dual_stream_level1.cu
```
