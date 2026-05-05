# CUDA Stream 与异步执行：GPU 并行的第二层含义

配合 `streams.cu` 阅读。

**难度**: ⭐⭐ 进阶
**前置知识**: 基础 CUDA kernel 编写（[tutorial Part 1](../tutorial.md#part-1-从-cpu-到-gpu--你的第一个-cuda-程序)）
**读完你能做什么**: 理解 CUDA Stream 的异步执行模型，能用多 Stream 实现数据传输与计算重叠


## 什么是 CUDA Stream

### 之前的所有程序都在"排队"

回顾你写过的每个 CUDA 程序的流程：

```
cudaMemcpy(d_a, h_a, ...);    // 1. 传数据到 GPU    → 等传完...
cudaMemcpy(d_b, h_b, ...);    // 2. 传更多数据      → 等传完...
kernel<<<...>>>(d_a, d_b, ...); // 3. GPU 计算       → 等算完...
cudaMemcpy(h_c, d_c, ...);    // 4. 传结果回 CPU    → 等传完...
```

所有操作严格排队，一个做完下一个才开始。
但 GPU 有多个独立的硬件引擎——它们其实可以**同时工作**！

```
GPU 的 3 个独立引擎:

  ┌──────────────────┐  可以同时运行!
  │ Copy Engine (H2D)│  负责 CPU → GPU 传输
  │ Copy Engine (D2H)│  负责 GPU → CPU 传输
  │ Compute Engine   │  负责 kernel 计算
  └──────────────────┘

单 Stream (默认):
  时间 →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→
  Copy H2D:  [=====]
  Compute:          [===========]
  Copy D2H:                      [=====]
  总时间 = T_传 + T_算 + T_传
  → Copy 和 Compute 引擎大部分时间都在闲着!
```


### Stream = 一个操作队列

Stream 是 CUDA 的"工作流"概念：

```
同一 Stream 内: 操作按提交顺序串行执行 (先进先出)
不同 Stream 间: 操作可以并行执行!
```

用两个 Stream 处理两批数据：

```
Stream A: 传数据 A → 计算 A → 取结果 A
Stream B: 传数据 B → 计算 B → 取结果 B

单 Stream (全部排队):
  [传A][===算A===][取A][传B][===算B===][取B]
  总时间: 2 × (T_传 + T_算 + T_传)

多 Stream (重叠执行):
  Stream A:  [传A][===算A===][取A]
  Stream B:       [传B][===算B===][取B]
  
  Copy 引擎:  [传A][传B]          [取A][取B]
  Compute:         [===算A===][===算B===]
  
  传 B 和 算 A 同时进行! 取 A 和 算 B 同时进行!
  总时间 ≈ T_传 + T_算 + T_算 + T_传 (而不是 2 倍)
  → 省掉了等待时间!
```

> **常见陷阱: Default Stream (Stream 0) 的隐式同步**
>
> 如果不显式创建 Stream，所有操作都在 **Default Stream** (stream 0) 中执行。
> Legacy default stream 有一个坑人的特性：**它会隐式同步所有其他 Stream**。
>
> ```
> // 这样写没有 overlap!
> kernel<<<grid, block>>>(d_a);          // 在 default stream
> cudaStream_t s;
> cudaStreamCreate(&s);
> kernel<<<grid, block, 0, s>>>(d_b);    // 在 stream s
> // → default stream 会等 s 完成，s 也会等 default stream!
> // → 两边的 kernel 串行执行，完全没有重叠!
> ```
>
> **解决方法**：要么所有操作都用显式 Stream（推荐），要么编译时加 `--default-stream per-thread` 让 default stream 不再隐式同步。
>
> 详细机制见 `theory/02_cuda_programming_model.md` §2.4。


### Pinned Memory：让传输本身更快

除了重叠执行，传输速度本身也有优化空间：

```
普通 malloc 分配的内存 (Pageable Memory):
  cudaMemcpy 的实际过程:
    1. CPU 先把数据从你的内存复制到一块临时的 pinned 缓冲区
    2. GPU 的 DMA 引擎从 pinned 缓冲区搬到显存
    → 2 次搬运!

Pinned Memory (cudaMallocHost 分配):
  cudaMemcpy 的过程:
    1. GPU 的 DMA 引擎直接从你的内存搬到显存
    → 只需 1 次搬运! 快 1.5-2 倍!
  
  而且: 只有 Pinned Memory 才能和 kernel 真正重叠执行!
  普通内存的 cudaMemcpy 会在内部同步, 无法异步。

**关键陷阱: cudaMemcpyAsync + Pageable Memory 的实际行为**:

```
cudaMemcpyAsync 的名字有 "Async", 但如果你传了 pageable memory:
  → CUDA 驱动会退化为同步行为!
  → CPU 线程被阻塞, 直到整个拷贝完成才返回!
  → 即使你用了显式 Stream, 重叠效果也完全消失!

为什么? GPU 的 DMA 引擎只能访问物理地址固定的内存。
Pageable memory 的物理页可能随时被 OS 换出 → DMA 不能直接用它。
所以驱动会:
  1. 先在内部分配一块临时的 pinned buffer
  2. CPU 把数据拷贝到 pinned buffer (这一步阻塞了 CPU!)
  3. DMA 从 pinned buffer 传到 GPU
  → 第 2 步让"Async"名存实亡!

验证方法: 
  cudaMemcpyAsync(dst, pageable_src, size, cudaMemcpyHostToDevice, stream);
  // 如果这是真正的异步, CPU 会立即执行下一行
  // 如果用了 pageable memory, CPU 会在这里卡住!
  
正确做法:
  float *pinned_data;
  cudaMallocHost(&pinned_data, size);  // 必须是 cudaMallocHost!
  cudaMemcpyAsync(dst, pinned_data, size, cudaMemcpyHostToDevice, stream);
  // 现在 CPU 立即返回 → 真正的异步!
```
```

这就是为什么 PyTorch DataLoader 的 `pin_memory=True` 很重要：
它让数据传输和模型计算重叠，隐藏传输延迟。


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


## Stream 同步原语 — 完整的工具箱

除了 `cudaDeviceSynchronize()` (等待所有 Stream), 还有更精细的控制:

```
cudaStreamSynchronize(stream):
  等待指定 Stream 的所有操作完成。
  比 cudaDeviceSynchronize() 更轻量 — 只等一个 Stream, 不影响其他。
  
  用法: 确保某个 Stream 的数据传输完成后再读取结果。

cudaStreamWaitEvent(stream, event, flags):
  让 Stream A 等待 Stream B 中的某个事件完成。
  → 实现跨 Stream 的依赖关系!
  
  例: Stream A 算完后, Stream B 才能开始用 Stream A 的结果:
    cudaEventRecord(event, stream_A);           // 在 Stream A 中打点
    cudaStreamWaitEvent(stream_B, event, 0);    // Stream B 在此等待
    kernel<<<..., stream_B>>>(...);              // event 完成后才执行

cudaStreamQuery(stream):
  非阻塞检查: Stream 完成了没有?
  返回 cudaSuccess (已完成) 或 cudaErrorNotReady (还在跑)。
  → 用于轮询, 不需要阻塞 CPU。

完整的依赖管理:
  Stream A: [H2D chunk 0] → [compute chunk 0] → [D2H chunk 0]
                              ↓ (event)
  Stream B:                  [wait event] → [compute chunk 1] → [D2H chunk 1]
  
  → Stream B 的 compute 必须等 Stream A 的 compute 完成
  → 但 H2D 可以并行!
```

### PCIe 双向带宽的物理限制

```
上面的时间线图显示 H2D 和 D2H 完美并行, 但这只在 NVLink 系统上成立!

PCIe 系统 (绝大多数用户的场景):
  H2D 和 D2H 共享同一条 PCIe 总线 → 双向传输时带宽各减半!
  
  例: PCIe 4.0 x16 = ~32 GB/s 单向, 但如果同时 H2D + D2H:
    每个方向只能用 ~16 GB/s → 不是真正的"并行", 而是带宽分割!
  
  NVLink 系统 (DGX, HGX):
    有独立的双向链路 → H2D 和 D2H 可以同时跑满带宽。
  
  实际影响:
    如果你的 workload 同时有大量 H2D 和 D2H, 在 PCIe 系统上:
    总时间 > max(T_H2D, T_D2H), 因为带宽被分割了。
```

### Hyper-Q 和多 CPU 线程提交

```
现代 GPU (Kepler+) 有 Hyper-Q 技术:
  多个 CPU 线程可以同时向 GPU 提交 work (每个线程用独立的 Stream)。
  → 每个 CPU 线程的 Stream 映射到独立的硬件队列
  → 不会因为单线程提交而产生伪串行化

  这对 PyTorch DataLoader 很重要:
    num_workers=4 → 4 个 CPU 线程 → 每个线程独立提交 H2D
    → 如果没有 Hyper-Q, 这些提交会在驱动层排队
    → 有 Hyper-Q → 真正的并行提交 → 更好的重叠
```

### Stream 优先级

```
cudaStreamCreateWithPriority(&stream, cudaStreamDefault, priority);
  priority: 越低越优先 (和 nice 值一样!)
  范围: 通过 cudaDeviceGetStreamPriorityRange(&low, &high) 查询
  
  用途:
    - 推理 + 训练混合场景: 推理用高优先级 (保证延迟)
    - 数据加载 Stream 用低优先级: 不影响计算关键路径

Copy Engine 数量因 GPU 而异:
  A100: 1 个 H2D + 1 个 D2H (或配置为 2 个双向)
  H100: 2+ 个 Copy Engine (取决于系统配置, NVLink 网络可能提供额外带宽)
  → 更多 Copy Engine = 可以同时做更多方向的传输
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
