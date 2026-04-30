# CUDA 调试与优化手册

**你应该在什么时候读这个文件**:
- kernel 输出结果不对
- kernel 莫名其妙崩溃
- kernel 跑通了但很慢, 不知道从哪里优化

## Part 1: 调试 — 结果不对或程序崩溃

### 第一件事: 加上 CUDA_CHECK 错误检查

```
任何 CUDA 程序都应该在每个 CUDA API 调用后检查错误。
不加检查 → 错误静默发生 → 你看到的症状和真实原因相隔千里 → 浪费大量调试时间。

推荐的 CUDA_CHECK 宏 (复制到你的每个 .cu 文件开头):

  #define CUDA_CHECK(call) do {                                              \
      cudaError_t err = (call);                                              \
      if (err != cudaSuccess) {                                              \
          fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                  cudaGetErrorString(err)); exit(1); }                       \
  } while(0)

使用方法:
  CUDA_CHECK(cudaMalloc(&d_ptr, bytes));            // 分配可能失败 (显存不够)
  CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, bytes, ...)); // 拷贝可能失败 (指针错误)
  CUDA_CHECK(cudaDeviceSynchronize());               // 捕获之前所有异步错误
  CUDA_CHECK(cudaGetLastError());                    // 捕获 kernel launch 错误

为什么 kernel launch 需要特殊处理?
  kernel<<<grid, block>>>() 本身不返回 cudaError_t。
  它只是把命令提交到 GPU 队列，错误要等执行时才发生。
  两种方式捕获:
    1. 在 kernel 后立即调用 CUDA_CHECK(cudaGetLastError()) — 检查 launch 参数
    2. 在 synchronize 后调用 CUDA_CHECK — 检查 kernel 执行中的错误

完整的 kernel 检查模板:
  my_kernel<<<grid, block>>>(args...);
  CUDA_CHECK(cudaGetLastError());         // 捕获 launch 错误 (如 blockSize > 1024)
  CUDA_CHECK(cudaDeviceSynchronize());    // 捕获执行错误 (如越界访问)
  // 开发阶段加 synchronize，发布时去掉（性能开销）

常见的被 CUDA_CHECK 立即定位的问题:
  - cudaMalloc 返回 out of memory → 显存不够
  - cudaMemcpy 返回 invalid argument → 指针或大小写错了
  - kernel launch 返回 invalid configuration → blockSize 超过 1024
  - cudaDeviceSynchronize 返回 illegal memory access → kernel 越界
```

### 症状 1: kernel 输出全是 0 或垃圾值

```
最常见原因 (按概率排序):

1. 忘了 cudaMemcpy (Host → Device)
   → 检查: GPU 上的输入数据有没有正确拷贝过去?
   → 陷阱: cudaMalloc 不会清零! 新分配的显存内容是随机的。

2. kernel launch 配置有误
   → 检查: gridSize × blockSize ≥ 数据量 N?
   → 陷阱: gridSize 算错 → 部分元素没有被处理 → 结果有随机值

3. 线程索引计算错误
   → 检查: int idx = blockIdx.x * blockDim.x + threadIdx.x 对不对?
   → 2D Grid: row = blockIdx.y * blockDim.y + threadIdx.y (不是 .x!)

4. 缺少边界检查
   → 检查: if (idx < n) 有没有?
   → 总线程数通常 > n → 多余线程必须跳过

5. 忘了 cudaMemcpy (Device → Host)
   → 检查: 是不是忘了把结果拷回来?
```

### 症状 2: "illegal memory access" 崩溃

```
原因: kernel 中的指针访问了非法地址 (越界或空指针)。

定位方法:
  compute-sanitizer --tool memcheck ./your_program
  
  输出类似:
    ========= Invalid __global__ read of size 4 bytes
    =========     at 0x00000148 in my_kernel(float*, int)
    =========     by thread (255,0,0) in block (4095,0,0)
    =========     Address 0x7f3a00400000 is out of bounds
    
  → 告诉你哪个 kernel、哪个线程、什么地址越界了!

常见原因:
  1. 缺少 if (idx < n) 边界检查
  2. 2D 索引的 row/col 超出矩阵范围
  3. Shared Memory 数组越界 (__shared__ float s[256]; s[300] = ...)
  4. 传入 kernel 的指针来自已 cudaFree 的内存
  5. 在 Host 端解引用 Device 指针 (d_ptr[0] → 段错误!)
```

### 症状 3: 结果"几乎对"但有微小误差

```
这通常不是 bug, 而是浮点精度问题。

检查清单:
  1. 你的 tolerance 是否合理?
     float: 1e-5 到 1e-6 是合理的 (7 位有效数字)
     half:  1e-2 到 1e-3 (3-4 位有效数字)
  
  2. 是否涉及大量累加?
     float 累加 100 万个数 → 相对误差可能到 1e-3
     → 用 double 做参考, 或用 Kahan 求和
  
  3. 是否用了快速数学 (--use_fast_math)?
     __sinf/__expf 等快速版本精度较低 (~2 ULP 误差)
     标准版 sinf/expf 更精确但更慢
  
  4. FMA 的精度差异?
     a*b+c 可能被编译为 FMA (1次舍入) 或 MUL+ADD (2次舍入)
     两者结果可能有 1 ULP 差异 → 正常!
```

### 症状 4: kernel 运行但结果不稳定 (每次跑不一样)

```
原因: 数据竞争 (Race Condition)

常见场景:
  1. 多个线程写同一地址但没用 atomicAdd
     → 结果取决于哪个线程先写, 每次不同
     
  2. 读 Shared Memory 之前忘了 __syncthreads()
     → 可能读到其他线程还没写完的值
     → 在某些 GPU 上可能碰巧正确, 换 GPU 就挂了!
  
  3. 跨 Block 通信没用 __threadfence()
     → 写入对其他 Block 不可见的时机不确定

检测工具:
  compute-sanitizer --tool racecheck ./your_program
  → 检测 Shared Memory 的数据竞争
  
  compute-sanitizer --tool initcheck ./your_program
  → 检测读取未初始化的显存
```


## Part 2: 优化 — "我的 kernel 慢, 怎么办?"

### 完整的优化思维流程

```
Step 1: 量化问题 — "到底有多慢?"
  ├── 计算理论最短时间:
  │     Memory Bound: t_theory = total_bytes / peak_bandwidth
  │     Compute Bound: t_theory = total_flops / peak_flops
  ├── 测量实际时间 (cudaEvent)
  └── 计算效率 = t_theory / t_actual
      > 80% → 已经很好了, 优化空间有限
      < 50% → 有明显问题, 继续排查
      < 20% → 有严重问题

Step 2: 判断瓶颈类型 — "Memory Bound 还是 Compute Bound?"
  ├── 方法 A (理论): 算术强度 AI = FLOP / Byte
  │     AI < Ridge Point → Memory Bound
  │     AI > Ridge Point → Compute Bound
  └── 方法 B (实测): ncu --set full ./program
        看 "GPU Speed Of Light":
        Memory% >> Compute% → Memory Bound
        Compute% >> Memory% → Compute Bound

Step 3: 根据瓶颈类型选择优化方向
```

### Memory Bound 的优化路径

```
你的 kernel 是 Memory Bound (大多数情况):

检查 1: 合并访问
  ncu 指标: Global Load Efficiency 或 Sectors/Request
  效率 < 50%? → 重新设计内存访问模式
  → 配合: 06_coalescing/ 实验 + Ch3.4 理论

检查 2: 向量化
  是否用了 float4 加载?
  → 不是 → 改成 float4, 重新测
  → 配合: Ch3.4 "向量化加载" + 10_fused_kernel/ 版本 C

检查 3: 能不能融合?
  这个 kernel 前后是不是还有其他小 kernel?
  → 是 → 融合成一个, 减少中间数据的 HBM 读写
  → 配合: 10_fused_kernel/ + Ch8.2

检查 4: Shared Memory 利用
  有没有数据被重复从 HBM 读取?
  → 有 → 先搬到 Shared Memory, 复用
  → 配合: 02_matrix_mul/ + Ch3.3

检查 5: Bank Conflict
  ncu 指标: Shared Bank Conflicts > 0?
  → 加 padding 或 swizzle
  → 配合: 05_bank_conflict/ + Ch3.3
```

### Compute Bound 的优化路径

```
你的 kernel 是 Compute Bound (GEMM 等):

检查 1: 是否用了 Tensor Core?
  ncu: Tensor Core Utilization > 0?
  → 没用 → 换 FP16/BF16 + WMMA/cuBLAS
  → 配合: Ch6 (Tensor Core)

检查 2: 寄存器分块 (Register Tiling)
  每线程只算 1 个元素?
  → 改成每线程算 TM×TN 个 → 提高数据复用
  → 配合: 09_register_tiling/ + Ch5.9

检查 3: ILP
  连续指令是否有依赖链?
  → 展开循环, 增加独立指令
  → 配合: Ch8.1

检查 4: Occupancy
  ncu: Achieved Occupancy < 25%?
  → 可能寄存器用太多 → __launch_bounds__
  → 或 Shared Memory 用太多 → 调整配置
  → 配合: Ch4.5
```

### 优化后的验证清单

```
每次优化后必须做:
  □ 正确性验证: 和 CPU 参考/优化前的结果对比 (误差 < tolerance)
  □ 性能验证: 确认确实变快了 (用 cudaEvent 计时, 100 次取平均)
  □ 回归测试: 不同数据大小 (小/中/大) 都正确且都更快
  □ 边界情况: N 不是 blockSize 的倍数? N=1? N=很大?

常见的"优化变慢"原因:
  - 寄存器用多了 → Occupancy 降低 → 延迟隐藏不够 → 反而变慢
  - Shared Memory 用多了 → 同上
  - 循环展开太多 → I-Cache 压力 → 指令缓存 Miss → 变慢
  → 每次优化只改一个变量, 对比前后性能!
```
