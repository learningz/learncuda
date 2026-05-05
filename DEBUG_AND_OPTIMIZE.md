# CUDA 调试与优化手册

**你应该在什么时候读这个文件**:
- kernel 输出结果不对
- kernel 莫名其妙崩溃
- kernel 跑通了但很慢, 不知道从哪里优化

> **动手练习**: [`debug_exercises/`](./debug_exercises/) 包含 3 个故意有 bug 的 CUDA 程序（方向写反 / syncthreads 缺失 / exp 溢出），编译运行 → 观察现象 → 定位修复。建议配合本手册一起做。

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
  → 配合: 06_coalescing/ 实验 + theory/03 §3.4 (全局内存)

检查 2: 向量化
  是否用了 float4 加载?
  → 不是 → 改成 float4, 重新测
  → 配合: theory/03 §3.4 "向量化加载" + 10_fused_kernel/ 版本 C

检查 3: 能不能融合?
  这个 kernel 前后是不是还有其他小 kernel?
  → 是 → 融合成一个, 减少中间数据的 HBM 读写
  → 配合: 10_fused_kernel/ + theory/08 §8.2 (算子融合)

检查 4: Shared Memory 利用
  有没有数据被重复从 HBM 读取?
  → 有 → 先搬到 Shared Memory, 复用
  → 配合: 02_matrix_mul/ + theory/03 §3.3 (Shared Memory)

检查 5: Bank Conflict
  ncu 指标: Shared Bank Conflicts > 0?
  → 加 padding 或 swizzle
  → 配合: 05_bank_conflict/ + theory/03 §3.3 (Shared Memory)
```

### Compute Bound 的优化路径

```
你的 kernel 是 Compute Bound (GEMM 等):

检查 1: 是否用了 Tensor Core?
  ncu: Tensor Core Utilization > 0?
  → 没用 → 换 FP16/BF16 + WMMA/cuBLAS
  → 配合: theory/06 (Tensor Core)

检查 2: 寄存器分块 (Register Tiling)
  每线程只算 1 个元素?
  → 改成每线程算 TM×TN 个 → 提高数据复用
  → 配合: 09_register_tiling/ + theory/05 §5.9 (GEMM 优化)

检查 3: ILP
  连续指令是否有依赖链?
  → 展开循环, 增加独立指令
  → 配合: theory/08 §8.1 (ILP)

检查 4: Occupancy
  ncu: Achieved Occupancy < 25%?
  → 可能寄存器用太多 → __launch_bounds__
  → 或 Shared Memory 用太多 → 调整配置
  → 配合: theory/04 §4.5 (Occupancy)
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


## Part 3: PyTorch + CUDA 调试链

### 3.1 让异步错误变同步

```
PyTorch 的 CUDA 操作默认是异步的: kernel launch 立即返回,
错误要等到 GPU 实际执行时才发生 → 此时 Python 已经跑到后面了,
traceback 指向的位置和真实的错误位置可能隔了几十行。

解决方法: CUDA_LAUNCH_BLOCKING

  CUDA_LAUNCH_BLOCKING=1 python train.py

效果: 每次 kernel launch 后 CPU 会等 GPU 执行完,
      如果 kernel 内越界/非法访问, 错误会在 launch 的那一行立刻抛出。
      开发时必须开! 生产关闭 (有性能开销)。

在 PyTorch 代码内启用:
  import os
  os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

或用 torch 的 debug 模式:
  torch.autograd.set_detect_anomaly(True)  # 检测反向传播中的 nan/inf
```

### 3.2 用 compute-sanitizer 检查 PyTorch 扩展

```
PyTorch 内部有很多 CUDA 调用, 直接 run compute-sanitizer 会报告
大量 PyTorch 内部的"预期"警告, 淹没你的 bug。

过滤方法:
  compute-sanitizer --tool memcheck \
    --launch-timeout 0 \
    python train.py 2>&1 | grep -v "torch\|c10\|caffe2"

更好的做法: 写一个只测试你的 kernel 的 standalone 程序:
  // test_my_kernel.cu
  int main() {
      // 只分配你的 tensor, 只跑你的 kernel
      // 不引入 PyTorch 的任何依赖
  }
  nvcc -o test_my_kernel test_my_kernel.cu
  compute-sanitizer --tool memcheck ./test_my_kernel

这样可以最快定位到越界/race condition。
```

### 3.2.1 compute-sanitizer 输出解读

知道怎么跑还不够，得知道看到错误后怎么定位。

**memcheck 发现越界访问：**

```
========= Invalid __global__ read of size 4 bytes
=========     at 0x2b0 in saxpy_kernel(float, float const*, float*, int)
=========     by thread (31,0,0) in block (0,0,0)
=========     Address 0x7fff0000 is out of bounds
=========     Saved host backtrace up to driver entry point
```

解读：
1. `Invalid __global__ read of size 4 bytes` → 读了 4 bytes (1个float) 越界的全局内存
2. `at 0x2b0 in saxpy_kernel` → 错误在 saxpy_kernel 的偏移 0x2b0 处。用 `cuobjdump -sass your_program | grep "2b0"` 定位到具体 SASS 指令
3. `thread (31,0,0) in block (0,0,0)` → Block(0,0) 的 thread 31 触发的。`thread (31,0,0)` = threadIdx.x=31 → 说明是边界条件问题，最后一个线程越界了
4. `Address 0x7fff0000` → 访问的非法地址。如果是 0x0 说明是空指针解引用

**最常见错误模式：**

```
# 模式 1: 数组下标越界
错误: "out of bounds" + 最后一个线程/最后一个 block
修复: 检查 if (idx < n) 边界条件
调试: 在 kernel 中加 printf("idx=%d, n=%d\n", idx, n)

# 模式 2: Shared Memory 越界写
错误: "Invalid __shared__ write" + thread (0,0,0)
修复: 检查 SMEM 数组大小 vs 访问的索引
调试: 检查 __shared__ float smem[256]; ... smem[tid + stride]

# 模式 3: 使用已释放的内存
错误: Address 指向 free'd 区域
修复: 确保 cudaFree 在 kernel 执行之后

# 模式 4: 栈溢出
错误: "stack overflow" 或 kernel 崩溃
修复: 减少局部变量, 或加编译标志 -Xcompiler=-Wno-deprecated
       或显式指定更大的 stack frame
```

**racecheck 发现数据竞争：**

```
========= ERROR: Potential WAW hazard detected at 0x330 in reduce_kernel
=========     Write of size 4 by thread (5,0,0) in block (32,0,0)
=========     And thread (0,0,0) in block (32,0,0)
=========     Both writing to address 0x7fff8000
```

解读：两个不同 Block 的线程同时写同一地址 → 说明缺少 `__syncthreads()` 或应该用 `atomicAdd`。`WAW` = Write After Write hazard。还有 `RAW` (Read After Write) 和 `WAR` (Write After Read)。

**常用参数：**

```bash
# 内存越界检查 (最常用)
compute-sanitizer --tool memcheck ./program

# 竞争条件检查
compute-sanitizer --tool racecheck ./program

# 初始化检查 (使用了未初始化的内存)
compute-sanitizer --tool initcheck ./program

# 同步检查 (__syncthreads 位置不对)
compute-sanitizer --tool synccheck ./program

# 显示更详细的 backtrace
compute-sanitizer --print-limit 100 --tool memcheck ./program

# 组合: 输出所有错误，每个错误最多 100 条记录
compute-sanitizer --print-limit 100 --launch-timeout 0 --tool memcheck ./program
```

**关联回源码行（关键步骤）：**

```bash
# 1. 用 -lineinfo 编译
nvcc -O2 -lineinfo -o my_kernel my_kernel.cu

# 2. 跑 compute-sanitizer
compute-sanitizer --tool memcheck ./my_kernel
# 输出: at 0x2b0 in my_kernel

# 3. 用 cuobjdump 找到地址对应行
cuobjdump -sass my_kernel | grep -B5 "2b0"

# 4. 用 ncu 的源码视图 (更方便!):
ncu --set source -o report ./my_kernel
ncu-ui report.ncu-rep
# GUI 中直接看到每行 CUDA 源码 + 对应的 SASS 指令 + 性能数据
```


### 3.2.2 用 cuda-gdb 断点调试

compute-sanitizer 告诉你"有什么问题"，但有时你需要像普通程序一样**单步执行 GPU 代码**来看看问题到底怎么发生的。这就是 cuda-gdb 的用途。

**基本用法：**

```bash
# 编译时加 -G (debug 模式, 关闭优化!)
nvcc -G -g -o my_kernel_debug my_kernel.cu

# 启动 cuda-gdb
cuda-gdb ./my_kernel_debug

# 在 cuda-gdb 内:
(cuda-gdb) break saxpy_kernel        # 在 kernel 入口设断点
(cuda-gdb) run                       # 运行程序
# → 停在第一个 block 的第一个线程
(cuda-gdb) info cuda threads         # 查看所有活跃的 GPU 线程
(cuda-gdb) info cuda kernels         # 查看当前执行的 kernel
(cuda-gdb) cuda thread (0,0,0)       # 切换到 block(0,0) thread(0,0)
(cuda-gdb) stepi                     # 单步一条 SASS 指令 (step instruction)
(cuda-gdb) info registers            # 查看当前 GPU 线程的寄存器值
(cuda-gdb) print $R0                 # 查看寄存器 R0 的值
(cuda-gdb) continue                  # 继续执行到下一个断点
```

**常用 cuda-gdb 命令：**

```
断点:
  break kernel_name                  kernel 入口断点
  break file.cu:42                   CUDA 源码行断点
  break file.cu:42 thread (0,0,0)   只在线程(0,0,0)断点

切换上下文:
  cuda thread (b,t,0)               切换到 block(b), thread(t)
  cuda block (b,0,0)                切换到 block(b) 的所有线程
  cuda device 0 sm 0 warp 0 lane 0  按 SM/Warp/Lane 选择

查看状态:
  info cuda threads                 所有活跃线程 (前 10 个)
  info cuda kernels                 当前执行的所有 kernel
  info cuda sms                     SM 状态
  print variable_name               查看当前线程的变量
  print *((float*)0x7fff0000)       查看指定地址的内存内容

执行控制:
  stepi                            单步一条 SASS 指令 (当前线程)
  nexti                            单步一条 SASS 指令 (跳过函数调用)
  continue                         继续到下一断点
  cuda all stepi                   所有活跃线程一起单步!
```

**重要限制：**

```
1. -G 编译会关闭几乎所有优化
   → 寄存器分配、循环展开、指令调度全部不同!
   → 你在 cuda-gdb 中看到的 SASS 和 release 版完全不同
   → 适合找逻辑 bug, 不适合分析性能

2. 不能同时调试 CPU 和 GPU 代码
   → cuda-gdb 一次只能调试一端

3. 单步在大量线程上很慢
   → 用条件断点: break kernel_name thread (3,0,0) if tid == 123
   → 或先用 compute-sanitizer 找到出错的线程, 再设断点

4. printf 比断点更高效
   → 对于 "这个循环走了几次" 这类问题:
     printf("tid=%d, iter=%d\n", tid, i);
   → 比 cuda-gdb 打断点 + continue 快得多!
```

**何时用 cuda-gdb vs printf vs compute-sanitizer：**

```
场景                                推荐工具
────                                ────────
结果总是 NaN/Inf                    printf (看中间值)
结果偶尔不对 (race condition)        compute-sanitizer --tool racecheck
结果在特定配置下不对 (死锁等)         cuda-gdb (条件断点)
越界访问                            compute-sanitizer --tool memcheck
性能不好                            ncu (不是 cuda-gdb!)
理解 kernel 的执行流程              cuda-gdb + 小数据量 (如 N=4)
```



### 3.3 用 ncu profile PyTorch 脚本

```
直接跑:
  ncu --set basic --target-processes all python train.py

问题: PyTorch 有 1000+ 个 kernel, ncu 输出太多, 找不到你想要的那一个。

更好的做法: 只 profile 特定的代码段
  import torch.cuda.profiler as profiler

  profiler.start()
  # 只放你想 profile 的几行代码
  output = my_custom_layernorm(x, gamma, beta)
  torch.cuda.synchronize()
  profiler.stop()

  # 然后用 ncu 运行:
  # ncu --set basic --profile-from-start off python train.py

或用 PyTorch Profiler:
  with torch.profiler.profile(
      activities=[torch.profiler.ProfilerActivity.CUDA],
      record_shapes=True
  ) as prof:
      output = my_model(input)
      torch.cuda.synchronize()
  print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 3.4 用 autograd.gradcheck 验证反向传播

```python
import torch
from torch.autograd import gradcheck

# 你的自定义 autograd Function
class MyLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        ctx.save_for_backward(x, gamma, beta)
        ctx.eps = eps
        return my_cuda_ext.forward(x, gamma, beta, eps)

    @staticmethod
    def backward(ctx, grad_output):
        x, gamma, beta = ctx.saved_tensors
        return my_cuda_ext.backward(grad_output, x, gamma, beta, ctx.eps)

# 梯度检查 (比对你的 backward 和数值梯度)
x = torch.randn(4, 256, device='cuda', requires_grad=True)
gamma = torch.randn(256, device='cuda', requires_grad=True)
beta = torch.randn(256, device='cuda', requires_grad=True)

test = gradcheck(
    MyLayerNorm.apply,
    (x, gamma, beta, 1e-5),
    eps=1e-3,        # 数值梯度的步长
    atol=1e-4,       # 绝对误差容忍
    rtol=1e-3        # 相对误差容忍
)
print("gradcheck:", "PASS" if test else "FAIL")
```

### 3.5 PyTorch 扩展开发流程

```
推荐开发顺序 (减少 debug 痛苦):

Stage 1: 纯 C++ standalone (不含 PyTorch)
  cd your_module && nvcc -o test test_kernel.cu
  ./test
  → 验证 kernel 逻辑正确

Stage 2: C++ 单元测试
  compute-sanitizer --tool memcheck ./test
  → 验证无越界

Stage 3: PyTorch binding
  python setup.py install
  python -c "import my_ext; my_ext.forward(...)"
  CUDA_LAUNCH_BLOCKING=1 python test_my_ext.py
  → 验证 binding 正确

Stage 4: 反向传播验证
  python test_backward.py  # 包含 gradcheck

Stage 5: 性能 profile
  python profile_my_ext.py
  ncu --set basic python profile_my_ext.py

不要在 PyTorch 里 debug kernel 逻辑!
进入 PyTorch 之前先确保 standalone kernel 是正确的。
```
