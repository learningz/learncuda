# CUDA 优化速查表

## Roofline 模型

```
算术强度 AI = FLOP / Byte

A100 参考值:
  峰值计算: FP32=19.5 TFLOPS, FP16 TC=312 TFLOPS
  峰值带宽: ~2 TB/s
  Ridge Point = 19.5T / 2T ≈ 10 FLOP/Byte

常见算子的 AI:
  向量加法:  1/(3×4) = 0.08   → Memory Bound
  GELU:      ~2               → Memory Bound
  Softmax:   ~2-3             → Memory Bound
  LayerNorm: ~2-4             → Memory Bound
  GEMM 512:  ~170             → Compute Bound
  Conv 3×3:  ~15-30           → 边界区域

Memory Bound → 优化访存 (合并, SMEM, 融合)
Compute Bound → 优化计算 (Tensor Core, ILP)
```

## 线程编号公式

```cuda
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// Grid-Stride Loop (每线程处理多元素)
for (int i = idx; i < N; i += blockDim.x * gridDim.x) { ... }
```

## Kernel Launch 配置

```
blockSize: 必须是 32 的倍数, ≤1024, 常用 256
gridSize:  ceil(N / blockSize)
Occupancy: 每个 SM 最多 2048 线程 / 64 Warps / 32 Blocks

经验值:
  Memory Bound → 用大的 blockSize (512-1024), 减少 gridSize
  Compute Bound → 用小的 blockSize (128-256), 增加 Occupancy
  寄存器压力大 → 降低 blockSize
```

## 内存层级

```
层级        容量/SM     延迟        scope
寄存器       256KB       0 cycle    单线程
Shared Mem   ~100KB      5 cycles   Block 内
L1 Cache     128KB      28 cycles   自动
L2 Cache     ~40MB      200 cycles  全局
HBM          80GB       500 cycles  全局
```

## Shared Memory 关键操作

```cuda
// 声明
__shared__ float tile[16][16];

// 协作加载 (每个线程搬 1 个)
tile[threadIdx.y][threadIdx.x] = global[row * N + col];

// 屏障 (必须!)
__syncthreads();

// Bank Conflict 避免:
// - 加 padding: __shared__ float tile[16][16+1];
// - 改变访问模式: tile[threadIdx.x][threadIdx.y] vs tile[threadIdx.y][threadIdx.x]
// Bank 数 = 32, Bank 宽度 = 4 bytes
// Conflict: 同 Bank 的不同地址 → 串行访问
```

## Warp Shuffle (寄存器级通信)

```cuda
// 不需要 Shared Memory, 不需要 __syncthreads!
float val = ...;
val += __shfl_down_sync(0xffffffff, val, 16);
val += __shfl_down_sync(0xffffffff, val, 8);
val += __shfl_down_sync(0xffffffff, val, 4);
val += __shfl_down_sync(0xffffffff, val, 2);
val += __shfl_down_sync(0xffffffff, val, 1);
// val 现在是 Warp 内所有线程值的总和 (仅 lane 0 持有)

// 掩码 0xffffffff = 全部 32 线程参与
// shfl_xor: 蝴蝶交换
// shfl: 从指定 lane 取值
```

## 常见优化技术

| 技术 | 适用场景 | 预期收益 |
|------|----------|----------|
| 合并访问 | 所有 kernel | 最高 32× 带宽提升 |
| Shared Memory Tiling | 有数据复用的 kernel | 2-10× |
| Warp Shuffle | Reduce/Scan | 1.5-3× vs SMEM |
| Grid-Stride Loop | 减少调度开销 | 1.2-2× |
| float4 向量化 | Memory Bound | 1.2-1.5× |
| 循环展开 (ILP) | 内存延迟大 | 1.2-2× |
| 算子融合 | 多个小 kernel | 1.5-5× |
| Tensor Core | GEMM/Conv | 5-16× vs FP32 |

## ncu 关键指标

```
ncu --set full ./my_program

关键指标:
  Memory Throughput: 越接近峰值越好 (Memory Bound 时关注)
  SOL DRAM:         显存带宽利用率 (%)
  SOL SM:           SM 利用率 (%)
  SOL L1/TEX:       L1/Tex Cache 命中率
  
  Long Scoreboard:  Warp 等待内存 → Memory Bound
  Short Scoreboard: Warp 等待指令结果 → Compute Bound
  Not Selected:     Warp 未就绪 → Occupancy 不够
  
  Achieved Occupancy: 实际每个 SM 的活跃 Warp 数
  Theoretical:        理论最大 Occupancy
```

## 数值精度速查

```
FP32: 1s+8e+23m, 范围 ±3.4e38, 精度~7位, 4 bytes
TF32: 1s+8e+10m, 范围 ±3.4e38, 精度~3位, (内部格式)
BF16: 1s+8e+7m,  范围 ±3.4e38, 精度~2位, 2 bytes
FP16: 1s+5e+10m, 范围 ±65504,  精度~3位, 2 bytes
FP8:  1s+4e+3m,  范围 ±240,    精度~1位, 1 byte (E4M3)

混合精度黄金法则:
  - 矩阵乘: FP16/BF16 输入 + FP32 累加
  - Reduce/Norm: FP32 内部计算
  - 权重更新: 始终 FP32 (否则小梯度丢失)
  - FP16 训练: 需要 Loss Scaling
  - BF16 训练: 通常不需要 Loss Scaling
```

## 调试速查

```bash
# 越界检查
compute-sanitizer --tool memcheck ./program

# 数据竞争检查
compute-sanitizer --tool racecheck ./program

# 快速检查 ncu 瓶颈
ncu --set basic --target-processes all ./program

# 编译时保留行号
nvcc -lineinfo -o prog prog.cu
```

## CUDA_CHECK 宏

```cuda
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// kernel launch 检查:
kernel<<<g,b>>>(...);
CUDA_CHECK(cudaGetLastError());        // launch 参数错误
CUDA_CHECK(cudaDeviceSynchronize());   // 执行时错误
```
