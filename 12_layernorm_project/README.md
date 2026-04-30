# 综合实战: 手写 LayerNorm 并接入 PyTorch

**难度**: ⭐⭐⭐
**预计时间**: 2-4 小时
**你将练习**: Shared Memory（参考 [`theory/03_memory_hierarchy.md`](../theory/03_memory_hierarchy.md)） + Warp Shuffle（参考 [`theory/04_warp_and_sync.md`](../theory/04_warp_and_sync.md)） + 算子开发（参考 [`theory/05_operator_development.md`](../theory/05_operator_development.md)） + Welford 算法（参考 [`theory/07_classic_operators.md`](../theory/07_classic_operators.md) 7.2 节）


## 背景

LayerNorm 是 Transformer 的基本组件, 对一行数据做归一化:
```
y[i] = gamma * (x[i] - mean) / sqrt(variance + eps) + beta
```
其中 mean 和 variance 对每一行独立计算。

这个算子综合了你学过的几乎所有核心技术:
- **Reduce**（求 mean 和 variance，可参考 [`03_reduce/`](../03_reduce/) 和 [`theory/04_warp_and_sync.md`](../theory/04_warp_and_sync.md)）
- **Shared Memory**（Block 内线程通信，可参考 [`theory/03_memory_hierarchy.md`](../theory/03_memory_hierarchy.md)）
- **数值稳定性**（Welford 算法 → [`theory/07_classic_operators.md`](../theory/07_classic_operators.md) 7.2 节）
- **PyTorch 接入**（`autograd.Function` + C++ binding，可参考 [`04_pytorch_extension/`](../04_pytorch_extension/) 和 [`theory/05_operator_development.md`](../theory/05_operator_development.md)）


## 任务

### Part 1: 写 CUDA kernel (在 layernorm_cuda.cu 中)

```
实现 layernorm_forward_kernel:
  输入:  x (shape: [rows, cols]), gamma, beta (shape: [cols])
  输出:  y (shape: [rows, cols])
  
  每个 Block 处理一行:
  1. 用 Welford 算法一次遍历计算 mean 和 variance
     (提示: 见 [`theory/07_classic_operators.md`](../theory/07_classic_operators.md) 7.2 节)
  2. 用 Warp Shuffle 在 Block 内归约 Welford 统计量
     (提示: Welford 的合并公式见 [`theory/07_classic_operators.md`](../theory/07_classic_operators.md) 7.2 节)
  3. 归一化: y[i] = gamma[i] * (x[i] - mean) * rsqrt(var + eps) + beta[i]

性能目标:
  只遍历数据 2 次 (1次算统计 + 1次归一化)
  使用 Warp Shuffle 减少同步开销
```

### Part 2: 接入 PyTorch (在 setup.py + test_layernorm.py 中)

```
完整的 PyTorch 接入代码已准备好:
  layernorm_cuda.cu   — CUDA kernel + C++ binding (含详细注释)
  setup.py            — 编译配置
  test_layernorm.py   — 正确性验证 + 性能对比 + 梯度检查

安装和测试:
  cd 12_layernorm_project
  pip install -e .              # 编译 CUDA 扩展
  python test_layernorm.py      # 运行测试

如果你想自己从零写接入代码 (更有挑战性):
  1. 参考 [04_pytorch_extension/](../04_pytorch_extension/) 的模式
  2. 写 C++ binding: 接收 torch::Tensor, 调用你的 kernel
  3. 写 Python autograd.Function: 封装 forward/backward
  4. 对比你的实现和 torch.nn.functional.layer_norm 的:
     - 数值正确性 (最大误差 < 1e-5)
     - 性能 (用 torch.cuda.Event 计时)
```

### Part 3: 用 ncu 分析性能

```
1. ncu --set full ./your_program
2. 检查:
   - Memory Throughput (应该接近峰值, 因为 LayerNorm 是 Memory Bound)
   - Bank Conflict (如果用了 Shared Memory)
   - Warp Stall Reasons (应该主要是 Long Scoreboard = 等内存)
3. 尝试加 float4 向量化, 看带宽是否提升
```


## 提示 (如果卡住了)

```
提示 1: Welford 合并公式
  两组统计量 (count_a, mean_a, M2_a) 和 (count_b, mean_b, M2_b) 合并:
  count = count_a + count_b
  delta = mean_b - mean_a
  mean = (count_a * mean_a + count_b * mean_b) / count
  M2 = M2_a + M2_b + delta^2 * count_a * count_b / count

提示 2: 每个 Block 处理一行, blockDim.x = 256
  如果 cols = 4096, 每个线程用 Grid-Stride Loop 处理 4096/256 = 16 个元素
  先局部 Welford, 再 Warp Shuffle 合并, 再 Block 合并

提示 3: PyTorch autograd.Function 模板
  class MyLayerNorm(torch.autograd.Function):
      @staticmethod
      def forward(ctx, x, gamma, beta, eps):
          ctx.save_for_backward(x, gamma, beta)
          ctx.eps = eps
          return my_cuda_ext.forward(x, gamma, beta, eps)
      
      @staticmethod
      def backward(ctx, grad_output):
          x, gamma, beta = ctx.saved_tensors
          return my_cuda_ext.backward(grad_output, x, gamma, ctx.eps)
```


## 评估标准

```
✓ 正确性: 和 torch.nn.functional.layer_norm 的最大误差 < 1e-5
✓ 性能:   达到理论带宽的 50%+ (有效带宽 / GPU 峰值带宽)
✓ 代码:   关键步骤有注释解释 "为什么这样做"
★ 加分:   支持 FP16 输入 + FP32 累加器 (混合精度)
★ 加分:   实现反向传播 kernel
```
