# 调试实战练习

**目标**: 每个文件包含 1-2 个故意埋入的 bug。编译运行后你会看到**错误的结果或崩溃**。
你的任务是：定位 bug → 理解原因 → 修复它。

**建议工具**:
- `compute-sanitizer --tool memcheck ./program` — 检测越界访问
- `compute-sanitizer --tool racecheck ./program` — 检测数据竞争
- `CUDA_CHECK` 宏 — 捕获 CUDA API 错误
- 详细的调试方法论见 [`../DEBUG_AND_OPTIMIZE.md`](../DEBUG_AND_OPTIMIZE.md)

## 练习列表

| 文件 | 难度 | 涉及知识点 |
|------|------|-----------|
| `bug1_vector_add.cu` | ⭐ | 内存管理、cudaMemcpy 方向 |
| `bug2_reduce.cu` | ⭐⭐ | __syncthreads 缺失、数据竞争 |
| `bug3_softmax.cu` | ⭐⭐ | 数值稳定性、Online 算法修正因子 |

## 怎么做

```bash
cd debug_exercises

# 编译 (会成功, bug 在逻辑里不在语法里)
nvcc -O2 -o bug1 bug1_vector_add.cu
nvcc -O2 -o bug2 bug2_reduce.cu
nvcc -O2 -o bug3 bug3_softmax.cu

# 运行, 观察错误现象
./bug1    # 结果全是 0 — 为什么?
./bug2    # 结果不稳定, 每次跑可能不一样 — 为什么?
./bug3    # 大输入时输出全是 NaN — 为什么?
```

## 提示 (先自己找, 找不到再看)

<details>
<summary>bug1 提示</summary>
仔细看 cudaMemcpy 的第四个参数 (方向)。数据真的到 GPU 了吗?
</details>

<details>
<summary>bug2 提示</summary>
在读 Shared Memory 之前, 确保所有写操作都完成了吗?
用 compute-sanitizer --tool racecheck 跑一下。
</details>

<details>
<summary>bug3 提示</summary>
Softmax 的数值稳定性靠什么保证? Online 算法的修正因子方向对吗?
</details>

## 参考答案

每个文件底部的注释中有答案（搜索 "BUG ANSWER"）。先自己修！
