// ============================================================
// 12: LayerNorm CUDA + PyTorch Binding
//
// 这个文件将 layernorm_solution.cu 中的 kernel 封装为 PyTorch 可调用的 C++ 扩展。
//
// 编译链路:
//   setup.py → torch.utils.cpp_extension → nvcc 编译本文件
//   → 生成 custom_layernorm.so → Python 中 import custom_layernorm
//
// 和纯 CUDA 版本的区别:
//   1. 不需要手动 cudaMalloc/cudaMemcpy — PyTorch 管理内存
//   2. 输入是 torch::Tensor — 直接用 .data_ptr<float>() 拿到 GPU 指针
//   3. 通过 pybind11 暴露给 Python — 可以嵌入 autograd 计算图
//
// 配合阅读: 04_pytorch_extension/pytorch_extension.md
// ============================================================

#include <torch/extension.h>
#include <cuda_runtime.h>

// ============================================================
// Warp 级归约：32 个线程通过寄存器直接交换数据求和
// 比 Shared Memory 归约快 5 倍（~1 cycle vs ~5 cycle），且不需要 __syncthreads()
// ============================================================
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================
// 前向 Kernel
//
// 每个 Block 处理输入矩阵的一行 (一个 sample 的一个 hidden dimension)
// 两遍扫描:
//   Pass 1: Grid-Stride Loop 累加 sum 和 sum_of_squares → 求 mean 和 variance
//   Pass 2: 逐元素归一化 y[i] = gamma[i] * (x[i] - mean) * rsqrt(var + eps) + beta[i]
//
// 为什么是两遍而不是一遍?
//   归一化需要 mean 和 rstd，而这两个值需要看完整行才能算出来。
//   不能一边算统计量一边归一化（除非用 Online 算法写回，但那更复杂）。
//
// 为什么保存 mean 和 rstd?
//   反向传播需要它们。ctx.save_for_backward() 会引用这些 tensor。
// ============================================================
__global__ void layernorm_forward_kernel(
    const float *x,       // [rows, cols] 输入
    const float *gamma,   // [cols] 可学习的缩放参数
    const float *beta,    // [cols] 可学习的偏移参数
    float *y,             // [rows, cols] 输出
    float *mean_out,      // [rows] 保存每行的均值（反向需要）
    float *rstd_out,      // [rows] 保存每行的 1/sqrt(var+eps)（反向需要）
    int rows, int cols, float eps)
{
    // blockIdx.x 对应行号 — 每个 Block 独立处理一行
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *xr = x + row * cols;    // 指向本行起始位置
    float *yr = y + row * cols;

    // ---- Pass 1: 计算 mean 和 variance ----
    // 每个线程用 Grid-Stride Loop 处理 cols/blockDim.x 个元素
    // 例如 cols=768, blockDim=256 → 每线程处理 3 个元素
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = xr[i];
        local_sum += val;           // 累加 x[i]
        local_sq_sum += val * val;  // 累加 x[i]²
    }

    // 阶段 1: Warp 内归约 (32 个线程 → 1 个部分和)
    local_sum = warp_reduce_sum(local_sum);
    local_sq_sum = warp_reduce_sum(local_sq_sum);

    // 阶段 2: Warp 间归约 (通过 Shared Memory)
    // blockDim=256 → 8 个 Warp → 8 个部分和需要汇总
    __shared__ float s_sum[32];     // 最多支持 32 个 Warp (blockDim ≤ 1024)
    __shared__ float s_sq[32];
    int warp_id = tid / 32;
    int lane = tid % 32;

    // 每个 Warp 的 lane 0 把本 Warp 的归约结果写入 Shared Memory
    if (lane == 0) {
        s_sum[warp_id] = local_sum;
        s_sq[warp_id] = local_sq_sum;
    }
    __syncthreads();  // 等所有 Warp 写完

    // 阶段 3: 第一个 Warp 读出所有部分和，再做一次 Warp 归约
    int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        local_sum = (lane < num_warps) ? s_sum[lane] : 0.0f;
        local_sq_sum = (lane < num_warps) ? s_sq[lane] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        local_sq_sum = warp_reduce_sum(local_sq_sum);
    }

    // 用 Shared Memory 广播 mean 和 rstd 给所有线程
    __shared__ float s_mean, s_rstd;
    if (tid == 0) {
        float mean = local_sum / cols;
        // var = E[x²] - E[x]² （等价于 Σ(x-mean)²/N，但只需一遍扫描）
        float var = local_sq_sum / cols - mean * mean;
        float rstd = rsqrtf(var + eps);   // rsqrt = 1/sqrt，硬件单条指令
        s_mean = mean;
        s_rstd = rstd;
        mean_out[row] = mean;     // 保存给反向传播用
        rstd_out[row] = rstd;
    }
    __syncthreads();  // 等 thread 0 算完

    float mean = s_mean;
    float rstd = s_rstd;

    // ---- Pass 2: 逐元素归一化 ----
    for (int i = tid; i < cols; i += blockDim.x) {
        yr[i] = gamma[i] * (xr[i] - mean) * rstd + beta[i];
    }
}

// ============================================================
// 反向 Kernel
//
// LayerNorm 的反向传播比前向复杂得多。
//
// 数学推导（链式法则）:
//   前向: y[i] = gamma[i] * xhat[i] + beta[i]
//          xhat[i] = (x[i] - mean) * rstd
//
//   对 x 的梯度:
//     dx[i] = rstd * gamma[i] * (dy[i] - mean(dy·gamma) - xhat[i] * mean(dy·gamma·xhat))
//
//   对 gamma 的梯度: dgamma[i] = Σ_rows dy[i] * xhat[i]
//   对 beta 的梯度:  dbeta[i]  = Σ_rows dy[i]
//
// 实现策略:
//   - 每个 Block 处理一行，计算该行的 dx
//   - dgamma/dbeta 需要跨所有行累加 → 用 atomicAdd（简单但非最优）
//   - 更高性能的做法是用两阶段 reduce，但为了教学清晰性这里用 atomic
// ============================================================
__global__ void layernorm_backward_kernel(
    const float *dy,         // [rows, cols] 从上游传来的梯度
    const float *x,          // [rows, cols] 前向时保存的输入
    const float *gamma,      // [cols] 前向时的 gamma
    const float *mean_saved, // [rows] 前向时保存的 mean
    const float *rstd_saved, // [rows] 前向时保存的 rstd
    float *dx,               // [rows, cols] 输出：对 x 的梯度
    float *dgamma,           // [cols] 输出：对 gamma 的梯度（调用前需清零！）
    float *dbeta,            // [cols] 输出：对 beta 的梯度（调用前需清零！）
    int rows, int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float mean = mean_saved[row];
    float rstd = rstd_saved[row];

    const float *dy_row = dy + row * cols;
    const float *x_row = x + row * cols;
    float *dx_row = dx + row * cols;

    // ---- 先算两个中间 reduce ----
    // sum_dy      = Σ dy[j] * gamma[j]         （归一化后梯度的加权和）
    // sum_dy_xhat = Σ dy[j] * gamma[j] * xhat[j]  （归一化后梯度与归一化值的协方差）
    float local_sum_dy = 0.0f;
    float local_sum_dy_xhat = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float xhat = (x_row[i] - mean) * rstd;
        float dy_gamma = dy_row[i] * gamma[i];
        local_sum_dy += dy_gamma;
        local_sum_dy_xhat += dy_gamma * xhat;
    }

    // Block 级归约（和前向完全相同的模式）
    local_sum_dy = warp_reduce_sum(local_sum_dy);
    local_sum_dy_xhat = warp_reduce_sum(local_sum_dy_xhat);

    __shared__ float s1[32], s2[32];
    int warp_id = tid / 32, lane = tid % 32;
    int num_warps = blockDim.x / 32;
    if (lane == 0) {
        s1[warp_id] = local_sum_dy;
        s2[warp_id] = local_sum_dy_xhat;
    }
    __syncthreads();
    if (warp_id == 0) {
        local_sum_dy = (lane < num_warps) ? s1[lane] : 0.0f;
        local_sum_dy_xhat = (lane < num_warps) ? s2[lane] : 0.0f;
        local_sum_dy = warp_reduce_sum(local_sum_dy);
        local_sum_dy_xhat = warp_reduce_sum(local_sum_dy_xhat);
    }
    __shared__ float ss_dy, ss_dy_xhat;
    if (tid == 0) {
        ss_dy = local_sum_dy;
        ss_dy_xhat = local_sum_dy_xhat;
    }
    __syncthreads();

    float inv_cols = 1.0f / cols;

    // ---- 计算 dx，并累加 dgamma/dbeta ----
    for (int i = tid; i < cols; i += blockDim.x) {
        float xhat = (x_row[i] - mean) * rstd;

        // dx 公式: rstd * gamma * (dy - (1/N)*sum_dy - (1/N)*xhat*sum_dy_xhat)
        // 直觉: 梯度需要减去两个"均值修正项"，因为 mean 和 var 也依赖于所有 x
        dx_row[i] = rstd * gamma[i] *
            (dy_row[i] - inv_cols * ss_dy - inv_cols * xhat * ss_dy_xhat);

        // dgamma 和 dbeta 需要跨所有行累加
        // atomicAdd: 多个 Block（多行）同时写同一个 dgamma[i]，需要原子操作避免数据竞争
        atomicAdd(&dgamma[i], dy_row[i] * xhat);
        atomicAdd(&dbeta[i], dy_row[i]);
    }
}

// ============================================================
// C++ 封装函数 — 桥接 PyTorch Tensor 和 CUDA Kernel
//
// 这些函数做三件事:
//   1. 输入校验（在 GPU 上？是 float？是 contiguous？）
//   2. 分配输出 tensor
//   3. 调用 kernel，返回结果
// ============================================================

std::vector<torch::Tensor> layernorm_forward(
    torch::Tensor x,      // [rows, cols]
    torch::Tensor gamma,  // [cols]
    torch::Tensor beta,   // [cols]
    float eps)
{
    // 输入校验 — 这些检查在开发时能帮你快速定位问题
    TORCH_CHECK(x.is_cuda(), "x 必须在 GPU 上");
    TORCH_CHECK(gamma.is_cuda(), "gamma 必须在 GPU 上");
    TORCH_CHECK(beta.is_cuda(), "beta 必须在 GPU 上");
    TORCH_CHECK(x.is_contiguous(), "x 必须是 contiguous 的");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "目前只支持 float32");

    int rows = x.size(0);
    int cols = x.size(1);

    // 分配输出 tensor — PyTorch 内部调用 cudaMalloc
    auto y = torch::empty_like(x);
    auto mean = torch::empty({rows}, x.options());   // [rows]
    auto rstd = torch::empty({rows}, x.options());   // [rows]

    // Launch kernel: 每行一个 Block，每 Block 256 线程
    int block_size = 256;
    layernorm_forward_kernel<<<rows, block_size>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        rows, cols, eps
    );

    // 返回 y, mean, rstd — mean 和 rstd 会被 save_for_backward 保存
    return {y, mean, rstd};
}

std::vector<torch::Tensor> layernorm_backward(
    torch::Tensor dy,         // [rows, cols] 上游梯度
    torch::Tensor x,          // [rows, cols] 前向输入
    torch::Tensor gamma,      // [cols]
    torch::Tensor mean,       // [rows]
    torch::Tensor rstd)       // [rows]
{
    TORCH_CHECK(dy.is_cuda() && dy.is_contiguous());
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());

    int rows = x.size(0);
    int cols = x.size(1);

    auto dx = torch::empty_like(x);
    // dgamma 和 dbeta 要清零！因为 kernel 内用 atomicAdd 累加
    auto dgamma = torch::zeros_like(gamma);
    auto dbeta = torch::zeros_like(gamma);

    int block_size = 256;
    layernorm_backward_kernel<<<rows, block_size>>>(
        dy.data_ptr<float>(),
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        dx.data_ptr<float>(),
        dgamma.data_ptr<float>(),
        dbeta.data_ptr<float>(),
        rows, cols
    );

    return {dx, dgamma, dbeta};
}

// ============================================================
// pybind11 注册 — 让 Python 能 import custom_layernorm
//
// PYBIND11_MODULE 的第一个参数必须和 setup.py 中的 name 一致
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA)");
    m.def("backward", &layernorm_backward, "LayerNorm backward (CUDA)");
}
