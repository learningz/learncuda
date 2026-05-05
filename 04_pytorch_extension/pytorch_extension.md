# PyTorch 自定义算子：从 .cu 到 Python 可调用的完整链路

配合 `gelu_cuda.cu`, `setup.py`, `test_gelu.py` 阅读。


## 编译链路 — pip install 背后发生了什么

```
当你运行 pip install -e . 时:

setup.py
  │ 告诉 setuptools: "有一个 CUDAExtension, 源文件是 gelu_cuda.cu"
  │
  ▼
torch.utils.cpp_extension.BuildExtension
  │
  ├── 对 gelu_cuda.cu 调用 nvcc:
  │   nvcc -c gelu_cuda.cu -o gelu_cuda.o
  │     ├── 分离 Host 代码 (__global__ 之外的 C++) 和 Device 代码 (__global__)
  │     ├── Device 代码 → PTX → SASS → .cubin (嵌入 .o 中)
  │     └── Host 代码 → 系统 C++ 编译器 → .o
  │
  ├── 链接:
  │   g++ gelu_cuda.o -shared -o custom_gelu.cpython-310-x86_64-linux-gnu.so
  │     ├── 链接 libtorch.so (PyTorch 的 C++ 库)
  │     ├── 链接 libcudart.so (CUDA Runtime)
  │     └── 生成 Python 可导入的 .so 文件
  │
  └── 生成的 .so 文件包含:
      ├── pybind11 导出的 Python 接口 (forward, backward)
      ├── C++ 封装函数 (gelu_forward, gelu_backward)
      └── 嵌入的 GPU 二进制 (kernel 的 SASS/PTX)
```


## 运行时 — Python 调用到 GPU 执行

```
Python: output = custom_gelu.forward(input_tensor)

调用链:
  Python interpreter
    │ pybind11 将 Python Tensor 对象转换为 C++ torch::Tensor
    ▼
  gelu_forward() (C++ 函数, 在 .so 中)
    │ TORCH_CHECK: 检查 tensor 在 GPU 上、是 contiguous、是 float32
    │ torch::empty_like(input): 调用 PyTorch 的内存分配 (底层: cudaMalloc)
    │ input.data_ptr<float>(): 获取 tensor 底层的 GPU 显存指针
    ▼
  gelu_forward_kernel<<<grid, block>>>(ptr_in, ptr_out, n)
    │ CUDA Runtime 将 launch 命令写入 Command Buffer
    │ GPU 开始执行 kernel (CPU 立即返回)
    ▼
  Python: 返回 output tensor (GPU 上的计算可能还在进行!)

关键点:
  - Python → C++ → CUDA kernel 的调用链是同步的 (函数调用层面)
  - 但 kernel launch 本身是异步的 (GPU 在后台执行)
  - 只有当你 .cpu() 或 print() 时才会触发同步等待
```


## C++ 封装中的关键调用 — 为什么每次都要写这些

看 `gelu_cuda.cu` 中的 C++ 封装函数，你会发现几个固定套路：

```cpp
torch::Tensor gelu_forward(torch::Tensor input) {
    // 套路 1: 输入验证
    TORCH_CHECK(input.device().is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    // 套路 2: 分配输出
    auto output = torch::empty_like(input);

    // 套路 3: 获取裸指针
    int n = input.numel();
    float *in_ptr = input.data_ptr<float>();
    float *out_ptr = output.data_ptr<float>();

    // 套路 4: launch kernel
    gelu_forward_kernel<<<grid, block>>>(in_ptr, out_ptr, n);

    return output;
}
```

**为什么每个都要检查？**

```
TORCH_CHECK(input.is_contiguous(), ...):
  为什么需要? PyTorch 的 tensor 可能不是连续存储的:
    x = torch.randn(4, 256)        // 连续 ✓
    y = x.transpose(0, 1)          // y 不连续! 底层是 x 的转置视图
    z = x[0:2, :]                  // 连续 ✓ (slice 产生连续的视图)
    w = x[:, ::2]                   // 不连续! 步长是 2
  
  如果 tensor 不连续:
    .data_ptr<float>() 返回的是实际存储的首地址
    但元素之间的间距不是 sizeof(float), 而是 stride
    → kernel 里用 input[i] 访问会读到错误的值!
    → 甚至可能越界 (因为 numel() 和实际存储大小可能不同)
  
  修复方法: 调用 .contiguous() 会创建一个连续副本:
    input = input.contiguous();   // 如果不连续就复制一份, 连续就什么都不做
  
  为什么不在函数里自动调?: 因为 .contiguous() 有额外开销 (可能需要 cudaMemcpy 一次)。
  所以惯例是: 检查 + 报错, 让调用者决定是否 contiguous。

TORCH_CHECK(input.scalar_type() == torch::kFloat32, ...):
  原因: CUDA kernel 用 float* 指针访问数据。
  如果 Python 端传了个 half (FP16) tensor → 指针类型不匹配 → 读到垃圾数据。
  生产代码通常用模板 (template<typename T>) 支持多种精度, 让编译器为每种类型生成不同的 kernel。

TORCH_CHECK(input.device().is_cuda(), ...):
  原因: CUDA kernel 只能访问 GPU 显存。如果输入还在 CPU 上 → .data_ptr() 返回的是 CPU 指针 → GPU 访问时崩溃。
  
为什么调用 data_ptr<float>()?
  这是获取 tensor 底层 GPU 显存指针的唯一方式。
  注意: 返回的是 float*, 不是 torch::Tensor。
  一旦拿到裸指针, 就没有任何 shape/stride/device 信息了 → 前面的 TORCH_CHECK 是最后的安全网。
```

**完整的安全模板**:

```cpp
// 生产级的输入处理 (支持任意输入, 自动 contiguous)
torch::Tensor gelu_forward_safe(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "CUDA tensor required");
    
    // 如果输入不是 contiguous, 自动复制一份
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    
    // 如果是 FP16 输入, 先转 FP32 (或支持模板生成 FP16 kernel)
    input = input.to(torch::kFloat32);
    
    auto output = torch::empty_like(input);
    int n = input.numel();
    
    // 安全: input 一定是 contiguous float32 CUDA tensor
    gelu_forward_kernel<<<(n+255)/256, 256>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
    
    return output;
}
```

## autograd.Function — 反向传播怎么接入

```
test_gelu.py 中的 CustomGELU:

class CustomGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)    # ← 告诉 autograd: 反向时需要 input
        return custom_gelu.forward(input)
    
    @staticmethod  
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors    # ← 取出前向保存的 input
        return custom_gelu.backward(grad_output, input)

当你调用 loss.backward() 时:
  PyTorch autograd 引擎自动构建反向图
  → 到达 CustomGELU 节点 → 调用 backward()
  → 调用我们的 CUDA kernel: gelu_backward_kernel
  → grad_input 返回给上游节点继续反向传播

ctx.save_for_backward(input) 的硬件含义:
  input tensor 的 GPU 显存不会被释放 (引用计数 +1)
  → 反向传播时才能访问前向的输入数据
  → 这就是训练比推理用更多显存的原因之一!
```


## 练习题

完成 `gelu_cuda.cu` 的阅读后，可以在 [`exercises/`](./exercises/) 目录下做以下练习：

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_sigmoid/](./exercises/ex1_sigmoid/) | Sigmoid forward | 换一个激活函数写 kernel + C++ 封装（只填 kernel） |
| [ex2_leaky_relu/](./exercises/ex2_leaky_relu/) | LeakyReLU forward + backward | 链式法则求导 + 两个 kernel（只填 kernel） |

每个练习是独立的 PyTorch 扩展项目，安装方式：

```bash
cd exercises/ex1_sigmoid
pip install -e .
python test_sigmoid.py
```
