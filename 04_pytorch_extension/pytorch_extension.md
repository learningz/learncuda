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
