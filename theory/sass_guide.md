# SASS 阅读手册 — GPU 机器码快速解码

**难度**: ⭐⭐⭐ 专家
**前置**: 能写 CUDA kernel，看过至少一个 ncu profile 输出

教程中几乎每个 module 都会出现 SASS 指令（`LDG.E`、`FFMA`、`SHFL.BFLY`、`BAR.SYNC` 等），但从未系统解释过怎么读它们。本手册补上这个空缺。


## 快速入门：3 分钟看懂一段 SASS

### 第一步：生成 SASS

```bash
# 编译时保留调试信息 (关联源码行)
nvcc -O2 -lineinfo -o my_kernel my_kernel.cu

# 导出 SASS
cuobjdump -sass my_kernel > my_kernel.sass
```

### 第二步：看懂一条指令

```
/*0120*/            LDS.128     R8,      [R14] ;
 │                  │  │        │         │
 字节偏移            │  操作类型   目标寄存器  源操作数 (地址)
                   操作码 (助记符)
                   
/*0120*/ LDS.128 R8, [R14] ;
→ 从 Shared Memory 的地址 R14 处，加载 128-bit (=4 floats) 到寄存器 R8-R11
```

### 第三步：SASS 操作数类型速查

```
R0-R255         通用寄存器 (每个线程私有, 32-bit)
                一条指令可以访问连续的多个寄存器: R8-R11 表示 4 个 32-bit 寄存器

P0-P7           谓词寄存器 (1-bit, true/false)
                @P0 FFMA ...  → 只有 P0=true 的线程才执行

UR0-UR63        统一寄存器 (Uniform Register, 整个 Warp 共享)
                用于存储 Warp 内所有线程相同的值 (如循环边界、地址基址)

c[0][0x160]     常量内存/立即数
                c[bank][offset] — 编译器放入 Constant Bank 的常量

[R14]           间接寻址: 读/写地址存储在寄存器 R14 中
[R14+0x80]      基址+偏移: 地址 = R14 + 0x80

[UR4+0x10]      统一寄存器+偏移寻址

RZ              零寄存器 (Always 0)，用于清零或生成常量
```


## 完整指令格式

```
(@Px) OPCODE.TYPE Rdst, Ra, Rb, Rc ;
 │      │      │     │    │   │
 │      │      │     目标  源操作数
 │      │     数据类型 (位宽: .F32=float32, .U32=uint32, .128=128bit)
 │     助记符 (如 LDG=Global Load, FFMA=Fused Multiply-Add)
谓词掩码 (可选, @P0 表示只在 P0=true 的线程执行)
```

数据类型后缀：
```
.F32 / .F16 / .F64   浮点 32/16/64 bit
.U32 / .U16 / .U8    无符号整数
.S32 / .S16 / .S8    有符号整数
.128 / .64 / .32     原始位宽 (不指定类型)
```


## 常见指令族速查表

### 内存操作

| 指令 | 含义 | 延迟 | 说明 |
|------|------|------|------|
| `LDG.E.32 R, [addr]` | 从 Global Memory 加载 32-bit | ~500 cyc | `.E` = 绕过 L1 (或使用 L1) |
| `STG.E.32 [addr], R` | 存储 32-bit 到 Global Memory | ~400 cyc | |
| `LDG.128 R, [addr]` | 加载 128-bit (float4) | ~500 cyc | 一次搬 16B |
| `LDS.32 R, [addr]` | 从 Shared Memory 加载 32-bit | ~5 cyc | 不经过 L1/L2 |
| `STS.32 [addr], R` | 存储 32-bit 到 Shared Memory | ~5 cyc | |
| `LDL.32 R, [addr]` | 从 Local Memory 加载 | ~200 cyc | 寄存器溢出到 L1 Cache |
| `STL.32 [addr], R` | 存储到 Local Memory | ~200 cyc | |

> **注意**: LDG 延迟 ~500 cycles 但实际隐藏后接近 ~30 cycles/元素，因为多个 Warp 交替执行时延迟被重叠。

### 算术操作

| 指令 | 含义 | 延迟 | 说明 |
|------|------|------|------|
| `FFMA R, a, b, c` | R = a×b + c (Fused Multiply-Add) | ~4 cyc | 每条指令 2 FLOP |
| `FMUL R, a, b` | R = a × b | ~4 cyc | |
| `FADD R, a, b` | R = a + b | ~4 cyc | |
| `FMNMX R, a, b` | R = max(a, b) | ~4 cyc | Float MiN/MaX |
| `MUFU.EX2 R, a` | R = 2^a (exp2) | ~28 cyc | SFU 执行, 吞吐只有 FP32 的 1/4 |
| `MUFU.RCP R, a` | R = 1/a (reciprocal) | ~28 cyc | SFU 执行 |
| `IADD3 R, a, b, c` | R = a + b + c | ~4 cyc | 整数三操作数加法 (常用于地址计算) |
| `ISETP.LE P, a, b` | P = (a <= b) | ~4 cyc | 整数比较, 结果存谓词 |

### 数据移动

| 指令 | 含义 | 延迟 | 说明 |
|------|------|------|------|
| `MOV R, a` | R = a | ~1 cyc | 寄存器到寄存器 |
| `S2R R, SR_TID.X` | 读特殊寄存器到通用寄存器 | ~4 cyc | SR_TID.X = threadIdx.x |
| `PRMT R, a, b, mask` | 字节置换 (Permute) | ~4 cyc | 对 a 和 b 的字节按 mask 重排 |
| `LOP3.LUT R, a, b, c, lut` | 3 输入按 LUT 做逻辑操作 | ~4 cyc | |

### Warp 通信 (Shuffle)

| 指令 | 含义 | 延迟 | CUDA 对应 |
|------|------|------|-----------|
| `SHFL.DOWN R, a, offset, mask` | 从 lane+offset 取 a | ~1 cyc | `__shfl_down_sync` |
| `SHFL.UP R, a, offset, mask` | 从 lane-offset 取 a | ~1 cyc | `__shfl_up_sync` |
| `SHFL.BFLY R, a, offset, mask` | 从 lane^offset 取 a | ~1 cyc | `__shfl_xor_sync` (reduce 常用) |
| `SHFL.IDX R, a, lane, mask` | 从 lane 取 a (广播) | ~1 cyc | `__shfl_sync` |

> `mask` = 0x1f = 0b11111 = 5 bits = 32 lanes 全部参与。

### 同步与控制流

| 指令 | 含义 | 延迟 | 说明 |
|------|------|------|------|
| `BAR.SYNC 0` | 同步 Barrier #0 | ~25 cyc | `__syncthreads()` |
| `@P0 BRA target` | 如果 P0 为真, 跳转到 target | ~4 cyc | 条件分支 (Warp Divergence 的源头!) |
| `BRA target` | 无条件跳转 | ~1 cyc | |
| `EXIT` | kernel 结束 | ~4 cyc | |
| `BPT.TRAP 0` | 断点 (可用于调试) | — | 触发 CUDA debugger |

### Tensor Core

| 指令 | 含义 | 延迟 | 说明 |
|------|------|------|------|
| `HMMA.16816.F32 R, a, b, c` | FP16 MMA: 16×8×16 → FP32 累加 | ~8 cyc | 4096 FLOP / 指令! |

> HMMA 命名规则: `.16816` = M16 × N8 × K16 = 输出 16 行 × 8 列, 输入沿 K 方向 16 元素深。


## 实战：从 CUDA 代码到 SASS 的逐行映射

以最简单的 SAXPY kernel 为例：

```cuda
__global__ void saxpy(float a, float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
```

编译后的 SASS（简化，关键行标注）：

```
// 1. 读取线程索引
S2R   R0, SR_CTAID.X       ; R0 = blockIdx.x
S2R   R2, SR_TID.X          ; R2 = threadIdx.x

// 2. blockDim 在常量内存
ULDC  UR4, c[0x0][0x160]   ; UR4 = blockDim.x (存在 Uniform Register, 所有线程共用)

// 3. 计算 i = blockIdx.x * blockDim.x + threadIdx.x
IMAD  R2, R0, UR4, R2      ; R2 = R0 * UR4 + R2  (整数乘加)

// 4. 读取 x[i] 和 y[i]
ISETP.GE P0, R2, c[0x0][0x168], PT  ; P0 = (i >= n) ?
@P0  EXIT                           ; 如果 i >= n, 直接退出

SHL   R4, R2, 0x2          ; R4 = i * 4 (因为 sizeof(float) = 4)

LDG.E R6, [R_x + R4]        ; R6 = x[i]   → 加载延迟 ~500 cycles!
LDG.E R8, [R_y + R4]        ; R8 = y[i]   → 同时发射, 重叠延迟!

// 5. 计算 a * x[i] + y[i]
FFMA  R6, R6, c[0x0][0x170], R8  ; R6 = R6 * a + R8  (Fused Multiply-Add)

// 6. 写回
STG.E [R_y + R4], R6        ; y[i] = 结果

// 7. 结束
EXIT
```

**逐行理解要点**：

1. **地址计算绕不开的 `SHL R4, R2, 0x2`**：`i * sizeof(float)` = `i << 2` (左移 2 位 = 乘 4)。

2. **`c[0x0][0x160]` — 常量从哪里来**：编译器把 `blockDim.x` 和 `n` 放进 GPU 的 Constant Bank。不需要你手动分配，编译器自动完成。

3. **`@P0 EXIT` — 谓词化退出**：`ISETP` 先生成谓词 P0，`@P0 EXIT` 只退出"越界"的线程。其他线程继续执行。没有 if/else 分支 → 无 Warp Divergence！

4. **两条 LDG 背靠背**：GPU 可以同时发射多条不依赖的加载指令。两条 LDG 之间没有数据依赖 → 第 2 条的等待和第 1 条的等待重叠。

5. **FFMA 的三操作数**：`FFMA R6, R6, c[0x0][0x170], R8` = R6 = R6 * a + R8。一条指令完成乘加。R6 同时作为源和目标（原地修改）→ 编译器节省了一个寄存器。


## 进阶：ncu 中的 SASS 控制码

ncu (Nsight Compute) 的源码视图会显示 SASS 指令旁边的控制码：

```
FFMA R4, R0, R2, R4    ;  control code: 1210
                         ││││
                         │││└─ Yield flag
                         ││└── Reuse flag (寄存器复用)
                         │└─── Wait flag (在等前面指令的结果)
                         └──── Stall count (预测要等多少 cycles)
```

这些控制码帮你理解**这条指令为什么慢**：
- **Stall count** 高 → 操作数依赖前一条耗时长的指令（如 LDG）
- **Reuse flag** → 目标寄存器会立即被下一条指令读取（编译器优化了）
- **Wait flag** → 指令必须等 scoreboard 清掉依赖

> 详细解读需要 ncu 的 Source Counters 视图 — 这已超出快速手册范围。但记住关键点：**不要关心每一条 Stall，找出占 Stall 最多的那 1-2 条指令**。


## 如何用 SASS 诊断性能问题

```
工作流:
  1. ncu --set full ./my_kernel     → 找到瓶颈指标
  2. ncu --set source ./my_kernel   → 定位到源码行
  3. cuobjdump -sass my_kernel      → 对热点行查 SASS

常见 SASS 层面的问题:
  Issue                          SASS 证据
  ─────                          ─────────
  Memory Bound                  大量 LDG/STG, Stall Scoreboard Long 高
  Compute Bound                 密集 FFMA, FP64 指令居多
  LD/ST pipeline 瓶颈           大量连续的 LDG.32 (而非 LDG.128), LD/ST Unit 忙
  Warp Divergence               @P0 BRA + 另一条 @!P0 BRA (两个分支)
  Bank Conflict                 LDS 地址 stride=32×4B → 全落同一 Bank
  BAR.SYNC 等待时间长           BAR.SYNC 后有大量 Stall Barrier
  寄存器溢出                    大量 LDL/STL (Local Memory = 寄存器溢出到 L1)
```


## 速记卡

```
编译: nvcc -O2 -lineinfo -o prog prog.cu
导出: cuobjdump -sass prog

最常见的 10 条指令:
  LDG.E   — 读全局内存 (慢, ~500cyc)
  STG.E   — 写全局内存
  LDS     — 读 Shared Memory (快, ~5cyc)
  STS     — 写 Shared Memory
  FFMA    — 乘加 (快, ~4cyc, 每指令 2 FLOP)
  FMUL    — 乘法
  FADD    — 加法
  SHFL    — Warp 通信 (~1cyc)
  BAR.SYNC— 同步 (~25cyc)
  BRA     — 跳转

寄存器: R0-R255 (32-bit), P0-P7 (谓词), UR (Uniform), RZ (零)
常量: c[bank][offset] — blockDim, gridDim, kernel 参数
地址: [R] = *(R), [R+offset] = *(R+offset), [UR+offset]
谓词: @P0 指令 → 只在 P0=true 的线程执行
```
