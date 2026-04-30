# Warp Divergence：分支指令在硬件上的真实行为

配合 `warp_divergence.cu` 阅读。


## 无分歧 vs 有分歧的 SASS 对比

### 无分歧 (以 Warp 为边界判断)

```cuda
if (warp_id % 2 == 0) { output[idx] = val * val + val; }
else                   { output[idx] = val * 0.5f - 1.0f; }
```

```
SASS (简化):
  ISETP.EQ P0, R_warp_id_mod2, 0 ;   // P0 = (warp_id % 2 == 0)
  @P0 BRA PATH_A ;
  // PATH_B:
  FMUL R2, R0, 0.5 ;
  FADD R2, R2, -1.0 ;
  BRA END ;
  PATH_A:
  FFMA R2, R0, R0, R0 ;    // val*val+val (FMA)
  END:
  STG.E [R4], R2 ;

硬件行为:
  Warp 0 (全是 warp_id=0): P0 全 true → 全走 PATH_A → 无分歧!
  Warp 1 (全是 warp_id=1): P0 全 false → 全走 PATH_B → 无分歧!
  
  每个 Warp 内 32 线程走同一路径 → 正常速度执行。
  不同 Warp 走不同路径 → 完全没问题 (它们本来就是独立调度的)。
```


### 50% 分歧 (奇偶线程判断)

```cuda
if (threadIdx.x % 2 == 0) { output[idx] = val * val + val; }
else                       { output[idx] = val * 0.5f - 1.0f; }
```

```
SASS 行为 (分歧!):
  ISETP.EQ P0, R_tid_mod2, 0 ;
  
  同一 Warp 内:
    Thread 0:  P0 = true  → 想走 PATH_A
    Thread 1:  P0 = false → 想走 PATH_B
    Thread 2:  P0 = true  → 想走 PATH_A
    ...
    → 16 个线程要 A, 16 个要 B → 分歧!

  GPU 硬件的处理方式 (Pre-Volta):
    Step 1: 执行 PATH_A, Active Mask = 0x55555555 (偶数线程)
            Thread 0,2,4,...30 的 FMA 正常计算
            Thread 1,3,5,...31 被 mask, ALU 空转
    
    Step 2: 执行 PATH_B, Active Mask = 0xAAAAAAAA (奇数线程)
            Thread 1,3,5,...31 的 FMUL+FADD 正常计算
            Thread 0,2,4,...30 被 mask, ALU 空转
    
    Step 3: 汇合, Active Mask = 0xFFFFFFFF (全部恢复)
  
  代价: PATH_A 和 PATH_B 串行执行!
    时间 = time(A) + time(B), 而不是 max(time(A), time(B))
    如果两条路径等长 → 性能减半!
```


### 无分支算术 (编译器优化)

```cuda
output[idx] = (threadIdx.x % 2 == 0) ? path_a : path_b;
```

```
编译器可能优化为谓词指令 (无分支跳转!):

  FFMA R2, R0, R0, R0 ;       // 先算 path_a (全 32 线程都算)
  FMUL R3, R0, 0.5 ;          // 先算 path_b (全 32 线程都算)
  FADD R3, R3, -1.0 ;
  @P0 MOV R4, R2 ;            // 如果 P0=true: 用 path_a
  @!P0 MOV R4, R3 ;           // 如果 P0=false: 用 path_b
  STG.E [R6], R4 ;

  两条路径都执行了! 但没有分支跳转 → 没有 Active Mask 切换!
  只是最后用谓词选择 (SEL/MOV) 哪个结果写入。
  
  代价: 计算了两条路径 (有冗余), 但避免了分歧的串行化开销。
  当路径很短时 (1-2 条指令), 这比真正的分歧更快。
  当路径很长时, 冗余计算的代价超过分歧 → 编译器会选择真正的分支。
```


## 什么时候分歧代价大, 什么时候小

```
┌───────────────────────────┬──────────────────────────────┐
│ 场景                      │ 分歧代价                      │
├───────────────────────────┼──────────────────────────────┤
│ 两条路径都只有 1-3 条指令  │ 几乎为零 (编译器谓词化)       │
│ 两条路径各 10+ 条指令      │ 接近 2× 减速                 │
│ 一条路径 100 条, 另一条 1  │ 不是减半, 而是慢 101/100 ≈ 1% │
│ 分歧以 Warp 为边界        │ 零代价 (不同 Warp 独立调度)   │
│ 循环中的分歧 (不同迭代数)  │ 最坏: 最慢线程决定整个 Warp    │
└───────────────────────────┴──────────────────────────────┘
```


## 练习题

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_warp_uniform_level1.cu](./exercises/ex1_warp_uniform_level1.cu) | 用 Warp-uniform 条件消除分歧 | 让分支以 Warp 为单位变化（只填 kernel） |

```bash
nvcc -O2 --extended-lambda -o ex1_warp_uniform_level1 ex1_warp_uniform_level1.cu
```
