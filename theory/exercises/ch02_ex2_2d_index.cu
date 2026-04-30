// ============================================================
// Ch2 练习 2: 2D Grid 索引打印
// 配合: theory/02_cuda_programming_model.md 练习 2
//
// 编译: nvcc -O2 -o ch02_ex2_2d_index ch02_ex2_2d_index.cu
// 运行: ./ch02_ex2_2d_index
//
// 启动一个 2D Grid + 2D Block 的小 kernel, 打印每个 Block 的 (0,0) 线程
// 的 row 和 col, 帮你直观理解 2D 索引映射。
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>

#define TILE 4

__global__ void print_index(int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0 && threadIdx.y == 0 && row < M && col < N) {
        printf("Block(%d,%d) → row=%d, col=%d\n", blockIdx.x, blockIdx.y, row, col);
    }
}

int main() {
    const int M = 8, N = 12;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    printf("矩阵 %d×%d, Block %d×%d, Grid %d×%d\n\n",
           M, N, TILE, TILE, grid.x, grid.y);
    print_index<<<grid, block>>>(M, N);
    cudaDeviceSynchronize();

    printf("\n思考: blockIdx.x 对应列方向, blockIdx.y 对应行方向。\n");
    printf("为什么? 因为 threadIdx.x 连续的线程属于同一 Warp,\n");
    printf("让它们处理连续的列 → 内存地址连续 → 合并访问!\n");
    return 0;
}
