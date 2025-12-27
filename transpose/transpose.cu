#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

// 定义分块大小，通常为 32 (对应 Warp Size)
#define TILE_DIM 32

// ==========================================
// 1. 基础版本 (Naive Transpose)
// ==========================================
// 问题：读取是合并的 (Coalesced)，但写入是非合并的 (Strided)。
// 因为写入时 out[x * M + y]，相邻线程改变 x，导致内存地址跳跃 M，带宽利用率极低。
__global__ void transposeNaive(float *d_out, float *d_in, int M, int N) {
    // x 和 y 代表输入矩阵中的列和行索引
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查
    if (y < M && x < N) {
        // 输入索引: y行 x列 (y * N + x) -> 连续读取，高效
        // 输出索引: x行 y列 (x * M + y) -> 跨步写入，低效
        d_out[x * M + y] = d_in[y * N + x];
    }
}

// ==========================================
// 2. 优化版本 (Shared Memory Optimized)
// ==========================================
// 策略：
// 1. 将 Global Memory 的一块数据合并读取到 Shared Memory (TILE)。
// 2. 在 Shared Memory 内部进行转置 (交换下标)。
// 3. 将 Shared Memory 的数据合并写入到 Global Memory。
// 4. 使用 Padding 消除 Shared Memory Bank Conflict。
__global__ void transposeShared(float *d_out, float *d_in, int M, int N) {
    // 声明共享内存
    // TILE_DIM + 1 是为了 Padding，避免读取列数据时发生 Bank Conflict
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // --- 阶段 1: 读取数据到 Shared Memory ---
    // 计算输入矩阵的坐标 (读哪个位置)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查并读取
    // 这里的读取是 Coalesced 的，因为相邻 threadIdx.x 读取相邻的 d_in 地址
    if (y < M && x < N) {
        tile[threadIdx.y][threadIdx.x] = d_in[y * N + x];
    }

    // 必须同步，确保整个 Block 的数据都加载到了 Shared Memory
    __syncthreads();

    // --- 阶段 2: 将数据写入 Global Memory ---
    // 关键点：我们需要计算在输出矩阵中的新坐标。
    // 原来的 Block(bx, by) 处理的是输入矩阵的块。
    // 转置后，它应该负责输出矩阵中 (by, bx) 位置的块。
    
    // new_x: 输出矩阵的列索引 (对应原来的行 blockIdx.y)
    // new_y: 输出矩阵的行索引 (对应原来的列 blockIdx.x)
    // 注意：这里我们交换了 blockIdx.y 和 blockIdx.x 的作用
    int new_x = blockIdx.y * blockDim.x + threadIdx.x;
    int new_y = blockIdx.x * blockDim.y + threadIdx.y;

    // 边界检查
    if (new_x < M && new_y < N) {
        // 写入 Global Memory
        // 1. d_out 索引: new_y * M + new_x
        //    因为 threadIdx.x 连续变化，new_x 也连续变化，所以写入是 Coalesced 的。
        // 2. 读取 Shared Memory: tile[threadIdx.x][threadIdx.y]
        //    这里交换了下标，实现了转置。
        d_out[new_y * M + new_x] = tile[threadIdx.x][threadIdx.y];
    }
}

// 辅助函数：CPU 验证
void verify(float *h_in, float *h_out, int M, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (abs(h_out[i * M + j] - h_in[j * N + i]) > 1e-5) {
                printf("Error at (%d, %d)\n", i, j);
                return;
            }
        }
    }
    printf("Result Verified: Success!\n");
}

int main() {
    int M = 2048; // 行
    int N = 2048; // 列
    size_t size = M * N * sizeof(float);

    // Host 内存分配
    float *h_in = (float*)malloc(size);
    float *h_out_naive = (float*)malloc(size);
    float *h_out_shared = (float*)malloc(size);

    // 初始化
    for (int i = 0; i < M * N; i++) h_in[i] = (float)i;

    // Device 内存分配
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // 数据传输 H2D
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // 定义 Grid 和 Block
    dim3 block(TILE_DIM, TILE_DIM);
    // 向上取整
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // 1. 运行 Naive 版本
    transposeNaive<<<grid, block>>>(d_out, d_in, M, N);
    cudaDeviceSynchronize();
    
    // 拷贝回结果
    cudaMemcpy(h_out_naive, d_out, size, cudaMemcpyDeviceToHost);
    printf("Naive Kernel: ");
    verify(h_in, h_out_naive, M, N);

    // 清空输出显存
    cudaMemset(d_out, 0, size);

    // 2. 运行 Optimized 版本
    transposeShared<<<grid, block>>>(d_out, d_in, M, N);
    cudaDeviceSynchronize();

    // 拷贝回结果
    cudaMemcpy(h_out_shared, d_out, size, cudaMemcpyDeviceToHost);
    printf("Shared Optimized Kernel: ");
    verify(h_in, h_out_shared, M, N);

    // 释放内存
    free(h_in); free(h_out_naive); free(h_out_shared);
    cudaFree(d_in); cudaFree(d_out);

    return 0;
}