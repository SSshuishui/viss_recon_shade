#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>

#define RES 512  // Example resolution, adjust as necessary

__device__
void cart2sph(float u, float v, float w, float* theta, float* phi, float* r) {
    *r = sqrtf(u * u + v * v + w * w);
    *theta = atan2f(v, u); // -pi to pi
    *phi = asinf(w / (*r)); // -pi/2 to pi/2
}


__global__
void ShadeMatrixKernel(const float* u_grid, const float* v_grid, const float* w_grid, float* shadeM, int amount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= amount) return;

    float theta, phi, r;
    cart2sph(u_grid[idx], v_grid[idx], w_grid[idx], &theta, &phi, &r);

    theta += M_PI;
    phi += M_PI / 2;
    float r0 = atanf(1737.74f / (r + 1737.74f));
    float dtheta = 2 * M_PI / RES;
    float dphi = M_PI / RES;

    theta = 2 * M_PI - theta;
    phi = M_PI - phi;

    int thetaInd = roundf(theta / dtheta);
    int phiInd = roundf(phi / dphi);
    int thetaR = roundf(r0 / dtheta);
    int phiR = roundf(thetaR / 2);

    int thetaIndS = thetaInd - thetaR;
    int thetaIndE = thetaInd + thetaR;
    int phiIndS = phiInd - phiR;
    int phiIndE = phiInd + phiR;

    if (thetaIndS <= 0) thetaIndS += RES;
    if (thetaIndE > RES) thetaIndE -= RES;
    if (phiIndS <= 0) phiIndS += RES;
    if (phiIndE > RES) phiIndE -= RES;

    shadeM[idx * 4] = thetaIndS;
    shadeM[idx * 4 + 1] = thetaIndE;
    shadeM[idx * 4 + 2] = phiIndS;
    shadeM[idx * 4 + 3] = phiIndE;
}

std::vector<std::vector<int>> ShadeMatrix(const std::vector<float>& u_grid, const std::vector<float>& v_grid, const std::vector<float>& w_grid, int amount) {
    float* d_u_grid;
    float* d_v_grid;
    float* d_w_grid;
    float* d_shadeM;
    size_t size = amount * sizeof(float);

    cudaMalloc(&d_u_grid, size);
    cudaMalloc(&d_v_grid, size);
    cudaMalloc(&d_w_grid, size);
    cudaMalloc(&d_shadeM, amount * 4 * sizeof(float));

    cudaMemcpy(d_u_grid, u_grid.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_grid, v_grid.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_grid, w_grid.data(), size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (amount + blockSize - 1) / blockSize;
    ShadeMatrixKernel<<<numBlocks, blockSize>>>(d_u_grid, d_v_grid, d_w_grid, d_shadeM, amount);

    cudaDeviceSynchronize();

    std::vector<float> h_shadeM(amount * 4);
    cudaMemcpy(h_shadeM.data(), d_shadeM, amount * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_u_grid);
    cudaFree(d_v_grid);
    cudaFree(d_w_grid);
    cudaFree(d_shadeM);

    std::vector<std::vector<int>> result(amount, std::vector<int>(4));
    for (int i = 0; i < amount; ++i) {
        result[i][0] = static_cast<int>(h_shadeM[i * 4]);
        result[i][1] = static_cast<int>(h_shadeM[i * 4 + 1]);
        result[i][2] = static_cast<int>(h_shadeM[i * 4 + 2]);
        result[i][3] = static_cast<int>(h_shadeM[i * 4 + 3]);
    }

    return result;
}

int main() {
    // Example input
    std::vector<float> u_grid = { /* fill in with your data */ };
    std::vector<float> v_grid = { /* fill in with your data */ };
    std::vector<float> w_grid = { /* fill in with your data */ };

    int amount = u_grid.size(); // Assuming all grids have the same size

    auto result = ShadeMatrix(u_grid, v_grid, w_grid, amount);

    // Print the result
    for (const auto& row : result) {
        std::cout << row[0] << " " << row[1] << " " << row[2] << " " << row[3] << std::endl;
    }

    return 0;
}