#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>



using namespace std;
using Complex = thrust::complex<float>;

string address = "./frequency_10M/";
string F_address = "./F_recon_10M/";
string para;
string duration = "frequency10M";  // 第几个周期的uvw
string sufix = ".txt";
// const int amount = 1;


void writeToFile(const thrust::device_vector<Complex>& device_vector, const std::string& filename) {
    // 将数据从设备内存复制到主机内存
    std::vector<Complex> host_vector(device_vector.size());
    thrust::copy(device_vector.begin(), device_vector.end(), host_vector.begin());
    // 打开文件
    std::ofstream file(filename);
    if (file.is_open()) {
        // 按照指定格式写入文件
        for(const Complex& value : host_vector)
        {
            // file << value.real() << " " << value.imag() << std::endl;
            file << value.real() << std::endl;
        }
    }
    // 关闭文件
    file.close();
}

// 自定义的实部提取和加法
struct real_part_extractor {
    __host__ __device__
    float operator()(const thrust::complex<float>& c) const {
        return c.real();
    }
};


__global__
void Cart2SphKernel(Complex *u, Complex *v, Complex *w, Complex *theta, Complex *phi, Complex *r, int uvw_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= uvw_index) return;

    r[idx] = Complex(sqrtf(u[idx].real() * u[idx].real() + v[idx].real() * v[idx].real() + w[idx].real() * w[idx].real()), 0);
    theta[idx] = Complex(atan2f(v[idx].real(), u[idx].real()), 0); // -pi to pi
    phi[idx] = Complex(asinf(w[idx].real() / r[idx].real()), 0); // -pi/2 to pi/2
}


__global__
void ShadeMatrix(Complex *u, Complex *v, Complex *w, Complex *theta, Complex *phi, Complex *r, Complex *shadeM, int amount, int RES, float mean_r, float r0, float dtheta, float dphi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= amount) return;

    theta[idx] += Complex(M_PI, 0);
    phi[idx] += Complex(M_PI / 2, 0);

    // 计算遮挡点，遮挡中心点为卫星的共轭点
    theta[idx] = Complex(2 * M_PI, 0) - theta[idx];
    phi[idx] = Complex(M_PI, 0) - phi[idx];

    // 计算遮挡点像素位置
    int thetaInd = roundf(theta[idx].real() / dtheta);
    int phiInd = roundf(phi[idx].real() / dphi);
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

    shadeM[idx * 4] = Complex(thetaIndS, 0);
    shadeM[idx * 4 + 1] = Complex(thetaIndE, 0);
    shadeM[idx * 4 + 2] = Complex(phiIndS, 0);
    shadeM[idx * 4 + 3] = Complex(phiIndE, 0);
}


void save_Shade(const thrust::device_vector<Complex>& device_vector, const std::string& filename) {
    // 将数据从设备内存复制到主机内存
    std::vector<Complex> host_vector(device_vector.size());
    thrust::copy(device_vector.begin(), device_vector.end(), host_vector.begin());
    // 打开文件
    std::ofstream file(filename);
    if (file.is_open()) {
        // 按照指定格式写入文件
        for(size_t i = 0; i < host_vector.size(); i += 4) {
            if (i + 3 < host_vector.size()) {
                file << host_vector[i].real() << " " 
                     << host_vector[i + 1].real() << " " 
                     << host_vector[i + 2].real() << " " 
                     << host_vector[i + 3].real() << std::endl;
            }
        }
    }
    // 关闭文件
    file.close();
}


int main() {
    int RES = 20940;
    const int uvw_presize = 14400000;

    std::vector<Complex> cu(uvw_presize), cv(uvw_presize), cw(uvw_presize);
    thrust::device_vector<Complex> u(uvw_presize), v(uvw_presize), w(uvw_presize);

    // 读取uvw
    string address_uvw = address + "uvw1" + duration + sufix;
    cout << "address_uvw: " << address_uvw << std::endl;
    
    ifstream uvwFile(address_uvw);
    int uvw_index = 0;
    float u_point, v_point, w_point;
    string key_point;
    if (uvwFile.is_open()) {
        while (uvwFile >> u_point >> v_point >> w_point) {
            cu[uvw_index] = Complex(u_point, 0);
            cv[uvw_index] = Complex(v_point, 0);
            cw[uvw_index] = Complex(w_point, 0);
            uvw_index++;
        }
    }               
    cout << "uvw_index: " << uvw_index << endl; 

    // 复制到GPU上
    thrust::copy(cu.begin(), cu.begin() + uvw_index, u.begin());
    thrust::copy(cv.begin(), cv.begin() + uvw_index, v.begin());
    thrust::copy(cw.begin(), cw.begin() + uvw_index, w.begin());
    
    // 存储theta phi r
    thrust::device_vector<Complex> theta(uvw_index), phi(uvw_index), r(uvw_index);

    int blockSize;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Cart2SphKernel, 0, 0);
    int gridSize = floor(uvw_index + blockSize - 1) / blockSize;;  
    cout << "Cart2Sph Computing, blockSize: " << blockSize << endl;
    cout << "Cart2Sph Computing, girdSize: " << gridSize << endl;
    Cart2SphKernel<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(u.data()),
        thrust::raw_pointer_cast(v.data()),
        thrust::raw_pointer_cast(w.data()),
        thrust::raw_pointer_cast(theta.data()),
        thrust::raw_pointer_cast(phi.data()),
        thrust::raw_pointer_cast(r.data()),
        uvw_index
    );

    // 计算 r 的均值
    float sum_r = thrust::transform_reduce(r.begin(), r.end(), real_part_extractor(), 0.0f, thrust::plus<float>());
    float mean_r = sum_r / r.size();

    // 计算 r0 dtheta dphi
    float r0 = atanf(1737.74f / (mean_r + 1737.74f));
    float dtheta = 2 * M_PI / RES;
    float dphi = M_PI / RES; 

    // 存储shadeM
    thrust::device_vector<Complex> shadeM(uvw_index * 4);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ShadeMatrix, 0, 0);
    gridSize = floor(uvw_index + blockSize - 1) / blockSize;;  
    cout << "Shade Matrix Computing, blockSize: " << blockSize << endl;
    cout << "Shade Matrix Computing, girdSize: " << gridSize << endl;
    ShadeMatrix<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(u.data()),
        thrust::raw_pointer_cast(v.data()),
        thrust::raw_pointer_cast(w.data()),
        thrust::raw_pointer_cast(theta.data()),
        thrust::raw_pointer_cast(phi.data()),
        thrust::raw_pointer_cast(r.data()),
        thrust::raw_pointer_cast(shadeM.data()),
        uvw_index,
        RES,
        mean_r,
        r0,
        dtheta,
        dphi
    );

    cudaDeviceSynchronize();

    // 存储结果
    save_Shade(shadeM, "shade10Mperiod1.txt");
    
    return 0;
}