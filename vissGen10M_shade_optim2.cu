#include <cstdio>
#include <iostream>
#include <ctime>
#include <string>
#include <cmath>
#include <omp.h>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "error.cuh"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/binary_search.h>

#define _USE_MATH_DEFINES
#define EXP 0.0000000000

using namespace std;
using Complex = thrust::complex<float>;

// complexExp 函数的实现
__device__ thrust::complex<float> complexExp(const Complex &d) {
    float realPart = exp(d.real()) * cos(d.imag());
    float imagPart = exp(d.real()) * sin(d.imag());
    return thrust::complex<float>(realPart, imagPart);
}
// complexAbs 函数的实现
__device__ thrust::complex<float> ComplexAbs(const Complex &d) {
    // 复数的模定义为 sqrt(real^2 + imag^2)
    return thrust::complex<float>(sqrt(d.real() * d.real() + d.imag() * d.imag()));
}

struct timeval start, finish;
float total_time;

string address = "./frequency_10M/";
string lmn_address = "./lmn10M/";
string F_address = "./F_10M/";
string duration = "frequency10M";  // 第几个周期的uvw
string sufix = ".txt";

// 10 M
const int uvw_presize = 14400000;

// 定义常量
#define BLOCK_SIZE 128                     // 线程块大小
#define SHARED_MEM_SIZE BLOCK_SIZE         // 共享内存大小
#define MAX_THREADS_PER_BLOCK 1024        // GPU每个块的最大线程数


struct clip_functor {
    __host__ __device__
    float operator()(float x) const {
        return max(-1.0f, min(1.0f, x));
    }
};


// 定义计算可见度核函数, 验证一致
__global__ void visscal(
    int uvws_index, int lmnC_index, int res,
    const float* __restrict__ FF,           // 添加 const 和 __restrict__
    Complex* __restrict__ viss, 
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const int* __restrict__ shadeM1,
    const int* __restrict__ shadeM2,
    const int* __restrict__ shadeM3,
    const int* __restrict__ shadeM4,
    const float* __restrict__ NXq,
    const int* __restrict__ countLoc,
    const Complex I1,
    const Complex CPI,
    const Complex zero,
    const Complex two, 
    const float dl,
    const float dm,
    const float dn)
{
    const int uvws_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (uvws_ >= uvws_index) return;  // 改用提前返回的方式

    // 预加载频繁使用的数据到寄存器
    const float inv_dl = 1.0f / dl;
    const float inv_dm = 1.0f / dm;
    const float inv_dn = 1.0f / dn;
    const float u_val = u[uvws_] * inv_dl;
    const float v_val = v[uvws_] * inv_dm;
    const float w_val = w[uvws_] * inv_dn;
    // 预加载遮挡相关的数据
    const int shade_m1 = shadeM1[uvws_];
    const int shade_m2 = shadeM2[uvws_];
    const int shade_m3 = shadeM3[uvws_];
    const int shade_m4 = shadeM4[uvws_];

    // 初始化累加器
    Complex acc = zero;
    int start_idx = 0;
    // 保持原有的循环结构
    for (int lmnC_ = 0; lmnC_ < lmnC_index; ++lmnC_) {
        const int current_count = countLoc[lmnC_];
        float sumReal = 0;

        // 内层循环计算 sumReal
        for (int con = 0; con < current_count; con++) {
            const long long locViss = NXq[con + start_idx];
            float addFF = FF[locViss];

            // 优化遮挡检查
            for (int lo = 0; lo >= shade_m3 && lo <= shade_m4; ++lo) {
                if (locViss >= lo*res+shade_m1+1 && locViss < lo*res+shade_m2) {
                    addFF = 240;
                }
            }
            sumReal += addFF;
        }
        
        start_idx += current_count;
        const float C_tmp = sumReal / current_count;

        // 计算相位
        const float phase = u_val * l[lmnC_] + v_val * m[lmnC_] + w_val * (n[lmnC_] - 1.0f);
        const Complex exp_val = complexExp((zero - I1) * two * CPI * Complex(phase, 0.0f));
        acc += Complex(C_tmp, 0.0f) * exp_val;  
    }

    // 计算最终的复指数因子并存储结果
    const Complex final_exp = complexExp((zero - I1) * two * CPI * Complex(w_val, 0.0f));
    viss[uvws_] = acc * final_exp;
}


void launch_visscal(
    const int uvws_index,
    const int lmnC_index,
    const int res,
    Complex* d_viss,
    const float* d_FF,
    const float* d_u,
    const float* d_v,
    const float* d_w,
    const float* d_l,
    const float* d_m,
    const float* d_n,
    const int* d_shadeM1,
    const int* d_shadeM2,
    const int* d_shadeM3,
    const int* d_shadeM4,
    const float* d_NXq,
    const int* d_countLoc,
    const Complex I1,
    const Complex CPI,
    const Complex zero,
    const Complex two,
    const float dl,
    const float dm,
    const float dn)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, visscal, 0, 0);
    int blocksPerGrid = floor(uvws_index + threadsPerBlock - 1) / threadsPerBlock;

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动核函数
    visscal<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        uvws_index, lmnC_index, res,
        d_FF, 
        d_viss, 
        d_u, 
        d_v, 
        d_w, 
        d_l, 
        d_m, 
        d_n, 
        d_shadeM1, 
        d_shadeM2, 
        d_shadeM3, 
        d_shadeM4, 
        d_NXq, 
        d_countLoc, 
        I1, CPI, zero, two, 
        dl, dm, dn
    );    

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 销毁流
    cudaStreamDestroy(stream);
}


// 定义图像反演核函数  验证正确
__global__ void imagerecon(
    const int uvw_index,
    const int lmnC_index,
    const int res,
    Complex* __restrict__ F,                    
    const Complex* __restrict__ viss,           
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const float* __restrict__ uvwFrequencyMap,
    const float* __restrict__ thetaP0,          // 添加 __restrict__
    const float* __restrict__ phiP0,
    const float* __restrict__ dtheta,
    const float* __restrict__ dphi,
    const Complex I1,                    
    const Complex CPI,
    const Complex zero,
    const Complex two,
    const float dl,
    const float dm,
    const float dn)
{
    // 声明共享内存
    __shared__ float s_u[SHARED_MEM_SIZE];
    __shared__ float s_v[SHARED_MEM_SIZE];
    __shared__ float s_w[SHARED_MEM_SIZE];
    __shared__ float s_uvwFreq[SHARED_MEM_SIZE];
    __shared__ Complex s_viss[SHARED_MEM_SIZE];
    __shared__ float s_thetaP0[SHARED_MEM_SIZE];
    __shared__ float s_phiP0[SHARED_MEM_SIZE];
    __shared__ float s_dtheta[SHARED_MEM_SIZE];
    __shared__ float s_dphi[SHARED_MEM_SIZE];

    const int lmnC_ = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    if (lmnC_ >= lmnC_index) return;

    // 预计算常量
    const Complex amount(uvw_index, 0.0f);
    const float inv_dl = 1.0f / dl;
    const float inv_dm = 1.0f / dm;
    const float inv_dn = 1.0f / dn;
    const float l_val = l[lmnC_] * inv_dl;
    const float m_val = m[lmnC_] * inv_dm;
    const float n_val = n[lmnC_] * inv_dn;

    // 预计算 phiP 和 thetaP
    const float phiP = floorf(lmnC_ / res);
    const float thetaP = lmnC_ - phiP * res;

    // 使用复数累加器
    Complex acc = zero;

    // 使用共享内存分块处理数据
    for (int base = 0; base < uvw_index; base += SHARED_MEM_SIZE) {
        const int current_chunk_size = min(SHARED_MEM_SIZE, uvw_index - base);
        
        // 协作加载数据到共享内存
        for (int i = tid; i < current_chunk_size; i += blockDim.x) {
            const int global_idx = base + i;
            s_u[i] = u[global_idx];
            s_v[i] = v[global_idx];
            s_w[i] = w[global_idx];
            s_uvwFreq[i] = uvwFrequencyMap[global_idx];
            s_viss[i] = viss[global_idx];
            s_thetaP0[i] = thetaP0[global_idx];
            s_phiP0[i] = phiP0[global_idx];
            s_dtheta[i] = dtheta[global_idx];
            s_dphi[i] = dphi[global_idx];
        }
        
        // 确保所有线程完成数据加载
        __syncthreads();

        // 处理当前块中的数据
        #pragma unroll 8
        for (int i = 0; i < current_chunk_size; ++i) {
            // 检查条件
            bool skip_calculation = (fabs(s_thetaP0[i] - thetaP) < s_dtheta[i] && fabs(s_phiP0[i] - phiP) < s_dphi[i]);

            if (!skip_calculation) {
                // 计算相位
                const float phase = s_u[i] * l_val + s_v[i] * m_val + s_w[i] * n_val;
                // 计算复指数
                const Complex exp_val = complexExp(I1 * two * CPI * Complex(phase, 0.0f));
                // 累加结果
                acc += s_uvwFreq[i] * s_viss[i] * exp_val;
            }
        }

        // 同步后再处理下一块数据
        __syncthreads();
    }
    // 归一化并存储结果
    F[lmnC_] = acc / amount;
}

// 启动函数
void launch_imagerecon(
    const int uvw_index,
    const int lmnC_index,
    const int res,
    Complex* d_F,
    Complex* d_viss,
    float* d_u,
    float* d_v,
    float* d_w,
    float* d_l,
    float* d_m,
    float* d_n,
    float* d_uvwFrequencyMap,
    float* d_thetaP0,
    float* d_phiP0,
    float* d_dtheta,
    float* d_dphi,
    const Complex I1,
    const Complex CPI,
    const Complex zero,
    const Complex two,
    const float dl,
    const float dm,
    const float dn)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, visscal, 0, 0);
    int blocksPerGrid = floor(lmnC_index + threadsPerBlock - 1) / threadsPerBlock;

    // 计算共享内存大小
    const size_t sharedMemSize = SHARED_MEM_SIZE * (
        sizeof(float) * 8 +    // s_u, s_v, s_w, s_uvwFreq, s_thetaP0, s_phiP0, s_dtheta, s_dphi
        sizeof(Complex)        // s_viss
    );

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 设置缓存配置以优化共享内存的使用
    cudaFuncSetCacheConfig(imagerecon, cudaFuncCachePreferShared);

    imagerecon<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream>>>(
        uvw_index, lmnC_index, res,
        d_F, 
        d_viss, 
        d_u, 
        d_v, 
        d_w, 
        d_l, 
        d_m, 
        d_n, 
        d_uvwFrequencyMap, 
        d_thetaP0, 
        d_phiP0, 
        d_dtheta, 
        d_dphi, 
        I1, CPI, zero, two, dl, dm, dn
    );

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    // 销毁流
    cudaStreamDestroy(stream);
}


int vissGen(int id, int RES, int start_period) 
{   
    cout << "res: " << RES << endl;
    int days = 34;  // 一共有多少个周期  15月 * 30天 / 14天/周期
    cout << "periods: " << days << endl;
    Complex I1(0.0, 1.0);
    float dl = 2 * RES / (RES - 1);
    float dm = 2 * RES / (RES - 1);
    float dn = 2 * RES / (RES - 1);
    Complex zero(0.0, 0.0);
    Complex two(2.0, 0.0);
    Complex CPI(M_PI, 0.0);

    gettimeofday(&start, NULL);
    int nDevices;
    // 设置节点数量（gpu显卡数量）
    CHECK(cudaGetDeviceCount(&nDevices));
    // 设置并行区中的线程数
    omp_set_num_threads(nDevices);
    cout << "devices: " << nDevices << endl;

    // 加载存储 l m n C的文件（对于不同的frequency不一样，只与frequency有关）
    string para, address_l, address_m, address_n, address_C, address_NX, address_FF;
    ifstream lFile, mFile, nFile, cFile, NXFile, FFFile;
    para = "l";
    address_l = lmn_address + para + sufix;
    lFile.open(address_l);
    cout << "address_l: " << address_l << endl;
    para = "m";
    address_m = lmn_address + para + sufix;
    mFile.open(address_m);
    cout << "address_m: " << address_m << endl;
    para = "n";
    address_n = lmn_address + para + sufix;
    nFile.open(address_n);
    cout << "address_n: " << address_n << endl;
    para = "C";
    address_C = lmn_address + para + sufix;
    cFile.open(address_C);
    cout << "address_C: " << address_C << endl;
    para = "NX";
    address_NX = lmn_address + para + sufix;
    NXFile.open(address_NX);
    cout << "address_NX: " << address_NX << endl;
    para = "FF";
    address_FF = lmn_address + para + sufix;
    FFFile.open(address_FF);
    cout << "address_FF: " << address_FF << endl;
    if (!lFile.is_open() || !mFile.is_open() || !nFile.is_open() || !cFile.is_open() || !NXFile.is_open() || !FFFile.is_open()) {
        std::cerr << "无法打开一个或多个文件：" << std::endl;
        if (!lFile.is_open()) std::cerr << "无法打开文件: " << address_l << std::endl;
        if (!mFile.is_open()) std::cerr << "无法打开文件: " << address_m << std::endl;
        if (!nFile.is_open()) std::cerr << "无法打开文件: " << address_n << std::endl;
        if (!cFile.is_open()) std::cerr << "无法打开文件: " << address_C << std::endl;
        if (!NXFile.is_open()) std::cerr << "无法打开文件: " << address_NX << std::endl;
        if (!FFFile.is_open()) std::cerr << "无法打开文件: " << address_FF << std::endl;
        return -1; 
    }
    int lmnC_index = 0;
    int NX_index = 0;
    lFile >> lmnC_index;  // 读取l的第一行的行数
    FFFile >> NX_index;  // 读取FF的第一行的行数
    cout << "lmnC index: " << lmnC_index << endl;
    cout << "NX index: " << NX_index << endl;

    std::vector<float> cl(lmnC_index), cm(lmnC_index), cn(lmnC_index), cc(lmnC_index);
    std::vector<float> cNX(NX_index), cFF(NX_index);
    for (int i = 0; i < lmnC_index && lFile.good() && mFile.good() && nFile.good() && cFile.good(); ++i) {
        lFile >> cl[i];
        mFile >> cm[i];
        nFile >> cn[i];
        cFile >> cc[i];
    }
    for (int i = 0; i < NX_index && NXFile.good() && FFFile.good(); ++i) {
        NXFile >> cNX[i];
        FFFile >> cFF[i];
    }
    lFile.close();
    mFile.close();
    nFile.close();
    cFile.close();
    NXFile.close();
    FFFile.close();

    // 加载存储 countLoc的文件
    string address_countLoc = lmn_address + "countLoc" + sufix;
    ifstream countLocFile(address_countLoc);
    cout << "address_countLoc: " << address_countLoc << endl;
    if (!countLocFile.is_open()) {
        std::cerr << "无法打开文件: " << address_countLoc << std::endl;
        return -1;
    }
    std::vector<int> cCountLoc(lmnC_index);
    for (int i = 0; i < lmnC_index && countLocFile.good(); ++i) {
        countLocFile >> cCountLoc[i];
    }
    countLocFile.close();
    // 加载存储 NXq的文件
    string address_NXq = lmn_address + "NXq" + sufix;
    ifstream NXqFile(address_NXq);
    cout << "address_NXq: " << address_NXq << endl;
    if (!NXqFile.is_open()) {
        std::cerr << "无法打开文件: " << address_NXq << std::endl;
        return -1;
    }
    std::vector<float> cNXq(NX_index);
    for (int i = 0; i < NX_index && NXqFile.good(); ++i) {
        NXqFile >> cNXq[i];
    }
    NXqFile.close();


    // 开启cpu线程并行
    // 一个线程处理1个GPU
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();  // 从 0 开始编号的并行执行线程
        cudaSetDevice(tid);
        std::cout << "Thread " << tid << " is running on device " << tid << std::endl;

        // 遍历所有开启的线程处理， 一个线程控制一个GPU 处理一个id*amount/total的块
        for (int p = tid+start_period; p < days; p += nDevices) 
        {
            cout << "for loop: " << p+1 << endl;

            // 将 l m n C NX 数据从cpu搬到GPU上        
            thrust::device_vector<float> C(cc.begin(), cc.end());
            thrust::device_vector<float> l(cl.begin(), cl.end());
            thrust::device_vector<float> m(cm.begin(), cm.end());
            thrust::device_vector<float> n(cn.begin(), cn.end());
            // 将 n 数据限制在 -1 到 1 之间
            thrust::transform(n.begin(), n.end(), n.begin(), clip_functor());

            // 将 countLoc NXq 数据从cpu搬到GPU上
            thrust::device_vector<float> NXq = cNXq;
            thrust::device_vector<int> countLoc = cCountLoc;

            thrust::device_vector<float> dNX = cNX;
            thrust::device_vector<float> dFF = cFF;
            thrust::device_vector<float> dFF2(NX_index);

            // 创建用来存储不同index中【u, v, w】
            std::vector<float> cu(uvw_presize), cv(uvw_presize), cw(uvw_presize);
            thrust::device_vector<float> u(uvw_presize), v(uvw_presize), w(uvw_presize);

            // 常见用来存储shadeM的 4个 变量
            std::vector<int> M1(uvw_presize), M2(uvw_presize), M3(uvw_presize), M4(uvw_presize);
            thrust::device_vector<int> shadeMat1(uvw_presize), shadeMat2(uvw_presize), shadeMat3(uvw_presize), shadeMat4(uvw_presize);

            // 创建存储uvw坐标对应的频次
            std::vector<float> uvwMapVector(uvw_presize);
            thrust::device_vector<float> uvwFrequencyMap(uvw_presize);
        
            // 存储计算后的到的最终结果
            thrust::device_vector<Complex> F(lmnC_index);

            // 计时统计
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            // 记录开始事件
            cudaEventRecord(start);

            // 记录uvw开始事件
            cudaEvent_t uvwstart, uvwstop;
            cudaEventCreate(&uvwstart);
            cudaEventCreate(&uvwstop);
            cudaEventRecord(uvwstart);

            // 创建一个临界区，保证只有一个线程进入，用于构建u v w
            int uvw_index, shade_index;
            #pragma omp critical
            {
                // 读取uvw
                string address_uvw = address + "updated_uvw" + to_string(p+1) + duration + sufix;
                cout << "address_uvw: " << address_uvw << std::endl;
                
                ifstream uvwFile(address_uvw);
                // 同时用一个向量保存每一个uvw坐标点的frequency
                uvw_index = 0;
                float u_point, v_point, w_point, freq_point;
                if (uvwFile.is_open()) {
                    while (uvwFile >> u_point >> v_point >> w_point >> freq_point) {
                        // cu, cv, cw 需要存储原始坐标
                        cu[uvw_index] = u_point;
                        cv[uvw_index] = v_point;
                        cw[uvw_index] = w_point;
                        uvwMapVector[uvw_index] = 1 / freq_point;
                        uvw_index++;
                    }
                }              
                cout << "load uvw with uvw_index: " << uvw_index << endl; 
                // 复制到GPU上
                thrust::copy(cu.begin(), cu.begin() + uvw_index, u.begin());
                thrust::copy(cv.begin(), cv.begin() + uvw_index, v.begin());
                thrust::copy(cw.begin(), cw.begin() + uvw_index, w.begin());
                thrust::copy(uvwMapVector.begin(), uvwMapVector.begin() + uvw_index, uvwFrequencyMap.begin());

                // 读取shadeM
                string address_shadeM = address + "shadeM" + to_string(p+1) + duration + sufix;
                cout << "address_shadeM: " << address_shadeM << std::endl;
                ifstream shadeMFile(address_shadeM);
                shade_index = 0;
                if (shadeMFile.is_open()) {
                    int shadeM1, shadeM2, shadeM3, shadeM4;
                    // matlab中是从1开始，因此所有值都减去 1
                    while (shadeMFile >> shadeM1 >> shadeM2 >> shadeM3 >> shadeM4) {
                        if(shadeM1 > shadeM2){
                            M1[shade_index] = shadeM2-1;
                            M2[shade_index] = shadeM1-1;
                        }else{
                            M1[shade_index] = shadeM1-1;
                            M2[shade_index] = shadeM2-1;
                        }
                        if(shadeM3 > shadeM4){
                            M3[shade_index] = shadeM4-1;
                            M4[shade_index] = shadeM3-1;
                        }else{
                            M3[shade_index] = shadeM3-1;
                            M4[shade_index] = shadeM4-1;
                        }
                        shade_index++;
                    }
                }
                cout << "load shade matrix with shade_index: " << shade_index << endl; 
                if(shade_index != uvw_index){
                    cout << "load wrong! uvw shape must be equal to shadeM shape" << endl;
                }else{
                    cout << "load right! uvw shape is equal to shadeM shape" << endl;
                }
                // 复制到GPU上
                thrust::copy(M1.begin(), M1.begin() + shade_index, shadeMat1.begin());
                thrust::copy(M2.begin(), M2.begin() + shade_index, shadeMat2.begin());
                thrust::copy(M3.begin(), M3.begin() + shade_index, shadeMat3.begin());
                thrust::copy(M4.begin(), M4.begin() + shade_index, shadeMat4.begin()); 
            }

            // 记录uvw结束事件
            cudaEventRecord(uvwstop);
            cudaEventSynchronize(uvwstop);
            // 计算经过的时间
            float uvwMS = 0;
            cudaEventElapsedTime(&uvwMS, uvwstart, uvwstop);
            printf("Period %d Load UWV Cost Time is: %f s\n", p+1, uvwMS/1000);
            // 销毁事件
            cudaEventDestroy(uvwstart);
            cudaEventDestroy(uvwstop);


            // 记录viss开始事件
            cudaEvent_t vissstart, vissstop;
            cudaEventCreate(&vissstart);
            cudaEventCreate(&vissstop);
            cudaEventRecord(vissstart);

            
            // 存储计算后的可见度
            cout << "Viss Computing..." << endl;
            thrust::device_vector<Complex> viss(uvw_index);
            // 调用函数计算可见度
            launch_visscal(uvw_index, lmnC_index, RES,
                    thrust::raw_pointer_cast(viss.data()),
                    thrust::raw_pointer_cast(dFF.data()),
                    thrust::raw_pointer_cast(u.data()),
                    thrust::raw_pointer_cast(v.data()),
                    thrust::raw_pointer_cast(w.data()),
                    thrust::raw_pointer_cast(l.data()),
                    thrust::raw_pointer_cast(m.data()),
                    thrust::raw_pointer_cast(n.data()),
                    thrust::raw_pointer_cast(shadeMat1.data()),
                    thrust::raw_pointer_cast(shadeMat2.data()), 
                    thrust::raw_pointer_cast(shadeMat3.data()), 
                    thrust::raw_pointer_cast(shadeMat4.data()),
                    thrust::raw_pointer_cast(NXq.data()),
                    thrust::raw_pointer_cast(countLoc.data()),
                    I1, CPI, zero, two, dl, dm, dn
            );
            CHECK(cudaDeviceSynchronize());
            cout << "period " << p+1 << " viss compute success!" << endl;

            // 记录viss结束事件
            cudaEventRecord(vissstop);
            cudaEventSynchronize(vissstop);
            // 计算经过的时间
            float vissMS = 0;
            cudaEventElapsedTime(&vissMS, vissstart, vissstop);
            printf("Period %d Compute Viss Cost Time is: %f s\n", p+1, vissMS/1000);
            // 销毁事件
            cudaEventDestroy(vissstart);
            cudaEventDestroy(vissstop);


            // 记录imagerecon开始事件
            cudaEvent_t imagereconstart, imagereconstop;
            cudaEventCreate(&imagereconstart);
            cudaEventCreate(&imagereconstop);
            cudaEventRecord(imagereconstart);

            // 创建预处理的theta 和 phi
            cout << "Process theta and phi ..." << endl;
            thrust::device_vector<float> thetaP0(uvw_presize), phiP0(uvw_presize), dtheta(uvw_presize), dphi(uvw_presize);
            // 直接在GPU上计算
            // 因为前面加载的时候确保了M1 > M2, M3 > M4, 因此abs函数可以去掉
            thrust::transform(shadeMat1.begin(), shadeMat1.end(), shadeMat2.begin(), thetaP0.begin(), thrust::divides<float>());
            thrust::transform(shadeMat3.begin(), shadeMat3.end(), shadeMat4.begin(), phiP0.begin(), thrust::divides<float>());
            thrust::transform(shadeMat1.begin(), shadeMat1.end(), shadeMat2.begin(), dtheta.begin(), thrust::divides<float>());
            thrust::transform(shadeMat3.begin(), shadeMat3.end(), shadeMat4.begin(), dphi.begin(), thrust::divides<float>());
            
            cout << "Image Reconstruction ..." << endl;
            // 调用image_recon函数计算图像反演
            launch_imagerecon(uvw_index, lmnC_index, RES,
                    thrust::raw_pointer_cast(F.data()),
                    thrust::raw_pointer_cast(viss.data()),
                    thrust::raw_pointer_cast(u.data()),
                    thrust::raw_pointer_cast(v.data()),
                    thrust::raw_pointer_cast(w.data()),
                    thrust::raw_pointer_cast(l.data()),
                    thrust::raw_pointer_cast(m.data()),
                    thrust::raw_pointer_cast(n.data()),
                    thrust::raw_pointer_cast(uvwFrequencyMap.data()),
                    thrust::raw_pointer_cast(thetaP0.data()),
                    thrust::raw_pointer_cast(phiP0.data()),
                    thrust::raw_pointer_cast(dtheta.data()),
                    thrust::raw_pointer_cast(dphi.data()),
                    I1, CPI, zero, two, dl, dm, dn
            );
            // 进行线程同步
            CHECK(cudaDeviceSynchronize());
            cout << "Period " << p+1 << " Image Reconstruction Success!" << endl;
            
            // 记录imagerecon结束事件
            cudaEventRecord(imagereconstop);
            cudaEventSynchronize(imagereconstop);
            // 计算经过的时间
            float imagereconMS = 0;
            cudaEventElapsedTime(&imagereconMS, imagereconstart, imagereconstop);
            printf("Period %d Image Reconstruction Cost Time is: %f s\n", p+1, imagereconMS/1000);
            // 销毁事件
            cudaEventDestroy(imagereconstart);
            cudaEventDestroy(imagereconstop);


            // 记录saveimage开始事件
            cudaEvent_t saveimagestart, saveimagestop;
            cudaEventCreate(&saveimagestart);
            cudaEventCreate(&saveimagestop);
            cudaEventRecord(saveimagestart);
            // 创建一个临界区，用于保存图像反演结果
            #pragma omp critical
            {   
                // 将数据从设备内存复制到主机内存
                std::vector<Complex> host_F(F.size());
                CHECK(cudaMemcpy(host_F.data(), thrust::raw_pointer_cast(F.data()), F.size() * sizeof(Complex), cudaMemcpyDeviceToHost));
                CHECK(cudaDeviceSynchronize());
                // 打开文件
                string address_F = "F_recon_10M/F" + to_string(p+1) + "period10M_optim2.txt";
                cout << "Period " << p+1 << " save address_F: " << address_F << endl;
                std::ofstream file(address_F);
                if (file.is_open()) {
                    // 按照指定格式写入文件
                    for(const Complex& value : host_F)
                    {
                        file << value.real() << std::endl;
                    }
                }
                // 关闭文件
                file.close();
                std::cout << "Period " << p+1 << " save F success!" << std::endl;
            }

            // 记录saveimage结束事件
            cudaEventRecord(saveimagestop);
            cudaEventSynchronize(saveimagestop);
            // 计算经过的时间
            float saveimageMS = 0;
            cudaEventElapsedTime(&saveimageMS, saveimagestart, saveimagestop);
            printf("Period %d Save Image Cost Time is: %f s\n", p+1, saveimageMS/1000);
            // 销毁事件
            cudaEventDestroy(saveimagestart);
            cudaEventDestroy(saveimagestop);

            // 记录全程结束事件
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            // 计算经过的时间
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("Period %d Elapsed time: %f s\n", p+1, milliseconds/1000);
            // 销毁事件
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }
    
    gettimeofday(&finish, NULL);
    total_time = ((finish.tv_sec - start.tv_sec) * 1000000 + (finish.tv_usec - start.tv_usec)) / 1000000.0;
    cout << "total time: " << total_time << "s" << endl;
    return 0;
}


int main()
{   
    int start_period = 1;  // 从哪个周期开始，一共是130个周期
    vissGen(0, 20940, start_period);
}

