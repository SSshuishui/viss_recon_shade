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

string address = "./frequency_1M/";
string F_address = "./F_recon_1M/";
string para;
string duration = "frequency1M";  // 第几个周期的uvw
string sufix = ".txt";

// 1 M
const int uvw_presize = 4000000;

// 定义常量
#define BLOCK_SIZE 128                     // 线程块大小
#define SHARED_MEM_SIZE BLOCK_SIZE         // 共享内存大小
#define MAX_THREADS_PER_BLOCK 1024        // GPU每个块的最大线程数


// 计算特定 q 值的索引有几个
__global__ void computeLocCount(
    const float* __restrict__ NX,      // 添加 const 和 __restrict__
    int* __restrict__ countLoc,
    const int lmnC_index,
    const int NX_index)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= lmnC_index) return;

    // 使用局部计数器
    int local_count = 0;
    
    // 主循环展开
    #pragma unroll 4
    for (int i = 0; i < NX_index; i++) {
        // 预取数据到寄存器
        const float nx_val = NX[i] - 1;
        // 使用直接比较代替条件语句
        local_count += (nx_val == q);
    }
    
    // 存储最终结果
    countLoc[q] = local_count;
}


// 把每个 q 值对应的索引保存下来
__global__ void computeLocViss(
    const float* __restrict__ NX,
    int* __restrict__ NXq,
    const int* __restrict__ countLoc,
    const int lmnC_index,
    const int NX_index)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= lmnC_index) return;

    // 预计算起始索引
    int start_idx = 0;
    if (q > 0) {
        #pragma unroll 4
        for (int i = 0; i < q; i++) {
            start_idx += countLoc[i];
        }
    }

    // 使用局部索引计数器
    int local_idx = start_idx;
    
    // 主循环展开
    #pragma unroll 4
    for (int i = 0; i < NX_index; i++) {
        // 预取数据到寄存器
        const float nx_val = NX[i] - 1;
        // 使用条件赋值代替if语句
        bool match = (nx_val == q);
        if (match) {
            NXq[local_idx++] = i;
        }
    }
}


// 定义计算可见度核函数
__global__ void visscal(
    const int uvws_index,
    const int lmnC_index,
    const int res,
    const float* __restrict__ FF,
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
    const int* __restrict__ NXq,
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
    if (uvws_ >= uvws_index) return;

    // 预加载频繁使用的数据到寄存器
    const float u_val = u[uvws_] / dl;
    const float v_val = v[uvws_] / dm;
    const float w_val = w[uvws_] / dn;
    
    // 预加载遮挡相关的数据
    const int shade_m1 = shadeM1[uvws_];
    const int shade_m2 = shadeM2[uvws_];
    const int shade_m3 = shadeM3[uvws_];
    const int shade_m4 = shadeM4[uvws_];

    // 初始化累加器
    Complex acc = zero;
    int start_idx = 0;

    for (int lmnC_ = 0; lmnC_ < lmnC_index; ++lmnC_) {
        const int current_count = countLoc[lmnC_];
        float sumReal = 0.0f;

        for (int con = 0; con < current_count; ++con) {
            const int locViss = NXq[con + start_idx];
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

        // 计算相位和复指数
        const float phase = u_val * l[lmnC_] + v_val * m[lmnC_] + w_val * (n[lmnC_] - 1.0f);
        const Complex exp_val = complexExp((zero - I1) * two * CPI * Complex(phase, 0.0f));
        acc += Complex(C_tmp, 0.0f) * exp_val;    
    }

    // 计算最终结果
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
    const int* d_NXq,
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
    const int numStreams = 4;  // 使用4个流
    const int itemsPerStream = uvws_index / numStreams;  // 计算每个流处理的数据量
    
    // 创建流数组
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // 启动多个流
    for (int i = 0; i < numStreams; i++) {
        const int streamStart = i * itemsPerStream;
        const int streamSize = (i == numStreams-1) ? uvws_index - streamStart : itemsPerStream;
        
        if (streamSize <= 0) break;

        const int streamBlocks = (streamSize + threadsPerBlock - 1) / threadsPerBlock;

        // 启动核函数
        visscal<<<streamBlocks, threadsPerBlock, 0, streams[i]>>>(
            streamSize, lmnC_index, res,
            d_FF, 
            d_viss+streamStart,
            d_u+streamStart, 
            d_v+streamStart, 
            d_w+streamStart,
            d_l, 
            d_m, 
            d_n,
            d_shadeM1+streamStart, 
            d_shadeM2+streamStart, 
            d_shadeM3+streamStart, 
            d_shadeM4+streamStart,
            d_NXq, 
            d_countLoc,
            I1, CPI, zero, two,
            dl, dm, dn
        );
    }

    // 等待所有流完成（如果需要）
    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // 清理流
    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
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
    const float* __restrict__ thetaP0,
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
            bool skip_calculation = (fabs(s_thetaP0[i] - thetaP) < s_dtheta[i] && 
                                  fabs(s_phiP0[i] - phiP) < s_dphi[i]);

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

    // 设置流的数量
    const int numStreams = 4;  // 可以根据需要调整
    const int itemPerStream = (lmnC_index + numStreams - 1) / numStreams;

    // 创建流数组
    cudaStream_t streams[numStreams];
    for(int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // 设置缓存配置
    cudaFuncSetCacheConfig(imagerecon, cudaFuncCachePreferShared);

    // 启动多个流
    for(int i = 0; i < numStreams; i++) {
        const int streamStart = i * itemPerStream;
        const int streamSize = (i == numStreams-1) ? lmnC_index-streamStart : itemPerStream;

        if(streamSize <= 0) break;

        const int streamBlocks = (streamSize + threadsPerBlock - 1) / threadsPerBlock;

        // 启动核函数，只对 l, m, n 相关的数组进行偏移
        imagerecon<<<streamBlocks, threadsPerBlock, sharedMemSize, streams[i]>>>(
            uvw_index,
            streamSize,        // 这个流处理的 lmnC 数量
            res,
            d_F + streamStart, 
            d_viss,           
            d_u,              
            d_v,              
            d_w,              
            d_l + streamStart, // 只处理部分 l
            d_m + streamStart, // 只处理部分 m
            d_n + streamStart, // 只处理部分 n
            d_uvwFrequencyMap,
            d_thetaP0,
            d_phiP0,
            d_dtheta,
            d_dphi,
            I1, CPI, zero, two,
            dl, dm, dn
        );
    }

    // 等待所有流完成（如果需要）
    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // 清理流
    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
}


int vissGen(int id, int RES, int start_period) 
{   
    cout << "res: " << RES << endl;
    int days = 1;  // 一共有多少个周期  15月 * 30天 / 14天/周期
    cout << "periods: " << days << endl;
    Complex I1(0.0, 1.0);
    float dl = 2 * RES / (RES - 1);
    float dm = 2 * RES / (RES - 1);
    float dn = 2 * RES / (RES - 1);
    Complex zero(0.0, 0.0);
    Complex two(2.0, 0.0);
    Complex CPI(M_PI, 0.0);

    gettimeofday(&start, NULL);
    int nDevices = 1;
    // 设置节点数量（gpu显卡数量）
    CHECK(cudaGetDeviceCount(&nDevices));
    // 设置并行区中的线程数
    omp_set_num_threads(nDevices);
    cout << "devices: " << nDevices << endl;

    // 加载存储 l m n C nt的文件（对于不同的frequency不一样，只与frequency有关）
    string para, address_l, address_m, address_n, address_C, address_NX, address_FF;
    ifstream lFile, mFile, nFile, cFile, NXFile, FFFile;
    para = "l";
    address_l = address + para + sufix;
    lFile.open(address_l);
    cout << "address_l: " << address_l << endl;
    para = "m";
    address_m = address + para + sufix;
    mFile.open(address_m);
    cout << "address_m: " << address_m << endl;
    para = "n";
    address_n = address + para + sufix;
    nFile.open(address_n);
    cout << "address_n: " << address_n << endl;
    para = "C";
    address_C = address + para + sufix;
    cFile.open(address_C);
    cout << "address_C: " << address_C << endl;
    para = "NX";
    address_NX = address + para + sufix;
    NXFile.open(address_NX);
    cout << "address_NX: " << address_NX << endl;
    para = "FF";
    address_FF = address + para + sufix;
    FFFile.open(address_FF);
    cout << "address_FF: " << address_FF << endl;
    if (!lFile.is_open() || !mFile.is_open() || !nFile.is_open() || !cFile.is_open() || !NXFile.is_open() ||!FFFile.is_open()) {
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

    // 导入uvw坐标的出现频次，txt文件的每一行每个坐标的频次
    auto uvwMapStart = std::chrono::high_resolution_clock::now();
    // 创建map存储
    std::unordered_map<std::string, float> cUVWFrequencyMap;
    string uvwmap_address = address + "uvwMap130.txt";
    std::ifstream uvwMapFile(uvwmap_address);
    if (uvwMapFile.is_open()) {
        // 读取第一行获取总行数
        string firstLine;
        std::getline(uvwMapFile, firstLine);
        int totalLines = std::stoi(firstLine);
        cout << "uvwMap totalLines: " << totalLines << endl;
        // 预分配内存
        cUVWFrequencyMap.reserve(totalLines);
        // 每一行的格式： -23 -288 -166 4
        string line;
        while (std::getline(uvwMapFile, line)) {
            std::istringstream iss(line);
            int u_point, v_point, w_point;
            int uvw_frequency;
            if (iss >> u_point >> v_point >> w_point >> uvw_frequency) {
                std::string key = std::to_string(u_point) + "_" + std::to_string(v_point) + "_" + std::to_string(w_point);
                cUVWFrequencyMap[key] = uvw_frequency;
            } else {
                cout << "Failed to parse line: " << line << endl; // 解析失败时的调试信息
            }
        }
        uvwMapFile.close();
    }
    // 打印测试确保是正确的
    int count = 0;
    int numElementsToPrint = 6; // 设定要打印的元素数量
    for (const auto& pair : cUVWFrequencyMap) {
        std::cout << pair.first << ": " << pair.second << std::endl;
        if (++count == numElementsToPrint) {
            break;
        }
    }
    cout << "Transfer uvw Frequency Success! " << endl;
    auto uvwMapFinish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> uvwMapElapsed = uvwMapFinish - uvwMapStart;
    cout << "Transfer uvw Frequency Elapsed Time: " << uvwMapElapsed.count() << " s\n";

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
            thrust::device_vector<float> l(cl.begin(), cl.end());
            thrust::device_vector<float> m(cm.begin(), cm.end());
            thrust::device_vector<float> n(cn.begin(), cn.end());
            thrust::device_vector<float> C(cc.begin(), cc.end());

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
                string address_uvw = address + "uvw" + to_string(p+1) + duration + sufix;
                cout << "address_uvw: " << address_uvw << std::endl;
                
                ifstream uvwFile(address_uvw);
                // 同时用一个向量保存每一个uvw坐标点的frequency
                uvw_index = 0;
                float u_point, v_point, w_point;
                string key_point;
                if (uvwFile.is_open()) {
                    while (uvwFile >> u_point >> v_point >> w_point) {
                        // 直接构造 key_point
                        key_point = std::to_string(static_cast<int>(u_point)) + "_" + 
                                    std::to_string(static_cast<int>(v_point)) + "_" + 
                                    std::to_string(static_cast<int>(w_point));

                        // 简化查找操作
                        auto it = cUVWFrequencyMap.find(key_point);
                        if (it != cUVWFrequencyMap.end()) {
                            uvwMapVector[uvw_index] = 1 / (it->second);  // 存储频次的倒数
                        } else {
                            uvwMapVector[uvw_index] = 1; 
                        }
                        // cu, cv, cw 需要存储原始坐标
                        cu[uvw_index] = u_point;
                        cv[uvw_index] = v_point;
                        cw[uvw_index] = w_point;
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

            int blockSize;
            int minGridSize; // 最小网格大小

            // 先提前计算每个 q 值的索引有几个
            thrust::device_vector<int> countLoc(lmnC_index);
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeLocCount, 0, 0);
            int gridSize = floor(lmnC_index + blockSize - 1) / blockSize;
            computeLocCount<<<gridSize, blockSize>>>(
                thrust::raw_pointer_cast(dNX.data()), 
                thrust::raw_pointer_cast(countLoc.data()), 
                lmnC_index, NX_index);
            CHECK(cudaDeviceSynchronize());

            // 然后存下来每个 q 值对应的索引
            thrust::device_vector<int> NXq(dNX.size());
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeLocViss, 0, 0);
            gridSize = floor(lmnC_index + blockSize - 1) / blockSize;
            cout << "Compute LocViss, girdSize: " << gridSize << endl;
            cout << "Compute LocViss, blockSize: " << blockSize << endl;
            printf("Compute LocViss... Here is gpu %d running process %d\n", omp_get_thread_num(), p+1);
            computeLocViss<<<gridSize, blockSize>>>(
                thrust::raw_pointer_cast(dNX.data()), 
                thrust::raw_pointer_cast(NXq.data()), 
                thrust::raw_pointer_cast(countLoc.data()), 
                lmnC_index, NX_index);
            CHECK(cudaDeviceSynchronize());

            // 存储计算后的可见度
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
                    I1, CPI, zero, two, dl, dm, dn);
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
            thrust::device_vector<float> thetaP0(uvw_presize), phiP0(uvw_presize), dtheta(uvw_presize), dphi(uvw_presize);
            // 直接在GPU上计算
            // 因为前面加载的时候确保了M1 > M2, M3 > M4, 因此abs函数可以去掉
            thrust::transform(shadeMat1.begin(), shadeMat1.end(), shadeMat2.begin(), thetaP0.begin(), thrust::divides<float>());
            thrust::transform(shadeMat3.begin(), shadeMat3.end(), shadeMat4.begin(), phiP0.begin(), thrust::divides<float>());
            thrust::transform(shadeMat1.begin(), shadeMat1.end(), shadeMat2.begin(), dtheta.begin(), thrust::divides<float>());
            thrust::transform(shadeMat3.begin(), shadeMat3.end(), shadeMat4.begin(), dphi.begin(), thrust::divides<float>());
            
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
                I1, CPI, zero, two, dl, dm, dn);
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
                // 在CPU上创建变量保存F结果
                thrust::host_vector<Complex> tempF = F;
                thrust::host_vector<Complex> extendF(NX_index);

                std::ofstream F_File;
                string address_F = "F_recon_1M/F" + to_string(p+1) + "_optim2.txt";
                cout << "address_F: " << address_F << endl;
                F_File.open(address_F);
                if (!F_File.is_open()) {
                    std::cerr << "Error opening file: " << address_F << endl;
                }
                for (int c = 0; c < NX_index; c++) {
                    int tmp = static_cast<int>(cNX[c]) - 1;  // matlab中下标从1开始
                    extendF[c] = tempF[tmp];
                    F_File << extendF[c].real() << std::endl;
                }
                F_File.close();
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
    int start_period = 0;  // 从哪个周期开始，一共是130个周期
    vissGen(0, 2094, start_period);
}

