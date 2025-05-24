#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>

#include "error.cuh"
#include <omp.h>
#include <cuda_runtime.h>


// Define constants
const int satnum = 8;
const float mu_e = 4.902793455e3; // unit in km^3/s^2
const float d2r = M_PI / 180;
const float a = 2038.14;
const float e = 0;
const float incl = 30 * d2r;
const float argp = 0 * d2r;
const std::vector<float> r1 = {0, 1e3, 4.45e3, 6e3, 9.1e3, 16.2e3, 35.3e3, 100e3};
const std::vector<float> r2 = {0, 0.1e3, 0.445e3, 0.6e3, 0.91e3, 1.62e3, 3.53e3, 10e3};


// LinSpace函数
struct linspace_functor {
    float a, step;
    linspace_functor(float _a, float _step) : a(_a), step(_step) {}
    __host__ __device__ float operator()(const int& x) const { 
        return a + step * x;
    }
};

void generate_linspace(thrust::device_vector<float>& d_vec, float start, float end, int num) {
    // Calculate step size
    float step = (end - start) / float(num - 1);
    // Generate sequence [0, 1, 2, ..., num-1]
    thrust::sequence(d_vec.begin(), d_vec.end());
    // Transform sequence to linear space
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), linspace_functor(start, step));
}

struct calculate_dM {
    float a; // 常数 a
    calculate_dM(float _a) : a(_a) {}
    __host__ __device__
    float operator()(float d) const {
        return asin(d / (2 * a * 1e3)) * 2;
    }
};

struct SortCompare {
    __host__ __device__
    bool operator()(const thrust::tuple<float, float, float>& a, const thrust::tuple<float, float, float>& b) const {
        if (thrust::get<0>(a) != thrust::get<0>(b)) {
            return thrust::get<0>(a) < thrust::get<0>(b); // 首先按 u 坐标排序
        } else if (thrust::get<1>(a) != thrust::get<1>(b)) {
            return thrust::get<1>(a) < thrust::get<1>(b); // 然后按 v 坐标排序
        } else {
            return thrust::get<2>(a) < thrust::get<2>(b); // 最后按 w 坐标排序
        }
    }
};

struct UniqueCompare {
    __host__ __device__
    bool operator()(const thrust::tuple<float, float, float>& a, const thrust::tuple<float, float, float>& b) const {
        return thrust::get<0>(a) == thrust::get<0>(b) &&
               thrust::get<1>(a) == thrust::get<1>(b) &&
               thrust::get<2>(a) == thrust::get<2>(b); // 检查 uvw 坐标是否都相等
    }
};


__global__ void calculateOrbit(float* dM, float* x, float* y, float* z, float* Mt, int OrbitCounts, int OrbitRes, float d2r, float e, float a, float mu_e, float argp, float incl, int index, int sat) {
    // 计算全局索引
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算g和k的值
    int k = globalIdx / OrbitRes;  // 圈数
    int po = globalIdx % OrbitRes;  // 点数
    if(k < OrbitCounts && po < OrbitRes) {
        float raan = k * d2r * 0.08 + (po / OrbitRes) * d2r * 0.08 + (index - 1) * OrbitCounts * d2r * 0.08;
        float M = Mt[po] + dM[k * OrbitRes + po];
        float E0 = M;
        for (int i = 1; i < 100; ++i) {
            float M0 = E0 - e * sin(E0);
            float error = M - M0;
            if (abs(error) < 1e-15) {
                break;
            }
            E0 = E0 + error / (1 - e * cos(E0));
        }
        float temp = tan(E0 / 2) / sqrt((1 - e) / (1 + e));
        float theta = atan(temp) * 2;
        float r = a * (1 - e * e) / (1 + e * cos(theta));
        float w = theta + argp;
        int tmp = (sat*OrbitRes*OrbitCounts) + k * OrbitRes + po;
        x[tmp] = r * (cos(w) * cos(raan) - sin(w) * cos(incl) * sin(raan));
        y[tmp] = r * (cos(w) * sin(raan) + sin(w) * cos(incl) * cos(raan));
        z[tmp] = r * (sin(w) * sin(incl));
    }
}


__global__ void uvwPosition(float* x, float* y, float* z, float* xt, float* yt, float* zt, int OrbitCounts, int OrbitRes, int interval, int span, int sat) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;   // OrbitCounts * OrbitRes / 3  大小和xt yt zt的维度一样
    int k = globalIdx / (OrbitRes/3) + 1;  // 第几圈

    int offset = 0;
    int tmp = sat*OrbitRes*OrbitCounts;  // 计算是用的x的哪一段
    if(k <= OrbitCounts){
        offset = (k-1)*interval;
        int endIndexWithoutMin = (k-1)*OrbitRes + offset + span;
        int startIdx = (k-1)*OrbitRes + offset; 
        if ((k-1)*(OrbitRes+interval) > OrbitRes*OrbitCounts) {
            int reset_index = (OrbitRes*OrbitCounts) / (OrbitRes+interval) + 1;
            // 当超出范围之后，从头开始采样，但是偏置还是存在
            startIdx = (k-1-reset_index)*OrbitRes + offset;
            endIndexWithoutMin = (k-1-reset_index)*OrbitRes + offset + span;
        }

        int circleIdx = startIdx + (globalIdx-(k-1)*span);  // 每一圈中对应的向量下标
        // 超出的部分就设置为0
        if(endIndexWithoutMin > OrbitRes * OrbitCounts){
            // startIdx 到 最后长度设置为正常值，超出部分设置为0
            if(circleIdx > OrbitRes*OrbitCounts){
                xt[tmp/3 + globalIdx] = 0;
                yt[tmp/3 + globalIdx] = 0;
                zt[tmp/3 + globalIdx] = 0;
            }else{
                xt[tmp/3 + globalIdx] = x[tmp + circleIdx];
                yt[tmp/3 + globalIdx] = y[tmp + circleIdx];
                zt[tmp/3 + globalIdx] = z[tmp + circleIdx];
            }
        }
        // 没有超出的部分就全部设置为正常值
        else{
            xt[tmp/3 + globalIdx] = x[tmp + circleIdx];
            yt[tmp/3 + globalIdx] = y[tmp + circleIdx];
            zt[tmp/3 + globalIdx] = z[tmp + circleIdx];
        }
    }
}

__global__ void computeUVW(float *xt, float *yt, float *zt, thrust::tuple<float, float, float> *UVWHash, int OrbitCounts, int OrbitRes, float lambda, int m, int n) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;    // OrbitCounts*OrbitRes/3  大小和xt yt zt的1/8维度一样
    int tmp = OrbitRes*OrbitCounts/3;
    float u_val = ceil((xt[tmp * n + globalIdx] - xt[tmp * m + globalIdx]) * 1e3 / lambda);
    float v_val = ceil((yt[tmp * n + globalIdx] - yt[tmp * m + globalIdx]) * 1e3 / lambda);
    float w_val = ceil((zt[tmp * n + globalIdx] - zt[tmp * m + globalIdx]) * 1e3 / lambda);
    UVWHash[globalIdx] = thrust::make_tuple(u_val, v_val, w_val);
}



void writeToFile(const thrust::device_vector<float>& device_vector, const std::string& filename, int satnum, int columns) {
    // 将数据从设备内存复制到主机内存
    std::vector<float> host_vector(device_vector.size());
    thrust::copy(device_vector.begin(), device_vector.end(), host_vector.begin());
    // 打开文件
    std::ofstream file(filename);
    if (file.is_open()) {
        // 按照指定格式写入文件
        for (int i = 0; i < columns; ++i) {
            for (int s = 0; s < satnum; ++s) {
                file << host_vector[s * columns + i] << " ";
            }
            file << std::endl; // 换行，开始新的一组数据
        }
    }
    // 关闭文件
    file.close();
}


void MIncline(int index, float frequency, float stride, int gpu_id) {
    int OrbitRes = 2.4 * 3600 * int(1.0/stride);

    int OrbitCounts = 140;
    int interval = OrbitRes / 360 * 10;  
    int span = OrbitRes / 3; 
    int lambda = 3e8/frequency;

    std::cout<< "=====================================================================" << std::endl;
    std::cout << "OrbitRes : " << OrbitRes << std::endl;
    std::cout << "interval : " << interval << std::endl;
    std::cout << "span : " << span << std::endl;
    std::cout << "lambda : " << lambda << std::endl;
    std::cout << "satnum : " << satnum << std::endl;
    std::cout << "OrbitCounts : " << OrbitCounts << std::endl;
    std::cout << "OrbitRes*OrbitCounts*satnum : " << OrbitRes * OrbitCounts * satnum << std::endl;
    std::cout << "OrbitRes*OrbitCounts*satnum/3 : " << OrbitRes * OrbitCounts * satnum / 3 << std::endl;
    std::cout<< "=====================================================================" << std::endl;


    cudaSetDevice(gpu_id);
    // 在GPU上创建xyz和Mt
    thrust::device_vector<float> x(OrbitRes * OrbitCounts * satnum, 0);
    thrust::device_vector<float> y(OrbitRes * OrbitCounts * satnum, 0);
    thrust::device_vector<float> z(OrbitRes * OrbitCounts * satnum, 0);
    thrust::device_vector<float> Mt(OrbitRes);

    thrust::device_vector<float> xt(OrbitRes * OrbitCounts * satnum / 3, 0);
    thrust::device_vector<float> yt(OrbitRes * OrbitCounts * satnum / 3, 0);
    thrust::device_vector<float> zt(OrbitRes * OrbitCounts * satnum / 3, 0);

    generate_linspace(Mt, 0, 2*M_PI, OrbitRes);
    std::cout << "GPU " << gpu_id << " Mt_all compute successe! " << std::endl;

    // 记录循环事件
    cudaEvent_t satstart, satstop;
    CHECK(cudaEventCreate(&satstart));
    CHECK(cudaEventCreate(&satstop));
    CHECK(cudaEventRecord(satstart));

    for (int ss = 0; ss < satnum; ss++) 
    {
        thrust::device_vector<float> d1(OrbitCounts * OrbitRes / 2);
        generate_linspace(d1, r1[ss], r2[ss], OrbitCounts * OrbitRes / 2);
        thrust::device_vector<float> d2(OrbitCounts * OrbitRes / 2);
        generate_linspace(d2, r2[ss], r1[ss], OrbitCounts * OrbitRes / 2);
        std::cout << "GPU " << gpu_id << " sat " << ss+1 << " d1 d2 compute successe! " << std::endl;

        // 创建dM向量，初始大小与d相同
        thrust::device_vector<float> dM(d1.size() + d2.size());
        // 对d1的每个元素进行转换，存入dM的前半部分
        thrust::transform(d1.begin(), d1.end(), dM.begin(), calculate_dM(a));
        // 对d2的每个元素进行转换，存入dM的后半部分
        thrust::transform(d2.begin(), d2.end(), dM.begin() + d1.size(), calculate_dM(a));
        std::cout << "GPU " << gpu_id << " sat " << ss+1 << " dM compute successe! " << std::endl;
        // 用完d1和d2后，释放它们的内存
        d1.clear();
        d1.shrink_to_fit();
        d2.clear();
        d2.shrink_to_fit();

        int minGridSize; // 最小网格大小
        int posBlockSize;
        int posGridSize;    // 实际网格大小
        // 记录position事件
        cudaEvent_t posstart, posstop;
        CHECK(cudaEventCreate(&posstart));
        CHECK(cudaEventCreate(&posstop));
        CHECK(cudaEventRecord(posstart));
        CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &posBlockSize, calculateOrbit, 0, 0));
        posGridSize = floor(OrbitCounts * OrbitRes + posBlockSize - 1) / posBlockSize;
        std::cout << "GPU" << gpu_id << " Calculate position, blockSize: " << posBlockSize << std::endl;
        std::cout << "GPU" << gpu_id << " Calculate position, girdSize: " << posGridSize << std::endl;
        calculateOrbit<<<posGridSize, posBlockSize>>>(
            thrust::raw_pointer_cast(dM.data()),
            thrust::raw_pointer_cast(x.data()),
            thrust::raw_pointer_cast(y.data()),
            thrust::raw_pointer_cast(z.data()), 
            thrust::raw_pointer_cast(Mt.data()), 
            OrbitCounts, OrbitRes, d2r, e, a, mu_e, argp, incl, index, ss
        );

        std::cout << "GPU" << gpu_id << " calculateOrbit compute success!" << std::endl;
        // 记录position结束事件
        CHECK(cudaEventRecord(posstop));
        CHECK(cudaEventSynchronize(posstop));
        // 计算经过的时间
        float posMS = 0;
        CHECK(cudaEventElapsedTime(&posMS, posstart, posstop));
        printf("Frequency-%f Index-%d Sat-%d On GPU-%d Calculate position xyz Cost Time is: %f s\n", frequency, index, ss+1, gpu_id, posMS/1000);
        std::cout << "sat " << ss+1 << " position compute successe! on GPU " << gpu_id << std::endl;
        CHECK(cudaDeviceSynchronize());  // 线程同步，等前面完成了再进行下一步    

        // 调用uvw_position核函数
        // 记录uvw开始事件
        cudaEvent_t uvwstart, uvwstop;
        CHECK(cudaEventCreate(&uvwstart));
        CHECK(cudaEventCreate(&uvwstop));
        CHECK(cudaEventRecord(uvwstart));
        int uvwBlockSize;
        int uvwGridSize;    // 实际网格大小
        CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &uvwBlockSize, uvwPosition, 0, 0));
        uvwGridSize = floor(OrbitCounts * OrbitRes / 3 + uvwBlockSize - 1) / uvwBlockSize;
        std::cout << "GPU" << gpu_id << " UVW position, blockSize: " << uvwBlockSize << std::endl;
        std::cout << "GPU" << gpu_id << " UVW position, girdSize: " << uvwGridSize << std::endl;
        uvwPosition<<<uvwGridSize, uvwBlockSize>>>(
            thrust::raw_pointer_cast(x.data()),
            thrust::raw_pointer_cast(y.data()),
            thrust::raw_pointer_cast(z.data()), 
            thrust::raw_pointer_cast(xt.data()),
            thrust::raw_pointer_cast(yt.data()),
            thrust::raw_pointer_cast(zt.data()), 
            OrbitCounts, OrbitRes, interval, span, ss
        );
        std::cout << "GPU" << gpu_id << " uvwPosition compute success!" << std::endl;
        // 记录uvw结束事件
        CHECK(cudaEventRecord(uvwstop));
        CHECK(cudaEventSynchronize(uvwstop));
        // 计算经过的时间
        float uvwMS = 0;
        CHECK(cudaEventElapsedTime(&uvwMS, uvwstart, uvwstop));
        printf("Frequency-%f Index-%d Sat-%d On GPU-%d Compute xt yt zt Cost Time is: %f s\n", frequency, index, ss+1, gpu_id, uvwMS/1000);
        std::cout << "Will Save! Sat " << ss+1 << " uvw position compute successe! on GPU " << gpu_id << std::endl;
        CHECK(cudaDeviceSynchronize());  // 线程同步，等前面完成了再进行下一步    
    }
    
    // writeToFile(x, "x_output.txt", satnum, OrbitRes*OrbitCounts);
    // writeToFile(y, "y_output.txt", satnum, OrbitRes*OrbitCounts);
    // writeToFile(z, "z_output.txt", satnum, OrbitRes*OrbitCounts);

    // writeToFile(xt, "xt.txt", satnum, OrbitRes*OrbitCounts/3);
    // writeToFile(yt, "yt.txt", satnum, OrbitRes*OrbitCounts/3);
    // writeToFile(zt, "zt.txt", satnum, OrbitRes*OrbitCounts/3);

    // 记录循环结束事件
    CHECK(cudaEventRecord(satstop));
    CHECK(cudaEventSynchronize(satstop));
    // 计算经过的时间
    float satMS = 0;
    CHECK(cudaEventElapsedTime(&satMS, satstart, satstop));
    printf("Frequency-%f Index-%d On GPU-%d All satellite Calculate Cost Time is: %f s\n", frequency, index, gpu_id, satMS/1000);
    // 销毁事件
    cudaEventDestroy(satstart);
    cudaEventDestroy(satstop);

    // 用完xyz后，释放它们的内存
    x.clear();
    x.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
    z.clear();
    z.shrink_to_fit();

    // 记录duplicate开始事件
    cudaEvent_t dupstart, dupstop;
    CHECK(cudaEventCreate(&dupstart));
    CHECK(cudaEventCreate(&dupstop));
    CHECK(cudaEventRecord(dupstart));

    thrust::device_vector<thrust::tuple<float, float, float>> UVWHash(OrbitRes*OrbitCounts/3, thrust::make_tuple(0.0, 0.0, 0.0));  // 用来存储uvw坐标的哈希值
    thrust::device_vector<thrust::tuple<float, float, float>> globalUVW;   // 用来存储全局UVW
    
    int blockSize;
    int minGridSize; // 最小网格大小
    int gridSize;    // 实际网格大
    int loopIndex = 0;
    for (int m=0; m<satnum; m++) {
        for (int n=0; n<satnum; n++)
        {
            if (m != n) {
                loopIndex++;
                std::cout << "loop: " << loopIndex << std::endl;

                CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeUVW, 0, 0));
                gridSize = floor(OrbitCounts*OrbitRes/3 + blockSize - 1) / blockSize;
                std::cout << "GPU" << gpu_id << "Calculate hash, blockSize: " << blockSize << std::endl;
                std::cout << "GPU" << gpu_id << "Calculate hash, girdSize: " << gridSize << std::endl;
                thrust::tuple<float, float, float>* d_UVWHash = thrust::raw_pointer_cast(UVWHash.data());
                computeUVW<<<gridSize, blockSize>>>( 
                    thrust::raw_pointer_cast(xt.data()),
                    thrust::raw_pointer_cast(yt.data()),
                    thrust::raw_pointer_cast(zt.data()), 
                    d_UVWHash,
                    OrbitCounts, OrbitRes, lambda, m, n
                );
                CHECK(cudaDeviceSynchronize());  // 线程同步，等前面完成了再进行下一步
                std::cout << "size of UVWHash: " << UVWHash.size() << std::endl;
                std::cout << "m:" << m << " n:" << n << " computed UVW" << std::endl;

                // 根据坐标值对数据进行排序
                thrust::sort(UVWHash.begin(), UVWHash.end(), SortCompare());
                std::cout << "m:" << m << " n:" << n << " sorted" << std::endl;

                // 去除连续重复元素
                auto new_end = thrust::unique(thrust::device, UVWHash.begin(), UVWHash.end(), UniqueCompare());
                // UVWHash.erase(new_end, UVWHash.end());
                int lengthOfUniqueUVW = thrust::distance(UVWHash.begin(), new_end);
                std::cout << "size of Unique UVWHash: " << lengthOfUniqueUVW << std::endl;
                std::cout << "m:" << m << " n:" << n << " uniqued" << std::endl;

                // 将去重后的 UVWHash 添加到 globalUVW 的后面
                globalUVW.insert(globalUVW.end(), UVWHash.begin(), new_end);
                std::cout << "m:" << m << " n:" << n << " globalUVW" << std::endl;

                CHECK(cudaDeviceSynchronize());  // 线程同步，等前面完成了再进行下一步    
            }
        }
    }
    CHECK(cudaDeviceSynchronize());  // 线程同步，等前面完成了再进行下一步    

    // 对全局的uvw进行排序去重
    thrust::sort(thrust::device, globalUVW.begin(), globalUVW.end(), SortCompare());
    auto new_end = thrust::unique(thrust::device, globalUVW.begin(), globalUVW.end(), UniqueCompare());
    globalUVW.erase(new_end, globalUVW.end());
    std::cout << "size of globalUVW: " << globalUVW.size() << std::endl;
    CHECK(cudaDeviceSynchronize());  // 线程同步，等前面完成了再进行下一步 

     // 记录duplicate结束事件
    CHECK(cudaEventRecord(dupstop));
    CHECK(cudaEventSynchronize(dupstop));
    // 计算经过的时间
    float dupMS = 0;
    CHECK(cudaEventElapsedTime(&dupMS, dupstart, dupstop));
    printf("Frequency %f Index %d On GPU %d Duplicate Cost Time is: %f s\n", frequency, index, gpu_id, dupMS/1000);
    // 销毁事件
    cudaEventDestroy(dupstart);
    cudaEventDestroy(dupstop);

    // 记录save开始事件
    cudaEvent_t savestart, savestop;
    CHECK(cudaEventCreate(&savestart));
    CHECK(cudaEventCreate(&savestop));
    CHECK(cudaEventRecord(savestart));
    // 复制到本地，然后保存
    thrust::host_vector<thrust::tuple<float, float, float>> h_globalUVW = globalUVW;
    // 写入到文件中
    std::ostringstream fname;
    fname << "frequency_10M/uvw" << index << "frequency10M.txt";
    std::ofstream file(fname.str());
    for(size_t i = 0; i < h_globalUVW.size(); i++) {
        auto& t = h_globalUVW[i];
        file << thrust::get<0>(t) << " " << thrust::get<1>(t) << " " << thrust::get<2>(t) << std::endl;
    }
    file.close();

    // 记录save结束事件
    CHECK(cudaEventRecord(savestop));
    CHECK(cudaEventSynchronize(savestop));
    // 计算经过的时间
    float saveMS = 0;
    CHECK(cudaEventElapsedTime(&saveMS, savestart, savestop));
    printf("Frequency %f Index %d On GPU %d Save txt Cost Time is: %f s\n", frequency, index, gpu_id, saveMS/1000);
    // 销毁事件
    cudaEventDestroy(savestart);
    cudaEventDestroy(savestop);
    std::cout << "GPU Compute position successe!" << std::endl;
}


int main()
{
    int periods = 34;
    float stride = 0.01;
    float frequency = 1e7;

    // auto pos_start = std::chrono::high_resolution_clock::now();

    // for (int index=1; index<=periods; index++){
    //     MIncline(index, frequency, stride, 1); 
    // }
    
    // auto pos_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<float, std::milli> pos_ms = pos_end - pos_start;
    // std::cout << "Position Generated in " << pos_ms.count()/1000 << "s" << std::endl;

    int nDevices;
    CHECK(cudaGetDeviceCount(&nDevices));
    int periodsPerGPU = periods / nDevices; // 每个 GPU 处理的周期数
    std::cout << "Num of GPU: " << nDevices << std::endl;

    // 设置并行区中的线程数
    omp_set_num_threads(nDevices);
    #pragma omp parallel 
    {   
        int tid = omp_get_thread_num();
        cudaSetDevice(tid);
        // 计算每个 GPU 处理的周期范围
        int startPeriod = tid * periodsPerGPU+1;
        int endPeriod = (tid + 1) * periodsPerGPU+1;
        if (tid == nDevices - 1) {
            // 确保最后一个 GPU 处理所有剩余的周期
            endPeriod = periods+1;
        }
        // 循环遍历每个线程负责的周期
        for (int index = startPeriod; index < endPeriod; index++) {
            MIncline(index, frequency, stride, tid);
        }
    }
    
    return 0;
}
