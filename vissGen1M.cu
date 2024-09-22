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
string viss_address = "./Viss_1M/";
string F_address = "./F_recon_1M/";
string para;
string duration = "frequency1M";  // 第几个周期的uvw
string sufix = ".txt";
const int amount = 42;  // 一共有多少个周期  15月 * 30天 / 14天/周期

// 1 M
const int uvw_presize = 4000000;


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


// 定义计算C的核函数，每一个线程处理一个q的值，q为0-nn的范围， 但是NX中保存的索引是1-nn，因此需要对齐，验证正确
__global__ void computeC(Complex *NX, Complex *FF, Complex *C, Complex *dN, int nn, int NX_size) {
    int q = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (q <= nn){
        float sumReal = 0.0;
        int count=0;
        for (int i=0; i < NX_size; ++i) {
            if (NX[i].real() == q) {
                sumReal += FF[i].real();
                count++;
            }
        }
        if (count > 0) {
            C[q-1] = Complex(sumReal/count, 0);
        }
    }
}

// 定义计算可见度核函数, 验证一致
__global__ void visscal(
            int uvw_index, int lmnC_index, 
            Complex *viss, Complex *u, Complex *v, Complex *w,
            Complex *l, Complex *m, Complex *n, Complex *nt, Complex *C,
            Complex I1, Complex CPI, Complex zero, Complex one,
            Complex two, Complex dl, Complex dm, Complex dn)
{
    int uvw_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (uvw_ < uvw_index)
    {
        viss[uvw_] = zero;
        for (int lmnC_ = 0; lmnC_ < lmnC_index; ++lmnC_) {
            viss[uvw_] = viss[uvw_] + C[lmnC_] * complexExp((zero - I1) * two * CPI * (u[uvw_]*l[lmnC_]/dl + v[uvw_]*m[lmnC_]/dm + w[uvw_]*(n[lmnC_]-one)/dn));
        } 
        viss[uvw_] *= complexExp((zero-I1) * two * CPI * w[uvw_]/dn);
    }
}


// 定义图像反演核函数  验证正确
__global__  void imagerecon(int uvw_index, int lmnC_index, 
            Complex *F, Complex *viss, Complex *u, Complex *v, Complex *w,
            Complex *l, Complex *m, Complex *n, Complex *nt, Complex *C, Complex *uvwFrequencyMap,
            Complex I1, Complex CPI, Complex zero, Complex one,
            Complex two, Complex dl, Complex dm, Complex dn)
{
    Complex amount((float)uvw_index, 0.0);
    
    int lmnC_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (lmnC_ < lmnC_index){  
        F[lmnC_] = zero;
        for(int uvw_=0; uvw_<uvw_index; ++uvw_)
        {   
            Complex temp;
            temp = uvwFrequencyMap[uvw_] * viss[uvw_] * complexExp(I1 * two * CPI * (u[uvw_]*l[lmnC_]/dl + v[uvw_]*m[lmnC_]/dm + w[uvw_]*n[lmnC_]/dn));
            F[lmnC_] = F[lmnC_] + temp;
        }
        F[lmnC_] = F[lmnC_] / amount;
    }
}


int vissGen(int id, int totalnode, int RES) 
{   
    Complex I1(0.0, 1.0);
    Complex dl((float) 2 * RES / (RES - 1), 0.0);
    Complex dm((float) 2 * RES / (RES - 1), 0.0);
    Complex dn((float) 2 * RES / (RES - 1), 0.0);
    Complex zero(0.0, 0.0);
    Complex one(1.0, 0.0);
    Complex two(2.0, 0.0);
    Complex CPI(M_PI, 0.0);

    gettimeofday(&start, NULL);
    int nDevices;
    // 设置节点数量（gpu显卡数量）
    CHECK(cudaGetDeviceCount(&nDevices));
    // 设置并行区中的线程数
    omp_set_num_threads(nDevices);
    cout << "devices: " << nDevices << endl;

    // 加载存储 l m n C nt的文件（对于不同的frequency不一样，只与frequency有关）
    string para, address_l, address_m, address_n, address_nt, address_NX, address_FF;
    ifstream lFile, mFile, nFile, ntFile, NXFile, FFFile;
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
    para = "nt";
    address_nt = address + para + sufix;
    ntFile.open(address_nt);
    cout << "address_nt: " << address_nt << endl;
    para = "NX";
    address_NX = address + para + sufix;
    NXFile.open(address_NX);
    cout << "address_NX: " << address_NX << endl;
    para = "FF";
    address_FF = address + para + sufix;
    FFFile.open(address_FF);
    cout << "address_FF: " << address_FF << endl;
    if (!lFile.is_open() || !mFile.is_open() || !nFile.is_open() || !ntFile.is_open() || !NXFile.is_open() ||!FFFile.is_open()) {
        std::cerr << "无法打开一个或多个文件：" << std::endl;
        if (!lFile.is_open()) std::cerr << "无法打开文件: " << address_l << std::endl;
        if (!mFile.is_open()) std::cerr << "无法打开文件: " << address_m << std::endl;
        if (!nFile.is_open()) std::cerr << "无法打开文件: " << address_n << std::endl;
        if (!ntFile.is_open()) std::cerr << "无法打开文件: " << address_nt << std::endl;
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

    std::vector<Complex> cl(lmnC_index), cm(lmnC_index), cn(lmnC_index), cnt(lmnC_index);
    std::vector<Complex> cNX(NX_index), cFF(NX_index);
    for (int i = 0; i < lmnC_index && lFile.good() && mFile.good() && nFile.good() && ntFile.good(); ++i) {
        lFile >> cl[i];
        mFile >> cm[i];
        nFile >> cn[i];
        ntFile >> cnt[i];
    }
    for (int i = 0; i < NX_index && NXFile.good() && FFFile.good(); ++i) {
        NXFile >> cNX[i];
        FFFile >> cFF[i];
    }
    lFile.close();
    mFile.close();
    nFile.close();
    ntFile.close();
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


    // 使用一个GPU计算处理得到C
    auto computeCStart = std::chrono::high_resolution_clock::now();
    CHECK(cudaSetDevice(0)); 
    thrust::device_vector<Complex> dNX = cNX;
    thrust::device_vector<Complex> dFF = cFF;
    thrust::device_vector<Complex> dC(lmnC_index);
    thrust::device_vector<Complex> dN = cn;
    // 调用CUDA核函数计算C
    int blockSize;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeC, 0, 0);
    int gridSize = (lmnC_index + blockSize - 1) / blockSize;;    // 线程块的数量
    cout << "Calculate C, blockSize: " << blockSize << endl;
    cout << "Calculate C, girdSize: " << gridSize << endl;
    computeC<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(dNX.data()),
        thrust::raw_pointer_cast(dFF.data()),
        thrust::raw_pointer_cast(dC.data()),
        thrust::raw_pointer_cast(dN.data()),
        lmnC_index, NX_index
    );
    CHECK(cudaDeviceSynchronize());
    std::cout << "C is computed in GPU 0!" << std::endl;

    // string computeC_dir = "computeC_div_n.txt";
    // writeToFile(dC, computeC_dir);

    // 将计算得到的C复制回主机
    std::vector<Complex> C_host(lmnC_index);
    thrust::copy(dC.begin(), dC.end(), C_host.begin());

    auto computeCFinish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> computeCElapsed = computeCFinish - computeCStart;
    cout << "Compute C Elapsed Time: " << computeCElapsed.count() << " s\n";

    // 开启cpu线程并行
    // 一个线程处理1个GPU
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();  // 从 0 开始编号的并行执行线程
        cudaSetDevice(tid);
        std::cout << "Thread " << tid << " is running on device " << tid << std::endl;

        // 将 l m n C NX 数据从cpu搬到GPU上        
        thrust::device_vector<Complex> l(cl.begin(), cl.end());
        thrust::device_vector<Complex> m(cm.begin(), cm.end());
        thrust::device_vector<Complex> n(cn.begin(), cn.end());
        thrust::device_vector<Complex> nt(cnt.begin(), cnt.end());
        thrust::device_vector<Complex> C(C_host.begin(), C_host.end());

        // 创建用来存储不同index中【u, v, w】
        std::vector<Complex> cu(uvw_presize), cv(uvw_presize), cw(uvw_presize);
        thrust::device_vector<Complex> u(uvw_presize), v(uvw_presize), w(uvw_presize);

        // 创建存储uvw坐标对应的频次
        std::vector<Complex> uvwMapVector(uvw_presize);
        thrust::device_vector<Complex> uvwFrequencyMap(uvw_presize);
       
        // 存储计算后的到的最终结果
        thrust::device_vector<Complex> F(lmnC_index);

        // 遍历所有开启的线程处理， 一个线程控制一个GPU 处理一个id*amount/total的块
        for (int p = tid + id*amount/totalnode; p < (id + 1) * amount / totalnode; p += nDevices) 
        // for (int p = tid + 96; p < (id + 1) * amount / totalnode; p += nDevices) 
        {
            cout << "for loop: " << p+1 << endl;

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
            int uvw_index;
            #pragma omp critical
            {
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
                            uvwMapVector[uvw_index] = Complex(1 / (it->second), 0);  // 存储频次的倒数
                        } else {
                            uvwMapVector[uvw_index] = Complex(1, 0); 
                        }
                        // cu, cv, cw 需要存储原始坐标
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
                thrust::copy(uvwMapVector.begin(), uvwMapVector.begin() + uvw_index, uvwFrequencyMap.begin());
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
            thrust::device_vector<Complex> viss(uvw_index);
            int blockSize;
            int minGridSize; // 最小网格大小
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, visscal, 0, 0);
            int gridSize = floor(uvw_index + blockSize - 1) / blockSize;;  
            cout << "Viss Computing, blockSize: " << blockSize << endl;
            cout << "Viss Computing, girdSize: " << gridSize << endl;
            printf("Viss Computing... Here is gpu %d running process %d on node %d\n", omp_get_thread_num(), p+1, id);
            // 调用函数计算可见度

            visscal<<<gridSize, blockSize>>>(uvw_index, lmnC_index,
                    thrust::raw_pointer_cast(viss.data()),
                    thrust::raw_pointer_cast(u.data()),
                    thrust::raw_pointer_cast(v.data()),
                    thrust::raw_pointer_cast(w.data()),
                    thrust::raw_pointer_cast(l.data()),
                    thrust::raw_pointer_cast(m.data()),
                    thrust::raw_pointer_cast(n.data()),
                    thrust::raw_pointer_cast(nt.data()),
                    thrust::raw_pointer_cast(C.data()),
                    I1, CPI, zero, one, two, dl, dm, dn);
            // 进行线程同步
            CHECK(cudaDeviceSynchronize());
            cout << "period" << p+1 << " viss compute success!" << endl;

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

            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, imagerecon, 0, 0);
            gridSize = floor(lmnC_index + blockSize - 1) / blockSize;
            cout << "Image Reconstruction, blockSize: " << blockSize << endl;
            cout << "Image Reconstruction, girdSize: " << gridSize << endl;
            printf("Image Reconstruction... Here is gpu %d running process %d on node %d\n",omp_get_thread_num(),p+1,id);
            // 调用image_recon函数计算图像反演
            imagerecon<<<gridSize,blockSize>>>(
                uvw_index, lmnC_index, 
                thrust::raw_pointer_cast(F.data()),
                thrust::raw_pointer_cast(viss.data()),
                thrust::raw_pointer_cast(u.data()),
                thrust::raw_pointer_cast(v.data()),
                thrust::raw_pointer_cast(w.data()),
                thrust::raw_pointer_cast(l.data()),
                thrust::raw_pointer_cast(m.data()),
                thrust::raw_pointer_cast(n.data()),
                thrust::raw_pointer_cast(nt.data()),
                thrust::raw_pointer_cast(C.data()),
                thrust::raw_pointer_cast(uvwFrequencyMap.data()),
                I1, CPI, zero, one, two, dl, dm, dn);
            // 进行线程同步
            CHECK(cudaDeviceSynchronize());
            cout << "Period " << p+1 << "Image Reconstruction Success!" << endl;
            
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

                std::ofstream ExFile;
                string para = "Ex";
                string address_Ex = F_address + para + to_string(p+1) + duration + sufix;
                cout << "address_Ex: " << address_Ex << endl;
                ExFile.open(address_Ex);
                if (!ExFile.is_open()) {
                    std::cerr << "Error opening file: " << address_Ex << endl;
                }
                for (int c = 0; c < NX_index; c++) {
                    int tmp = static_cast<int>(cNX[c].real()) - 1;  // matlab中下标从1开始
                    extendF[c] = tempF[tmp];
                    ExFile << extendF[c].real() << std::endl;
                }
                ExFile.close();
                std::cout << "save success!" << std::endl;
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

            // 记录结束事件
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
    vissGen(0, 1, 2094);
}

