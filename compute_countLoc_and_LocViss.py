import numpy as np
import pandas as pd
from pathlib import Path
import time

def process_data():
    print(f"开始处理数据... 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    total_start_time = time.time()
    
    # 1. 加载NX.txt
    print("加载NX.txt...")
    load_start = time.time()
    # 直接使用numpy加载，比pandas快
    NX = np.loadtxt('lmn10M/NX.txt', dtype=np.float32)
    print(f"NX加载完成，用时：{time.time() - load_start:.2f}秒")
    print(f"NX shape: {NX.shape}")
    
    # 2. 计算countLoc
    print("计算countLoc...")
    count_start = time.time()
    # 转换为整数类型
    NX_int = NX.astype(np.int32)
    # 使用numpy的unique函数直接计算计数
    unique_values, counts = np.unique(NX_int, return_counts=True)
    
    # 创建完整的countLoc数组
    countLoc = np.zeros(396005546, dtype=np.int32)
    countLoc[unique_values] = counts
    
    print(f"countLoc计算完成，用时：{time.time() - count_start:.2f}秒")
    
    # 保存countLoc
    print("保存countLoc.txt...")
    save_start = time.time()
    np.savetxt('lmn10M/countLoc.txt', countLoc, fmt='%d')
    print(f"countLoc保存完成，用时：{time.time() - save_start:.2f}秒")
    
    # 3. 计算NXq
    print("计算NXq...")
    nxq_start = time.time()
    
    # 计算前缀和，用于确定每个值的起始位置
    positions = np.zeros_like(countLoc, dtype=np.int32)
    positions[1:] = np.cumsum(countLoc[:-1])
    
    # 创建输出数组
    NXq = np.zeros(438483600, dtype=np.int32)
    
    # 创建临时数组记录当前位置
    current_positions = positions.copy()
    
    # 使用numpy的高效操作填充NXq
    original_indices = np.arange(len(NX_int), dtype=np.int32)
    for i in range(len(NX_int)):
        value = NX_int[i]
        pos = current_positions[value]
        NXq[pos] = original_indices[i]
        current_positions[value] += 1
    
    print(f"NXq计算完成，用时：{time.time() - nxq_start:.2f}秒")
    
    # 保存NXq
    print("保存NXq.txt...")
    save_start = time.time()
    np.savetxt('lmn10M/NXq.txt', NXq, fmt='%d')
    print(f"NXq保存完成，用时：{time.time() - save_start:.2f}秒")
    
    total_time = time.time() - total_start_time
    print(f"\n处理完成！总用时：{total_time:.2f}秒")
    
    # 输出详细的统计信息
    print(f"\n统计信息：")
    print(f"NX大小：{len(NX):,} 个元素")
    print(f"countLoc大小：{len(countLoc):,} 个元素")
    print(f"NXq大小：{len(NXq):,} 个元素")
    print(f"countLoc中非零元素数量：{np.count_nonzero(countLoc):,}")
    print(f"数据类型：")
    print(f"- NX: {NX.dtype}")
    print(f"- countLoc: {countLoc.dtype}")
    print(f"- NXq: {NXq.dtype}")
    print(f"内存使用（估计）：")
    print(f"- NX: {NX.nbytes / 1024**3:.2f} GB")
    print(f"- countLoc: {countLoc.nbytes / 1024**3:.2f} GB")
    print(f"- NXq: {NXq.nbytes / 1024**3:.2f} GB")

def verify_results():
    """验证结果的正确性"""
    print("\n开始验证结果...")
    
    # 加载所有文件
    NX = np.loadtxt('lmn10M/NX.txt', dtype=np.float32)
    countLoc = np.loadtxt('lmn10M/countLoc.txt', dtype=np.int32)
    NXq = np.loadtxt('lmn10M/NXq.txt', dtype=np.int32)
    
    # 验证1：检查countLoc的计算是否正确
    test_counts = np.zeros_like(countLoc)
    unique_values, counts = np.unique(NX.astype(np.int32), return_counts=True)
    test_counts[unique_values] = counts
    assert np.array_equal(countLoc, test_counts), "countLoc计算错误"
    
    # 验证2：检查NXq中的索引是否正确
    for i, idx in enumerate(NXq):
        assert NX[idx] == NX[NXq[i]], f"NXq索引错误at position {i}"
    
    print("验证完成：结果正确！")

if __name__ == "__main__":
    # 设置numpy显示选项
    np.set_printoptions(precision=3, suppress=True)
    
    try:
        process_data()
        verify_results()  # 添加验证步骤
    except Exception as e:
        print(f"处理过程中出现错误：{str(e)}")