import pandas as pd
import numpy as np
import time
from pathlib import Path

def load_and_process_data():
    print("开始加载数据...")
    start_time = time.time()
    
    # 加载已排序的 l m n 数据
    print("加载 lmn 数据...")
    l = pd.read_csv('lmn10M/l.txt', skiprows=1, header=None, names=['l'])
    m = pd.read_csv('lmn10M/m.txt', header=None, names=['m'])
    n = pd.read_csv('lmn10M/n.txt', header=None, names=['n'])
    
    # 构建 lmn DataFrame
    lmn = pd.concat([l, m, n], axis=1)
    print(f"lmn 形状: {lmn.shape}")
    
    # 加载 l1 m1 n1 数据
    print("\n加载 lmn10m 数据...")
    l1 = pd.read_csv('lmn10M/l10M.txt', header=None, names=['l'])
    m1 = pd.read_csv('lmn10M/m10M.txt', header=None, names=['m'])
    n1 = pd.read_csv('lmn10M/n10M.txt', header=None, names=['n'])
    
    # 构建 lmn10m DataFrame
    lmn10m = pd.concat([l1, m1, n1], axis=1)
    print(f"lmn10m 形状: {lmn10m.shape}")
    
    print(f"\n数据加载完成，用时: {time.time() - start_time:.2f} 秒")
    return lmn, lmn10m

def compute_NX(lmn, lmn10m):
    print("\n开始计算NX...")
    start_time = time.time()
    
    # 为lmn创建索引列
    lmn['idx'] = np.arange(len(lmn))
    
    # 使用merge来找到匹配的索引
    # 这里使用left join确保lmn10m中的每一行都能在结果中出现
    merged = pd.merge(lmn10m, lmn, 
                     on=['l', 'm', 'n'], 
                     how='left',
                     suffixes=('', '_ref'))
    
    # 获取NX（即匹配的索引）
    NX = merged['idx'].fillna(-1).astype(np.int32)
    
    # 统计信息
    total_points = len(NX)
    matched_points = (NX >= 0).sum()
    unmatched_points = (NX == -1).sum()
    
    print(f"\n统计信息:")
    print(f"总点数: {total_points:,}")
    print(f"匹配成功: {matched_points:,}")
    print(f"未匹配点: {unmatched_points:,}")
    print(f"匹配率: {(matched_points/total_points*100):.2f}%")
    
    # 验证NX的范围
    if len(NX) > 0:
        print(f"\nNX范围: [{NX.min()}, {NX.max()}]")
        
        # 检查是否有超出范围的索引
        invalid_indices = (NX >= len(lmn)) & (NX != -1)
        if invalid_indices.any():
            print(f"警告: 发现 {invalid_indices.sum():,} 个超出范围的索引!")
            
            # 显示一些超出范围的例子
            if invalid_indices.any():
                print("\n前几个超出范围的点:")
                invalid_examples = lmn10m[invalid_indices].head()
                print(invalid_examples)
    
    print(f"\n计算完成，用时: {time.time() - start_time:.2f} 秒")
    return NX

def validate_matches(lmn, lmn10m, NX, num_samples=5):
    """验证一些随机样本的匹配结果"""
    print("\n验证随机样本:")
    
    # 随机选择一些索引
    sample_indices = np.random.choice(len(NX), min(num_samples, len(NX)), replace=False)
    
    for idx in sample_indices:
        original_point = lmn10m.iloc[idx]
        nx_idx = NX[idx]
        
        print(f"\n样本 {idx}:")
        print(f"原始点: ({original_point['l']}, {original_point['m']}, {original_point['n']})")
        
        if nx_idx >= 0:
            matched_point = lmn.iloc[nx_idx]
            print(f"匹配点: ({matched_point['l']}, {matched_point['m']}, {matched_point['n']})")
            
            # 验证是否完全匹配
            is_match = (original_point == matched_point[['l', 'm', 'n']]).all()
            print(f"完全匹配: {'是' if is_match else '否'}")
        else:
            print("未找到匹配点")

def main():
    # 加载数据
    lmn, lmn10m = load_and_process_data()
    
    # 计算NX
    NX = compute_NX(lmn, lmn10m)
    
    # 验证结果
    validate_matches(lmn, lmn10m, NX)
    
    # 如果需要保存结果
    np.savetxt('lmn10M/NX.txt', NX)

if __name__ == "__main__":
    main()