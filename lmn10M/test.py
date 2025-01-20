import numpy as np
import time
from pathlib import Path

def load_coordinates(folder_path):
    """
    从指定文件夹读取l.txt, m.txt和n.txt文件
    """
    print("开始读取数据文件...")
    start_time = time.time()
    
    # 构建文件路径
    folder = Path(folder_path)
    l_path = folder / 'l.txt'
    m_path = folder / 'm.txt'
    n_path = folder / 'n.txt'
    
    # 读取文件
    try:
        l = np.loadtxt(l_path, skiprows=1)
        print(f"已读取 l.txt, 形状: {l.shape}")
        m = np.loadtxt(m_path)
        print(f"已读取 m.txt, 形状: {m.shape}")
        n = np.loadtxt(n_path)
        print(f"已读取 n.txt, 形状: {n.shape}")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None, None, None
    
    end_time = time.time()
    print(f"数据读取完成，用时: {end_time - start_time:.2f} 秒")
    
    return l, m, n

def check_sorting(l, m, n, max_violations=5):
    """
    检查三维坐标是否已排序
    返回是否排序以及前几个违反排序的位置
    """
    print("\n开始检查排序状态...")
    start_time = time.time()
    
    points = np.column_stack((l, m, n))
    violations = []
    is_sorted = True
    
    for i in range(len(points)-1):
        current = points[i]
        next_point = points[i+1]
        
        # 比较规则与CUDA代码相同
        if (current[0] > next_point[0] or 
            (current[0] == next_point[0] and current[1] > next_point[1]) or
            (current[0] == next_point[0] and current[1] == next_point[1] and current[2] > next_point[2])):
            
            is_sorted = False
            violations.append({
                'index': i,
                'current': current,
                'next': next_point
            })
            
            if len(violations) >= max_violations:
                break
    
    end_time = time.time()
    print(f"排序检查完成，用时: {end_time - start_time:.2f} 秒")
    
    return is_sorted, violations

def main():
    # 读取数据
    folder_path = "./"  # 请根据实际路径修改
    l, m, n = load_coordinates(folder_path)
    
    if l is None:
        return
    
    # 打印数据基本信息
    print("\n数据统计信息:")
    print(f"数据点数量: {len(l)}")
    print(f"l范围: [{l.min():.6f}, {l.max():.6f}]")
    print(f"m范围: [{m.min():.6f}, {m.max():.6f}]")
    print(f"n范围: [{n.min():.6f}, {n.max():.6f}]")
    
    # 检查排序状态
    is_sorted, violations = check_sorting(l, m, n)
    
    if is_sorted:
        print("\n✅ 数据已正确排序")
    else:
        print("\n❌ 数据未正确排序")
        print("\n前几个违反排序的位置:")
        for v in violations:
            print(f"\n位置 {v['index']}:")
            print(f"当前点:  ({v['current'][0]:.6f}, {v['current'][1]:.6f}, {v['current'][2]:.6f})")
            print(f"下一个点: ({v['next'][0]:.6f}, {v['next'][1]:.6f}, {v['next'][2]:.6f})")

if __name__ == "__main__":
    main()