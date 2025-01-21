import numpy as np
import pandas as pd
    
def merge(lmn, l1, m1, n1, period):
    
    values = pd.read_csv(f'F_recon_10M/F{period}period10M_optim1.txt', header=None, names=['value'])
    lmn_dup = pd.concat([l1, m1, n1, values], axis=1)

    print("lmn.shape = ", lmn.shape)    # (438483600, 3)
    print("lmn_dup.shape = ", lmn_dup.shape)  # (396005546, 3)

    # 合并
    merge_df = pd.merge(lmn, lmn_dup, on=['l','m','n'], how='left')
    # 检查
    last_column_has_nan = merge_df.iloc[:, -1].isna().any()
    if last_column_has_nan:
        print("最后一列包含NaN值。")
    else:
        print("最后一列不包含NaN值。")
    
    merge_df['value'].to_csv(f'F_recon_10M/period{period}_optim1.txt', header=False, index=False)
    print("保存完成！")

if __name__ == "__main__":

    # 读取原始数据
    l = pd.read_csv('lmn10M/l10M.txt', header=None, names=['l'])
    m = pd.read_csv('lmn10M/m10M.txt', header=None, names=['m'])
    n = pd.read_csv('lmn10M/n10M.txt', header=None, names=['n'])
    lmn = pd.concat([l, m, n], axis=1)
    del l, m, n

    # 读取去重数据
    l1 = pd.read_csv('lmn10M/l.txt', header=None, names=['l'], skiprows=1)
    m1 = pd.read_csv('lmn10M/m.txt', header=None, names=['m'])
    n1 = pd.read_csv('lmn10M/n.txt', header=None, names=['n'])

    for p in range(1, 2):
        print("period: ", p)
        merge(lmn, l1, m1, n1, p)