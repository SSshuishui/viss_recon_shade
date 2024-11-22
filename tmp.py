import numpy as np
import pandas as pd

# C = pd.read_csv('frequency_01M/C01M.txt', header=None, names=['C'])
# FF = pd.read_csv('frequency_01M/FF01M.txt', header=None, names=['FF'])
# NX = pd.read_csv('frequency_01M/NX01M.txt', header=None, names=['NX'])


# print(C.shape, FF.shape, NX.shape)

# for idx in range(C.shape[0]):
#     print("idx: ", idx)
#     if C['C'].loc[NX['NX'].loc[idx]-1] == FF['FF'].loc[idx]:
#         print("------")
#         print("FF: ", FF['FF'].loc[idx])
#         print("C: ", C['C'].loc[idx])
#         print("NX: ", NX['NX'].loc[idx]-1)
#         print("C[NX]: ", C['C'].loc[NX['NX'].loc[idx]-1])


C = pd.read_csv('frequency_01M/C01M_.txt', header=None, names=['C'])
FF = pd.read_csv('frequency_01M/FF01M.txt', header=None, names=['FF'])
NX = pd.read_csv('frequency_01M/NX01M.txt', header=None, names=['NX'])


print(C.shape, FF.shape, NX.shape)

for idx in range(C.shape[0]):
    print("------")

    print("idx: ", idx)
    print("C[NX]: ", C['C'].loc[NX['NX'].loc[idx]-1])
    print("FF: ", FF['FF'].loc[idx])
