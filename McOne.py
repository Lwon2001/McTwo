import numpy as np
from minepy import MINE


def mic(x, y):
    mine = MINE()
    mine.compute_score(x, y)
    return mine.mic()


def McOne(F, C, r):
    # 初始化
    s, k = F.shape  # 将数据集经过transpose后作为输入， s为数据量， k为每一条数据的特征数量
    micFC = k * [-1]
    Subset = k * [-1]
    numSubset = 1

    # 用r初步筛选特征
    for i in range(k):
        micFC[i] = mic(F[:, i], C)  # F[:, i] 表示每一行的第i列元素，即第i个特征
        if micFC[i] >= r:
            Subset[numSubset] = i
            numSubset += 1

    # 将Subset中保存的特征标号按对应FC的大小降序排列
    Subset = Subset[0:numSubset]
    Subset.sort(key=lambda x: micFC[x], reverse=True)

    # 去除冗余特征
    flag = [True] * numSubset  # flag用来标记是否保存对应的特征
    for e in range(numSubset):
        if flag[e]:
            q = e + 1
            while q < numSubset:
                if flag[q] and mic(F[:, Subset[e]], F[:, Subset[q]]) >= micFC[Subset[q]]:
                    flag[q] = False
                q += 1

    return F[:, np.array(Subset)[flag]]  # 截取flag为True的列返回
