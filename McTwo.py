import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def calcuBAcc(FS, C):
    """
    用k=1的KNN网络计算当前所选特征值的BAcc值
    :FS 要计算BAcc值的特征集
    :C Label
    :return 特征集FS运用到KNN网络中的BAcc值
    """
    FeatureNum, _ = FS.shape
    C = C.astype('int')
    NN = KNeighborsClassifier(n_neighbors=1)
    pred = []
    # Leave-One-Out: 使k等于数据集中数据的个数，每次只使用一个作为测试集，剩下的全部作为训练集
    for i in range(FeatureNum):
        NN.fit(FS[[x for x in range(FeatureNum) if x != i]],
               C[[x for x in range(FeatureNum) if x != i]])
        pred.append(NN.predict(FS[[i]]).tolist()[0])
    pred = np.array(pred)

    BAcc = (np.mean(pred[np.where(C == 0)] == C[np.where(C == 0)]) +
            np.mean(pred[np.where(C == 1)] == C[np.where(C == 1)])) / 2

    return BAcc


def McTwo(FR, C):
    """
    采用最佳优先算法，以特征集的BAcc值作为评估标准，
    :FR Reduced Features from McOne
    :C Label
    :return 经McTwo筛选后的特征集
    """
    s, k = FR.shape  # s为数据量，k为每一条数据的特征数
    curMaxBAcc = -1  # 当前的最大BAcc值
    curSet = set([])  # 当前的最大BAcc值所用的特征集
    leftSet = set([x for x in range(k)])  # 未用到的特征集

    # best first search策略，得到的是次优解
    while True:
        maxBAcc, maxIndex = -1, -1
        #  每次试探性的添加一个特征到当前特征集中，如果计算得到的BAcc值更大，那么则加入到当前特征集中，否则结束
        for x in leftSet:
            tmpBAcc = calcuBAcc(FR[:, list(curSet) + [x]], C)
            if tmpBAcc > maxBAcc:
                maxBAcc = tmpBAcc
                maxIndex = x
        if maxBAcc > curMaxBAcc:
            curMaxBAcc = maxBAcc
            curSet.add(maxIndex)
            leftSet.remove(maxIndex)
        else:
            break

    return FR[:, list(curSet)]
