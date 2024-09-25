'''
@File    :   t_sne_utils.py
@Time    :   2023/10/05 12:25:04
@Author  :   Chen Zhu 
@Version :   1.0
@Contact :   zc1213856@163.com
'''

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def t_sne(data, labels):
    # data: np.shape = (nsample, s1,s2,s3...)
    # label: np.shape = (nsample)
    # 创建 t-SNE 模型并拟合数据
    # 将矩阵拉成形状为 (n, X) 的矩阵，这里 X = m * x * y
    reshaped_data = np.reshape(data, (data.shape[0], -1))
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(reshaped_data)

    # 根据标签值设置颜色映射
    cmap = plt.cm.get_cmap("viridis", len(np.unique(labels)))

    # 绘制 t-SNE 可视化结果，并根据标签值着色
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap=cmap)
    plt.colorbar(ticks=np.unique(labels))
    plt.title("t-SNE Visualization with Labels")
    plt.show()


if __name__ == '__main__':
    # 创建示例高维数据和标签
    X = np.random.rand(100, 10, 3, 5)  # 假设有 100 个样本，每个样本有 10 个特征
    labels = np.random.randint(0, 5, 100)  # 假设有 5 个类别，随机生成 100 个标签
    t_sne(X, labels)