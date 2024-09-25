'''
@File    :   pca_utils.py
@Time    :   2023/08/13 17:38:14
@Author  :   Chen Zhu 
@Version :   1.0
@Contact :   zc1213856@163.com
'''

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


def pca(data, n_dim, picked_eig_vector=None):
    '''

    pca is O(D^3)
    :param data: (n_samples, n_features(D))
    :param n_dim: target dimensions
    :return: (n_samples, n_dim)
    '''
    data = data - np.mean(data, axis = 0, keepdims = True)

    cov = np.dot(data.T, data)

    eig_values, eig_vector = np.linalg.eig(cov)
    # print(eig_values)
    indexs_ = np.argsort(-eig_values)[:n_dim]
    picked_eig_values = eig_values[indexs_]
    if picked_eig_vector is None:
        picked_eig_vector = eig_vector[:, indexs_]
    data_ndim = np.dot(data, picked_eig_vector)
    return data_ndim, picked_eig_vector


# data 降维的矩阵(n_samples, n_features)
# n_dim 目标维度
# fit n_features >> n_samples, reduce cal
def highdim_pca(data, n_dim):
    '''

    when n_features(D) >> n_samples(N), highdim_pca is O(N^3)

    :param data: (n_samples, n_features)
    :param n_dim: target dimensions
    :return: (n_samples, n_dim)
    '''
    N = data.shape[0]
    data = data - np.mean(data, axis = 0, keepdims = True)

    Ncov = np.dot(data, data.T)

    Neig_values, Neig_vector = np.linalg.eig(Ncov)
    indexs_ = np.argsort(-Neig_values)[:n_dim]
    Npicked_eig_values = Neig_values[indexs_]
    # print(Npicked_eig_values)
    Npicked_eig_vector = Neig_vector[:, indexs_]
    # print(Npicked_eig_vector.shape)

    picked_eig_vector = np.dot(data.T, Npicked_eig_vector)
    picked_eig_vector = picked_eig_vector/(N*Npicked_eig_values.reshape(-1, n_dim))**0.5
    # print(picked_eig_vector.shape)

    data_ndim = np.dot(data, picked_eig_vector)
    return data_ndim

def vis_pca_2d(data):
    # data: list of np.array 
    color = ['b','g','r','c','m','y','k','w']
    plt.figure(figsize=(4,4))
    plt.title("PCA")
    for i, x in enumerate(data):
        r, c = x.shape
        mean = np.mean(x,axis = 0,keepdims=True)
        std = np.std(x,axis = 0,keepdims=True)
        x = (x - mean) / std
        if i==0:
            a, vector = pca(x,2)
        else:
            a, _ = pca(x,2,vector)
        plt.scatter(a[:, 0], a[:, 1], c = np.array([ii % 7 for ii in range(r)]))
        # plt.scatter(a[:, 0], a[:, 1], c = np.array([color[i]]*r))
    plt.show()
    print(vector)
        # sklearn_pca = PCA(n_components=2, copy=True)
        # s_x = preprocessing.scale(x)
        # PCA.fit(s_x)
        # print('debug')
    



if __name__ == "__main__":
    data = load_iris()
    X = data.data
    Y = data.target
    data_2d1 = pca(X, 2)
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("my_PCA")
    plt.scatter(data_2d1[:, 0], data_2d1[:, 1], c = Y)

    sklearn_pca = PCA(n_components=2)
    data_2d2 = sklearn_pca.fit_transform(X)
    plt.subplot(122)
    plt.title("sklearn_PCA")
    plt.scatter(data_2d2[:, 0], data_2d2[:, 1], c = Y)
    plt.show()


