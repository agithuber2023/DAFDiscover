import numpy as np
import math
import os
import pandas as pd
import random


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=1.0):  # 创造一个指定大小的矩阵
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def get_0_1_array(m, n, rate=0.2):
    array = np.ones(m * n).reshape(m, n)
    zeros_num = int(array.size * rate)  # 根据0的比率来得到 0的个数
    new_array = np.ones(array.size)  # 生成与原来模板相同的矩阵，全为1
    new_array[:zeros_num] = 0  # 将一部分换为0
    np.random.shuffle(new_array)  # 将0和1的顺序打乱
    re_array = new_array.reshape(array.shape)  # 重新定义矩阵的维度，与模板相同
    return re_array.tolist()


def read_data():
    fileHandler1 = open("train_data", "r")
    fileHandler2 = open("test_data", "r")
    listOfLines1 = fileHandler1.readlines()
    listOfLines2 = fileHandler2.readlines()
    data1 = []
    data2 = []

    for line in listOfLines1:
        newline = line.strip("\n").split(",")
        newline_ = [float(x) for x in newline]
        data1.append(newline_)

    for line in listOfLines2:
        newline = line.strip("\n").split(",")
        newline_ = [float(x) for x in newline]
        data2.append(newline_)
    data1 = data2[:100]
    data2 = data2[:1000]
    return np.array(data1), np.array(data2)


def get_index(x_list, i):
    res = 0
    for j in x_list:
        if j > x_list[i]:
            res += 1
    return res


def my_sort(x_list, k):
    res = []
    for i in range(len(x_list[0])):
        xx = []
        for j in range(len(x_list)):
            xx.append(x_list[j][i])
        res.append(get_index(xx, k))
    return res


def select_id(beta_1):
    min_value = []
    for j in range(len(beta_1[0])):
        maxn = 0
        for i in range(len(beta_1)):
            maxn = max(maxn, beta_1[i][j])
        min_value.append(maxn)
    return min_value




class AFD:
    def __init__(self, LHS_list, RHS):
        self.LHS = LHS_list
        self.RHS = RHS

    def print(self):
        for i in self.LHS:
            print("%d" % i, end=',')
        i = self.RHS
        print("%d" % i)


def work(data, x):
    x.generator(data)
    return x


if __name__ == "__main__":
    path = os.getcwd() + '\\123.csv'
    f = open(path, encoding='utf-8')
    data = pd.read_csv(f)
    data = data[~data['result'].isin([-1])]
    name_list = ["H2", "CH4", "C2H4", "C2H2", "C2H6", "CO", "CO2", "O2", "N2", "TOTALHYDROCARBON"]
    data_x = data[name_list]
    data_y = data['result']
    x = data_x.values
    y = data_y.values
    if len(x) == 0:
        print("all -1")
        exit(0)
    from sklearn.preprocessing import StandardScaler

    stand1 = StandardScaler()
    stand1.fit(x)
    x = stand1.transform(x)
    print(x.shape)

    data = x
    f = FD([1, 4, 5], 0)
    x = RFD(f, 0.477055385001098)
    y = work(data, x)
    x.print()
