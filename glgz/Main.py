# -*- coding: utf-8 -*-
# @Time     : 2018/11/21 13:00
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm

import numpy as np

from collections import OrderedDict

from pyexcel_xls import get_data
from pyexcel_xls import save_data


def read_xls_file():
    xls_data = get_data(r"datasets\4189696.xls")
    i = 0
    for sheet_n in xls_data.keys():
        if (i == 1):
            result = xls_data[sheet_n]
        i += 1
    return result


def compute_weiht(vector_result):
    np.random.seed(1)
    init_weiht = np.random.rand(195, 1)
    float_vector_result = vector_result.astype(float)
    # 分子
    no_avarage = np.dot(float_vector_result, init_weiht)
    # 分母
    count = np.sum(float_vector_result, axis=1, keepdims=True)
    count_1 = 1. / count
    # T1 T2 ... Tn
    T_sw_result = np.multiply(count_1, no_avarage)
    # sum T1 ... Tn
    sum_result = np.sum(T_sw_result)
    I_result = np.dot(float_vector_result.T, T_sw_result)
    I_sw_result = I_result / sum_result
    return I_sw_result


def compare_result(I_sw_result, compare_num):
    zeros_vector = np.zeros(I_sw_result.shape)
    zeros_vector[I_sw_result > compare_num] = 1
    return zeros_vector

def jlsjjz(vector):
    result = np.dot(vector, vector.T)
    return result


if __name__ == '__main__':
    result = read_xls_file()
    vector_result = np.array(result)
    I_sw_result = compute_weiht(vector_result)
    final = compare_result(I_sw_result, 0.35)
    matrix = jlsjjz(final)
    print(final)
