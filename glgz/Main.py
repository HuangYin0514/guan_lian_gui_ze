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
    return float_vector_result, I_sw_result, T_sw_result, sum_result


def compare_result(I_sw_result, compare_num):
    zeros_vector = np.zeros(I_sw_result.shape)
    zeros_vector[I_sw_result > compare_num] = 1
    return zeros_vector


def jlsjjz(vector):
    result = np.dot(vector[0:-1], vector[1:].T)
    result = np.array(result)
    return result


def pf2xj(float_vector_result, T_sw_result, sum_result, row, column):
    float_vector_result_T = float_vector_result.T
    # I1*I2 ...
    oneVector1 = float_vector_result_T[row[0], :] * float_vector_result_T[column[0]]

    flag = 0
    for i in row:
        if flag == 0:
            flag += 1
            continue

        column_tmp = column[i]
        oneVector2 = float_vector_result_T[i, :] * float_vector_result_T[column_tmp, :]
        oneVector1 = np.vstack((oneVector1, oneVector2))
        print()
    #     分子1 * 分子2 / sum
    result = (oneVector1 * T_sw_result.T)/sum_result
    return result


def findIndex(matrix):
    row, column = np.where(matrix == 1)
    column += 1
    return row, column


if __name__ == '__main__':
    result = read_xls_file()
    vector_result = np.array(result)
    float_vector_result, I_sw_result, T_sw_result, sum_result = compute_weiht(vector_result)
    final = compare_result(I_sw_result, 0.35)
    matrix = jlsjjz(final)
    row, column = findIndex(matrix)
    pf2xj_result = pf2xj(float_vector_result, T_sw_result, sum_result, row, column)
    jlsjjz_compare_result= compare_result(pf2xj_result,0.0021)
    row = np.where(jlsjjz_compare_result ==1)
    print()
