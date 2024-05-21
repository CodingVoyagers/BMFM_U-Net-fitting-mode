# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/1/3
'''
    计算二维矩阵的一阶统计特征
'''

import os
import math
import numpy as np
import time
# 判断该矩阵是否是二维矩阵，返回True表示是二维矩阵
def is2DArray(array):
    demension = np.array(array).shape
    return True if demension == 2 else False

# 判断该路径是否存在，不存在则创建并返回True，存在则直接返回False
def isPathExists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False

# 一阶统计特征
class FirstOrderStatistics(object):
    def __init__(self):
        pass
    
    def getFirstOrderStatistics(self, array):
        return {'energy': self.getEnergy(array),
                'entropy': self.getEntropy(array),
                'krutosis' : self.getKurtosis(array),
                'max': self.getMaximum(array),
                'min': self.getMinimum(array),
                'mean': self.getMean(array),
                #'absDev': self.getAbsoluteDeviation(array),
                'median': self.getMedian(array),
                'range': self.getRange(array),
                'RMS': self.getRootMeanSquare(array),
                'skewness': self.getSkewness(array),
                'stdDev': self.getStandardDeviation(array),
                'uniformity': self.getUniformity(array),
                'variance': self.getVariance(array)}
    

    def getN(self, array):
        width = array.shape[0]
        height = array.shape[1]
        return width * height

    def getHistogram(self, array):
        num = int(self.getMaximum(array)) + 10
        histogram = [0 for i in range(num)]
        width, height = array.shape
        for i in range(width):
            for j in range(height):
                histogram[int(array[i, j])] += 1

        return [float(i/self.getN(array)) for i in histogram]

    # 能量
    def getEnergy(self, array):
        square = np.multiply(array, array)
        return square.sum()

    # 熵
    def getEntropy(self, array):
        h = np.array(self.getHistogram(array))
        tmp = h * np.log2(h)
        tmp[np.isnan(tmp)]=0
        return np.apply_over_axes(np.sum, tmp, axes=(0))[0]

    # 峰度
    def getKurtosis(self, array):
        N = self.getN(array)
        numerator = np.power(array - array.mean(), 4).sum() / N
        denominator = np.power(np.sqrt(np.power(array - array.mean(), 2).sum() / N), 2)
        return numerator / denominator

    # 最大值
    def getMaximum(self, array):
        return array.max()

    # 最小值
    def getMinimum(self, array):
        return array.min()

    # 均值
    def getMean(self, array):
        return array.mean()

    def getAbsoluteDeviation(self, array):
        pass

    # 中值
    def getMedian(self, array):
        return np.median(array)

    # 波动范围
    def getRange(self, array):
        max_pixel = np.max(array)
        min_pixel = np.min(array)
        return max_pixel - min_pixel

    # 均方根
    def getRootMeanSquare(self, array):
        N = self.getN(array)
        under_sqrt = (np.multiply(array, array).sum()) / N
        return np.sqrt(under_sqrt)

    # 偏度
    def getSkewness(self, array):
        N = self.getN(array)
        numerator = np.power(array - array.mean(), 3).sum() / N
        denominator = np.power(np.sqrt(np.power(array - array.mean(), 2).sum() / N), 3)
        return numerator / denominator

    # 标准偏差
    def getStandardDeviation(self, array):
        N = self.getN(array)
        numerator = np.power(array - array.mean(), 2).sum()
        denominator = N - 1
        return np.power(numerator / denominator, 0.5)

    # 联合度
    def getUniformity(self, array):
        histogram = self.getHistogram(array)
        return sum([i**2 for i in histogram])

    # 方差
    def getVariance(self, array):
        N = self.getN(array)
        numerator = np.power(array - array.mean(), 2).sum()
        denominator = N - 1
        return numerator / denominator


if __name__ == '__main__':
    # if 1 == 1:
    #     print('程序正常退出！')
    #     exit()
    # else:
    #     print('异常退出')
    start = time.clock()
    s = FirstOrderStatistics()
    Img = np.array([[0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234,0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234],
                    [0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234,0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234],
                    [0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234,0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234],
                    [0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234,0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234],
                    [0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234,0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234],
                    [0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234,0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234],
                    [0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234,0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234],
                    [0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234,0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234],
                    [0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234,0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234],
                    [0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234,0.32234523424, 0.23423223423, 0.234252342, 0.234244234, 0.243424234]])
    a = s.getFirstOrderStatistics(Img)
    end = time.clock()
    print(end-start)
    print(a)
