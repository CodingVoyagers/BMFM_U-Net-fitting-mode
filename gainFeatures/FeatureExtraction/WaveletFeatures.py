# # _*_ encoding:utf-8 _*_
# # Author: Lg
# # Date: 19/4/8
# '''
#     对矩阵进行小波变化，提取小波特征：
#         1. 使用symlet小波将矩阵进行多级分解
#         2. 从小波分解中提取水平、垂直和对角线细节系数
#         3. 计算细节系数的均值和方差得到小波特征
# '''
# import cv2
# import pywt   # 小波分解工具包
# import numpy as np
# from pywt import dwt2

# class Wavelet(object):
#     def __init__(self):
#         pass


#     def getWaveletFeatures(self, array):
#         level = 4
#         wavelet = 'sym9'
#         rows, cols = array.shape
#         coeffs = self.my_wavedec2(array, wavelet=wavelet, level=level)
#         return coeffs


#     # 实现wavedec2()，省去了当数据size过小时不能按照预定层数分解
#     def my_wavedec2(self, data, wavelet, mode='symmetric', level=None, axes=(-2, -1)):
#         data = np.asarray(data)
#         axes = tuple(axes)
#         coeffs_list = []

#         a = data
#         for i in range(level):
#             a, ds = dwt2(a, wavelet, mode, axes)
#             coeffs_list.append(ds)

#         coeffs_list.append(a)
#         coeffs_list.reverse()

#         return coeffs_list


#     # def wavelet_extracts(self, array):
#     #     wavelet = 'sym9'
#     #     img_array = sitk.GetArrayFromImage(image)
#     #     _, rows, cols = img_array.shape
#     #     print(rows, cols)
#     #     array_1d = img_array.reshape([rows, cols])

#     #     coeffs = my_wavedec2(array_1d, wavelet=wavelet, level=4)
#     #     print(len(coeffs))
#     #     print('coeffs_0: ', len(coeffs[0]), '*', len(coeffs[0][0]))
#     #     print('coeffs_1: ', format(len(coeffs[1][0])))
#     #     print('coeffs_2: ', format(len(coeffs[2][0])))
#     #     print('coeffs_3: ', format(len(coeffs[3][0])))
#     #     print('coeffs_4: ', format(len(coeffs[4][0])))

# if __name__ == '__main__':
#     array = np.ones((20, 20))
#     wavelet = Wavelet()
#     f = wavelet.getWaveletFeatures(array)
#     print(f[4], len(f[4][0]))
     