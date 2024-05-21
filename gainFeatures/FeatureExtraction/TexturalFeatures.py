# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/1/4
'''
    提取二维矩阵的纹理特征信息，包括GLCM（灰度共生矩阵）、GLRLM（灰度游程矩阵）、LBP（局部二值模式）和HOG（方向梯度直方图）
'''
import warnings
warnings.filterwarnings('ignore')
import os
import math
import numpy as np
from scipy.stats import tmax
from skimage import feature
from skimage.feature import greycomatrix, greycoprops
from itertools import groupby
import time

class GLCM(object):
    '''
    Gray-Level Co-Occurrence Matrix based features
    '''
    def __init__(self):
        pass

    def getGLCMFeatures(self, array):
        distance = [1]
        theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        levels = tmax(array, axis=None) + 1
        comatrix = self.getGrayLevelComatrix(array, distance, theta, levels)

        contrast = self.getContrast(comatrix)[0]
        dissimilarity = self.getDissimilarity(comatrix)[0]
        homogeneity1 = self.getHomogeneity1(comatrix)[0]
        homogeneity2 = self.getHomogeneity2(comatrix)[0]
        ASM = self.getASM(comatrix)[0]
        energy = self.getEnergy(comatrix)[0]
        # correlation = self.getCorrelation(comatrix)[0]
        autocorrleation = self.getAutocorrelation(comatrix)[0]
        # entropy = self.getEntropy(comatrix)[0]
        inverseVariance = self.getInverseVariance(comatrix)[0]
        sumAverage = self.getSUMAverage(comatrix)[0]
        # sumEntropy = self.getSUMEntropy(comatrix)[0]
        # differenceEntropy = self.getDifferenceEntropy(comatrix)[0]
        sumVariance = self.getSUMVariance(comatrix)[0]
        maxPorbability = self.getMaximumPorbabiliy(comatrix)[0]

        return {'Contrast_0': contrast[0],   # Contrast
                'Contrast_90': contrast[1],
                'Contrast_180': contrast[2],
                'Contrast_270': contrast[3],
                'Dissimility_0': dissimilarity[0],   # Dissimility
                'Dissimility_90': dissimilarity[1],
                'Dissimility_180': dissimilarity[2],
                'Dissimility_270': dissimilarity[3],
                'Homogeneity1_0': homogeneity1[0],   # Homogeneity1
                'Homogeneity1_90': homogeneity1[1],
                'Homogeneity1_180': homogeneity1[2],
                'Homogeneity1_270': homogeneity1[3],
                'Homogeneity2_0': homogeneity2[0],   # Homogeneity2
                'Homogeneity2_90': homogeneity2[1],
                'Homogeneity2_180': homogeneity2[2],
                'Homogeneity2_270': homogeneity2[3],
                'ASM_0': ASM[0],   # ASM
                'ASM_90': ASM[1],
                'ASM_180': ASM[2],
                'ASM_270': ASM[3],
                'Energy_0': energy[0],   # Energy
                'Energy_90': energy[1],
                'Energy_180': energy[2],
                'Energy_270': energy[3],
                # 'Correlaion_0': correlation[0],   # Correlation
                # 'Correlaion_90': correlation[1],
                # 'Correlaion_180': correlation[2],
                # 'Correlaion_270': correlation[3],
                'AutoCorrelation_0': autocorrleation[0],   # AutoCorrelation
                'AutoCorrelation_90': autocorrleation[1],
                'AutoCorrelation_180': autocorrleation[2],
                'AutoCorrelation_270': autocorrleation[3],
                # 'Entropy_0': entropy[0],   # Entropy
                # 'Entropy_90': entropy[1],
                # 'Entropy_180': entropy[2],
                # 'Entropy_270': entropy[3],
                'InverseVar_0': inverseVariance[0],   # InverseVariance
                'InverseVar_90': inverseVariance[1],
                'InverseVar_180': inverseVariance[2],
                'InverseVar_270': inverseVariance[3],
                'SUMAverage_0': sumAverage[0],   # SUMAVerage
                'SUMAverage_90': sumAverage[1],
                'SUMAverage_180': sumAverage[2],
                'SUMAverage_270': sumAverage[3],
                # 'SUMEntropy_0': sumEntropy[0],   # SUMEntropy
                # 'SUMEntropy_90': sumEntropy[1],
                # 'SUMEntropy_180': sumEntropy[2],
                # 'SUMEntropy_270': sumEntropy[3],
                # 'DiffEntropy_0': differenceEntropy[0],   # DifferenceEntropy
                # 'DiffEntropy_90': differenceEntropy[1],
                # 'DiffEntropy_180': differenceEntropy[2],
                # 'DiffEntropy_270': differenceEntropy[3],
                'SUMVariance_0': sumVariance[0],   # SUMVariance
                'SUMVariance_90': sumVariance[1],
                'SUMVariance_180': sumVariance[2],
                'SUMVariance_270': sumVariance[3],
                'MaxProbabiliy_0': maxPorbability[0],
                'MaxProbabiliy_90': maxPorbability[1],
                'MaxProbabiliy_180': maxPorbability[2],
                'MaxProbabiliy_270': maxPorbability[3]}

        # return None

    def calcuteIJ(self, comatrix):
        (num_level, num_level2, num_dist, num_angle) = comatrix.shape
        assert num_level == num_level2
        assert num_dist > 0
        assert num_angle > 0
        I, J = np.ogrid[0:num_level, 0:num_level]
        return I, J, num_level

    def getGrayLevelComatrix(self, array, distance, theta, levels):
        return greycomatrix(array, distance, theta, levels)

    # 1.
    def getContrast(self, comatrix):
        return greycoprops(comatrix, 'contrast')

    # 2.
    def getDissimilarity(self, comatrix):
        return greycoprops(comatrix, 'dissimilarity')

    # 3.
    def getHomogeneity1(self, comatrix):
        I, J, num_level = self.calcuteIJ(comatrix)
        weights = 1. / (1. + np.abs(I - J))
        weights = weights.reshape((num_level, num_level, 1, 1))
        return np.apply_over_axes(np.sum, (comatrix * weights), axes=(0, 1))[0, 0]

    # 4.
    def getHomogeneity2(self, comatrix):
        return greycoprops(comatrix, 'homogeneity')

    # 5.
    def getASM(self, comatrix):
        return greycoprops(comatrix, 'ASM')

    # 6.
    def getEnergy(self, comatrix):
        return greycoprops(comatrix, 'energy')

    # 7.
    def getCorrelation(self, comatrix):
        return greycoprops(comatrix, 'correlation')

    # 8.
    def getAutocorrelation(self, comatrix):
        I, J, num_level = self.calcuteIJ(comatrix)
        weights = I * J
        weights = weights.reshape((num_level, num_level, 1, 1))
        return np.apply_over_axes(np.sum, (comatrix * weights), axes=(0, 1))[0, 0]

    # 9.
    def getEntropy(self, comatrix):
        log = np.log2(comatrix)
        return - np.apply_over_axes(np.sum, (comatrix * log), axes=(0, 1))[0, 0]

    # 10.
    def getInverseVariance(self, comatrix):
        I, J, num_level = self.calcuteIJ(comatrix)
        equals = np.array(I == J, dtype=np.float)
        weights = 1. / ((I - J + equals) ** 2) - equals   # 计算i != j，此处先将i == j置为1，计算除法，完成之后将其再置为0
        weights = weights.reshape((num_level, num_level, 1, 1))
        return np.apply_over_axes(np.sum, (comatrix * weights), axes=(0, 1))[0, 0]

    # 11.
    def getSUMAverage(self, comatrix):
        I, J, num_level = self.calcuteIJ(comatrix)
        pxPlusy = np.apply_over_axes(np.sum, comatrix, axes=(0, 1))[0, 0]
        i = np.array([[j] for j in range(num_level * 2 - 2)]).reshape(-1, 1)
        # i = i.reshape(-1, 1)
        return np.apply_over_axes(np.sum, i * pxPlusy, axes=(0))

    # 12.
    def getSUMEntropy(self, comatrix):
        I, J, num_level = self.calcuteIJ(comatrix)
        pxPlusy = np.apply_over_axes(np.sum, comatrix, axes=(0, 1))[0, 0]
        return -np.apply_over_axes(np.sum, pxPlusy * np.log2(pxPlusy), axes=(0))
        
    # 13.
    def getDifferenceEntropy(self, comatrix):
        I, J, num_level = self.calcuteIJ(comatrix)
        pxMinusy = np.apply_over_axes(np.sum, comatrix, axes=(0, 1))[0, 0]
        i = np.array([[j] for j in range(num_level - 1)]).reshpae(-1, 1)
        # i = i.reshape(-1, 1)
        return np.apply_over_axes(np.sum, pxMinusy * np.log2(pxMinusy), axes=(0))

    # 14.
    def getSUMVariance(self, comatrix):
        I, J, num_level = self.calcuteIJ(comatrix)
        pxPlusy = np.apply_over_axes(np.sum, comatrix, axes=(0, 1))[0, 0]
        SE = self.getSUMEntropy(comatrix)
        i = np.array([[j] for j in range(num_level * 2 - 2)]).reshape(-1, 1)
        return np.apply_over_axes(np.sum, (i - SE)** 2 * pxPlusy, axes=(0))

    # 15.
    def getMaximumPorbabiliy(self, comatrix):
        return np.apply_over_axes(np.max, comatrix, axes=(0, 1))[0, 0]



class GLRLM(object):
    '''
    Gray-Level Run-Length matrix based features
    '''
    def __init__(self):
        pass

    def getGLRLMFeatures(self, array):
        theta = ['deg0', 'deg45', 'deg90', 'deg135']
        rlmatrix = self.getGrayLevelRlmatrix(array, theta)

        sre = self.getShortRunEmphasis(rlmatrix)
        lre = self.getLongRunEmphasis(rlmatrix)
        gln = self.getGrayLevelNonUniformity(rlmatrix)
        rln = self.getRunLengthNonUniformity(rlmatrix)
        rp = self.getRunPercentage(rlmatrix)
        lglre = self.getLowGrayLevelRunEmphasis(rlmatrix)
        hglre = self.getHighGrayLevelRunEmphais(rlmatrix)
        srlgle = self.getShortRunLowGrayLevelEmphasis(rlmatrix)
        srhgle = self.getShortRunHighGrayLevelEmphasis(rlmatrix)
        # lrlgle = self.getLongRunLowGrayLevelEmphais(rlmatrix)
        # lrhgle = self.getLongRunHighGrayLevelEmphais(rlmatrix)

        return {'SRE_0': sre[0],   # SRE
                'SRE_45': sre[1],
                'SRE_90': sre[2],
                'SRE_135': sre[3],
                'LRE_0': lre[0],   # LRE
                'LRE_45': lre[1],
                'LRE_90': lre[2],
                'LRE_130': lre[3],
                'GLN_0': gln[0],   # GLN
                'GLN_45': gln[1],
                'GLN_90': gln[2],
                'GLN_135': gln[3],
                'RLN_0': rln[0],   # RLN
                'RLN_45': rln[1],
                'RLN_90': rln[2],
                'RLN_135': rln[3],
                'RP_0': rp[0],   # RP
                'RP_45': rp[1],
                'RP_90': rp[2],
                'RP_135': rp[3],
                'LGLRE_0': lglre[0],   # LGLRE
                'LGLRE_45': lglre[1],
                'LGLRE_90': lglre[2],
                'LGLRE_135': lglre[3],
                'HGLRE_0': hglre[0],   # HGLRE
                'HGLRE_45': hglre[1],
                'HGLRE_90': hglre[2],
                'HGLRE_135': hglre[3], 
                'SRLGLE_0': srlgle[0],   # SRLGLE
                'SRLGLE_45': srlgle[1],
                'SRLGLE_90': srlgle[2],
                'SRLGLE_135': srlgle[3],
                'SRHGLE_0': srhgle[0],   # SRHGLE
                'SRHGLE_45': srhgle[1],
                'SRHGLE_90': srhgle[2],
                'SRHGLE_135': srhgle[3],
                # 'LRLGLE_0': lrlgle[0],   # LRLGLE
                # 'LRLGLE_45': lrlgle[1],
                # 'LRLGLE_90': lrlgle[2],
                # 'LRLGLE_135': lrlgle[3],
                # 'LRHGLE_0': lrhgle[0],   # LRHGLE
                # 'LRHGLE_45': lrhgle[1],
                # 'LRHGLE_90': lrhgle[2],
                # 'LRHGLE_135': lrhgle[3]
                }

    def getGrayLevelRlmatrix(self, array, theta):
        '''
        计算给定图像的灰度游程矩阵
        参数：
        array: 输入，需要计算的图像
        theta: 输入，计算灰度游程矩阵时采用的角度，list类型，可包含字段:['deg0', 'deg45', 'deg90', 'deg135']
        glrlm: 输出，灰度游程矩阵的计算结果
        '''
        P = array
        x, y = P.shape
        min_pixels = np.min(P)   # 图像中最小的像素值
        run_length = max(x, y)   # 像素的最大游行长度
        num_level = np.max(P) - np.min(P) + 1   # 图像的灰度级数

        deg0 = [val.tolist() for sublist in np.vsplit(P, x) for val in sublist]   # 0度矩阵统计
        deg90 = [val.tolist() for sublist in np.split(np.transpose(P), y) for val in sublist]   # 90度矩阵统计
        diags = [P[::-1, :].diagonal(i) for i in range(-P.shape[0]+1, P.shape[1])]   #45度矩阵统计
        deg45 = [n.tolist() for n in diags]
        Pt = np.rot90(P, 3)   # 135度矩阵统计
        diags = [Pt[::-1, :].diagonal(i) for i in range(-Pt.shape[0]+1, Pt.shape[1])]
        deg135 = [n.tolist() for n in diags]

        def length(l):
            if hasattr(l, '__len__'):
                return np.size(l)
            else:
                i = 0
                for _ in l:
                    i += 1
                return i

        glrlm = np.zeros((num_level, run_length, len(theta)))   # 按照统计矩阵记录所有的数据， 第三维度表示计算角度
        for angle in theta:
            for splitvec in range(0, len(eval(angle))):
                flattened = eval(angle)[splitvec]
                answer = []
                for key, iter in groupby(flattened):   # 计算单个矩阵的像素统计信息
                    answer.append((key, length(iter)))   
                for ansIndex in range(0, len(answer)):
                    glrlm[int(answer[ansIndex][0]-min_pixels), int(answer[ansIndex][1]-1), theta.index(angle)] += 1   # 每次将统计像素值减去最小值就可以填入GLRLM矩阵中
        return glrlm


    def apply_over_degree(self, functions, x1, x2):
        rows, cols, nums = x1.shape
        result = np.ndarray((rows, cols, nums))
        for i in range(nums):
            # print(x1[:, :, i])
            result[:, :, i] = functions(x1[:, :, i], x2)
            # print(result[:, :, i])
        result[result == np.inf] = 0
        result[np.isnan(result)] = 0
        return result

    def calcuteIJ(self, rlmatrix):
        gray_level, run_length, _ = rlmatrix.shape
        I, J = np.ogrid[0:gray_level, 0:run_length]
        return I, J+1

    def calcuteS(self, rlmatrix):
        return np.apply_over_axes(np.sum, rlmatrix, axes=(0, 1))[0, 0]

        
    # 1. SRE
    def getShortRunEmphasis(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, rlmatrix, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
        
    # 2. LRE
    def getLongRunEmphasis(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.multiply, rlmatrix, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 3. GLN
    def getGrayLevelNonUniformity(self, rlmatrix):
        G = np.apply_over_axes(np.sum, rlmatrix, axes=1)
        numerator = np.apply_over_axes(np.sum, (G*G), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 4. RLN
    def getRunLengthNonUniformity(self, rlmatrix):
        R = np.apply_over_axes(np.sum, rlmatrix, axes=0)
        numerator = np.apply_over_axes(np.sum, (R*R), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 5. RP
    def getRunPercentage(self, rlmatrix):
        gray_level, run_length, _ = rlmatrix.shape
        num_voxels = gray_level * run_length
        return self.calcuteS(rlmatrix) / num_voxels

    # 6. LGLRE
    def getLowGrayLevelRunEmphasis(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, rlmatrix, (I*I)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 7. HGLRE
    def getHighGrayLevelRunEmphais(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.multiply, rlmatrix, (I*I)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 8. SRLGLE
    def getShortRunLowGrayLevelEmphasis(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 9. SRHGLE
    def getShortRunHighGrayLevelEmphasis(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        temp = self.apply_over_degree(np.multiply, rlmatrix, (I*I))
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 10. LRLGLE
    def getLongRunLowGrayLevelEmphais(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        temp = self.apply_over_degree(np.multiply, rlmatrix, (J*J), axes=(0, 1))
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 11. LRHGLE
    def getLongRunHighGrayLevelEmphais(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.multiply, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
    


class LBP(object):
    def __init__(self, P=1, R=1, method='default'):
        self.P = P
        self.R = R
        self.method = method

    def getLBPFeatures(self, array):
        P = self.P
        R = self.R
        method = self.method
        output = feature.local_binary_pattern(array, P, R, method=method)
        return output



class HOG(object):
    def __init__(self, orientation=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), feature_vector=True, boundary=False):
        self.boundary = boundary
        self.orientation = orientation
        self.feature_vector = feature_vector
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def getHOGFeatures(self, array, name='hog', flag=False):
        '''
            当flag=True时，直接采用了skimage库中的函数，采用分块的方法计算
            当flag=False时，采用自己实现的方法计算，不分块，直接计算
        '''
        output = None
        if flag:
            output = feature.hog(array, orientations=self.orientation, pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block, block_norm='L2', feature_vector=self.feature_vector)
        else:
            boundary = self.boundary
            result = self.getHOGMatrix(array, boundary, self.feature_vector)
            output = {name + '_0': result[0],
                      name + '_20': result[1],
                      name + '_40': result[2],
                      name + '_60': result[3],
                      name + '_80': result[4],
                      name + '_100': result[5],
                      name + '_120': result[6],
                      name + '_140': result[7],
                      name + '_160': result[8]}

        return output

    # 计算矩阵水平和垂直方向的梯度，在边界区域有两种计算方法：1.使用0填充；2.使用单个值计算梯度
    def getGradient(self, array, boundary):
        g_row = np.zeros(array.shape)
        g_row[0, :] = 0
        g_row[-1, :] = 0
        g_row[1:-1, :] = array[2:, :] - array[:-2, :]

        g_col = np.zeros(array.shape)
        g_col[:, 0] = 0
        g_col[:, -1] = 0
        g_col[:, 1:-1] = array[:, 2:] - array[:, :-2]

        if boundary:
            g_row[0, :] = array[1, :] - array[0, :]
            g_row[-1, :] = g_row[-1, :] - g_row[-2, :]

            g_col[:, 0] = array[:, 1] - array[:, 0]
            g_col[:, -1] = array[:, -1] - array[:, -2]

        return g_row, g_col

    def getHOGMatrix(self, array, boundary, feature_vector):
        gradient_x, gradient_y = self.getGradient(array, False)
        p_row, p_col = array.shape   # 像素
        orientation_histogram = np.zeros((1, self.orientation))

        magnitude = np.hypot(gradient_x, gradient_y)   # 计算欧几里得范数
        orientation = np.rad2deg(np.arctan(gradient_y, gradient_x)) % 180   # 计算反正切函数，将弧度转化为角度
        
        number_of_orientations_per_180 = 180. / self.orientation
        for i in range(self.orientation):
            orientation_start = number_of_orientations_per_180 * i
            orientation_end = number_of_orientations_per_180 * (i + 1)
            
            total = 0.
            for cell_row in range(0, p_row):
                cell_row_index = 0 + cell_row
                for cell_col in range(0, p_col):
                    cell_col_index = 0 + cell_col
                    if(orientation[cell_row_index, cell_col_index] >= orientation_start and orientation[cell_row_index, cell_col_index] < orientation_end):
                        total += magnitude[cell_row_index, cell_col_index]
            orientation_histogram[0, i] = total / (p_row * p_col)

        eps = 1e-5
        normalized = orientation_histogram / np.sqrt(np.sum(orientation_histogram ** 2) + eps **2)   # L2-norm
        if feature_vector:
            normalized = normalized.ravel()
        return normalized



if __name__ == '__main__':
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
    Img2 = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 2, 2],
                     [0, 0, 2, 2]])

    # glrlm = GLRLM()
    # result = glrlm.getGLRLMFeatures(Img.astype(np.int64))
    # print(result)
    start = time.clock()
    glrlm = GLCM()
    result = glrlm.getGLCMFeatures(Img.astype(np.int64))
    print(result)
    end = time.clock()
    print(end-start)

    # Img3 = 10 * np.random.randn(20, 20)
    
    # hog = HOGFeatures(feature_vector=False, pixels_per_cell=(1, 1), cells_per_block=(4, 4))
    # a1 = hog.getHOGFeatures(Img3, flag=True)
    # a1 = feature.hog(Img3, pixels_per_cell=(1, 1), cells_per_block=(20, 20), block_norm='L2', feature_vector=False)
    # a2 = hog.getHOGFeatures(Img3, flag=False)
    # lbp = LBPFeatures()
    # a3 = lbp.getLBPFeatures(Img2)
    # print(a1)
    # print('-----------------------')
    # print(a3)

    # glrlm = GLRLMFeatures(1)
    # result = glrlm.getGrayLevelRumatrix(Img2, ['deg45'])
    # print('\nresult:')
    # print(result[:, :, :])
    # print('---------------------')
    # # glrlm.getShortRunLowGrayLevelEmphasis(result)
    # data = glrlm.getShortRunHighGrayLevelEmphasis(result)
    # print('-----------------------')
    # print(data)
    # print(glrlm.getShortRunEmphasis(result))
    # print(glrlm.getGrayLevelNonUniformity(result))
    # temp = np.arange(4)
    # dat = np.true_divide(result[:, :, 0], (temp*temp))
    # print(dat)

    # Img3 = np.array([[2, 4, 6],
    #                  [4, 6, 8],
    #                  [6, 8, 10]])
    # print(np.divide(Img3, [1, 2, 3]))



    # image = np.array([[0, 0, 1, 1],
    #                   [0, 0, 1, 1],
    #                   [0, 2, 2, 2],
    #                   [2, 2, 3, 3]], dtype=np.uint8)
    # result = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=4)
    # print(result.shape)
    # print(result)
    # print('--------------------------------------')
    # (num_level, num_level2, num_dist, num_angle) = result.shape
    # # assert num_level == num_level2
    # # assert num_dist > 0
    # # assert num_angle > 0
    # I, J = np.ogrid[0:num_level, 0:num_level]
    # print(I, '\n')
    # print(J)
    # # print('--------------------------------------')
    # # print(I * J, '\n')
    # print(I - J )
    # print('--------------------------------------')
    # I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
    # J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
    # print(I, '\n')
    # print(J)
    # print('--------------------------------------')
    # diff_i = I - np.apply_over_axes(np.sum, (I * result), axes=(0, 1))[0, 0]
    # diff_j = J - np.apply_over_axes(np.sum, (J * result), axes=(0, 1))[0, 0]
    # print(diff_i, '\n')
    # print(diff_j)
    # print('--------------------------------------')
    # print(I)
    # print('>>>>>>>>>>>>>>>>>>')
    # print(result)
    # print('>>>>>>>>>>>>>>>>>>')
    # print(I * result)
    # print(np.apply_over_axes(np.sum, (I * result), axes=(0, 1)))
    # a = np.array([2, 4])
    # aa = np.array([[3],
    #                [2]])
    # print(a * aa)
    # print(-a * np.log2(a))
    