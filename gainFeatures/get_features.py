
'''
    Generate training data for annotated models from raw data
    1. There are three main categories: ___, ___ and ___.
    2. As the tumour regions of some categories are very small, it is estimated that the sampling conditions will be appropriately scaled so that as long as 80% of the data in the region belong to that type, it can be used as a training set

    Steps:
    Raw nii.gz ==> slices


    Scaling is used when getting patient patches 768/ (use '. /DataSets/seg_model/xgb_all_patch_scale.npy' ==> [ [0,255] [0,255] [0,255] [0,255] [0,255] ] )
    1. generate a raw matrix of (155, 5, 240, 240) based on the four data sequences for each patient
    2. extract three categories of patches (4, grid, grid) from each raw matrix and obtain the corresponding labels.

    When extracting features from the patches, we also need to deflate the patches, not the extracted features'. /DataSets/seg_model/ModelWeights/xgb_patch_scale_' + str(grid) + '.npy' 3.
    3. Based on the obtained patches, extract the features of each of the four data series and obtain the labels to be stored as excel data.
'''
import os
import random
import numpy as np
import pandas as pd
from glob import glob
from skimage import io
from FeatureExtraction import FirstOrderStatistics, TexturalFeatures
from file_path import seg_data_library_

radius = 5
sdl_ = seg_data_library_(radius)
dcm_root = sdl_.dcm_root #'E:/unet-Data/2020/HGG/'
raw_save_dir = sdl_.raw_save_dir #'./DataSets/Npy_Datas/'


class SegDataLibrary(object):
    def __init__(self, train_dir='', label_dir=''):
        self.total_patchs = 50000  # 2400
        self.label_lengths = [[], [1000, 500, 100], [1000, 500, 100], [], [500, 200, 80]]

        self.fos = FirstOrderStatistics.FirstOrderStatistics()
        self.glcm = TexturalFeatures.GLCM()
        self.glrlm = TexturalFeatures.GLRLM()
        self.hog = TexturalFeatures.HOG()
        self.lbp = TexturalFeatures.LBP()

        sdl_ = seg_data_library_(radius)
        self.raw_save_dir = sdl_.raw_save_dir
        self.radius = sdl_.radius
        self.patch_train_path = sdl_.patch_train_path
        self.patch_label_path = sdl_.patch_label_path
        self.patch_scale_path = sdl_.patch_scale_path  # #Min-Max save path for scaling patches
        self.excel_raw_path = sdl_.excel_raw_path
        self.all_patch_scale_path = './DataSets/seg_model/xgb_all_patch_scale.npy'
        self.all_np_scale_model = np.load(self.all_patch_scale_path) # [ [0,255] [0,255] [0,255] [0,255] ]


    # Process the raw dcm data into an npy matrix and save it (for ease of processing and to keep the data intact)
    def disposeRawDatas(self, dcm_dir, save_dir):
        print('Loading scans...')

        slices_by_mode = np.zeros((5, 155, 240, 240))
        slices_by_slices = np.zeros((155, 5, 240, 240))

        flair = glob(dcm_dir + '/*flair.nii.gz')
        t1 = glob(dcm_dir + '/*t1.nii.gz')
        t1_ce = glob(dcm_dir + '/*t1ce.nii.gz')
        t2 = glob(dcm_dir + '/*t2.nii.gz')
        seg = glob(dcm_dir + '/*seg.nii.gz')
        scans = [flair[0], t1[0], t1_ce[0], t2[0], seg[0]]
        patient_name = scans[0][scans[0].rfind("\\") + 1: scans[0].rfind('.')].replace("_flair", "")

        for scan_idx in range(5):  # Convert data formats
            slices_by_mode[scan_idx] = io.imread(scans[scan_idx], plugin='simpleitk').astype(float)
        for mode_idx in range(slices_by_mode.shape[0]):
            for slice_idx in range(slices_by_mode.shape[1]):
                slices_by_slices[slice_idx][mode_idx] = slices_by_mode[mode_idx][slice_idx]  #  The reshape data is the input to the CNN
        # print(slices_by_slices.shape)
        for s_idx in range(155):  #  Preservation according to conditions
            strips = slices_by_slices[s_idx]
            tmp = strips[-1]  ##segment img
            result = np.argwhere(tmp != 0)  # Returns a result not equal to 0 if = 0 then result is []
            if len(result) > 600:
                np.save(save_dir + '/{}_{}.npy'.format(patient_name, s_idx), strips)
        return slices_by_slices, patient_name

    # Scaling models Scaling is used when getting patient patches
    def scaleSlices(self, trains):
        np_scale = self.all_np_scale_model  # gb_all_patch_scale.npy' [ [0,255] [0,255] [0,255] [0,255] ]
        for i in range(4):
            min_val, max_val = np_scale[i]
            if min_val != max_val:
                trains[i, :, :] -= min_val
                trains[i, :, :] *= (768.0 / (max_val - min_val))
        return trains

    # 获Get the boundaries of the different categories (label={1, 2, 3}) and return the
    def getBoundaryCoordinates(self, array, label):
        array = array[0]
        array[array != label] = 0
        cnt = np.argwhere(array == label)
        return np.array(cnt) if np.any(array == label) else np.array([])

        # Get patches for different categories for each patient

    def getPatchsByLabel(self, images_t, images_l, radius, sentinel):
        train, label = [], []
        img_train = images_t[:4].reshape(4, 240, 240)
        img_train = self.scaleSlices(img_train)  # Scaling Models
        img_label = images_l.reshape(1, 240, 240)

        grid = 2 * radius

        cnt = self.getBoundaryCoordinates(img_label, sentinel)
        length = len(cnt)
        # [[], [1000, 500, 100], [1000, 500, 100], [], [500, 200, 80]]
        if length > self.label_lengths[sentinel][0]:
            num, search_times = 10, 300
        elif length > self.label_lengths[sentinel][1]:
            num, search_times = 5, 200
        elif length > self.label_lengths[sentinel][2]:
            num, search_times = 1, 100
        else:
            num, search_times = 0, 0

        for i in range(num):
            flag = True
            times = 1
            while (flag and times <= search_times):
                idx = random.randint(0, len(cnt) - 1)  # Horizontal coordinates of a random point
                x, y = cnt[idx]
                # radius 5
                l_x, l_y = x - radius, y - radius
                r_x, r_y = x + radius, y + radius
                tmp_train = img_train[:, l_y:r_y, l_x:r_x]
                tmp_label = img_label[0, l_y:r_y, l_x:r_x]
                numbers = np.argwhere(tmp_label == sentinel)
                if (((float(len(numbers) / (grid * grid))) >= 0.8) and (
                        tmp_train.shape[1] == tmp_train.shape[2] == grid) and np.all(tmp_train) >= 0):
                    train.append(tmp_train)
                    label.append([sentinel])
                    flag = False
                times += 1

        return np.array(train), np.array(label)

    # Get all patient patches (contains three types)
    def getDiseasedPatchs(self, radius, save_patch=False):
        trains, labels = [], []
        train_dir = './DataSets/Npy_Datas/'
        paths = [train_dir + '/' + s for s in os.listdir(train_dir)]
        random.shuffle(paths)
        sentinels = [1, 2, 4]
        for s in sentinels:  # Every label
            numbers = 0
            print("lable %s :"%s)
            for id, p in enumerate(paths): #Each 5-dimensional slice
                print("%s/%s (label %s)" % (id + 1, len(paths),s))
                images_t = np.load(p).reshape(5, 240, 240).astype('float')
                images_l = images_t[4, :, :]
                train, label = self.getPatchsByLabel(images_t, images_l, radius, sentinel=s)
                trains.extend(train)
                labels.extend(label)
                numbers += len(train)
                # print(s)
                if (numbers > self.total_patchs):  # Maximum 10,000 per label 1 2 4
                    break

            print('The num of label {} is: {}'.format(s, numbers))

        if save_patch == False:
            return trains, labels
        else:
            patch_train_path = self.patch_train_path
            patch_label_path = self.patch_label_path
            if not os.path.exists(patch_train_path):
                os.makedirs(patch_train_path)
            if not os.path.exists(patch_label_path):
                os.makedirs(patch_label_path)
            np.save(patch_train_path + 'seg_train_' + str(2 * radius) + '.npy', trains)
            np.save(patch_label_path + 'seg_label_' + str(2 * radius) + '.npy', labels)
            train_save_path = patch_train_path + 'seg_train_' + str(2 * radius) + '.npy'
            label_save_path = patch_label_path + 'seg_label_' + str(2 * radius) + '.npy'
            return train_save_path, label_save_path

    # Criteria for obtaining a normal patch
    def normPatch(self, images_t, images_l, radius, num):
        train, label = [], []
        img_train = images_t[:4].reshape(4, 240, 240)
        img_train = self.scaleSlices(img_train)  #  Scaling Models
        img_label = images_l.reshape(1, 240, 240)
        grid = 2 * radius
        cnt = np.argwhere(img_label[0] == 0)
        for i in range(num):
            flag = True
            times = 1
            while (flag and times != 200):
                idx = random.randint(0, len(cnt) - 1)
                x, y = cnt[idx]
                l_x, l_y = x - radius, y - radius  # Get the coordinates of the upper-left and lower-right corners of the small squares
                r_x, r_y = x + radius, y + radius
                tmp_train = img_train[:, l_y:r_y, l_x:r_x]
                tmp_label = img_label[0, l_y:r_y, l_x:r_x]
                numbers = np.argwhere(tmp_label != 0)  # The number of zeros that are not equal to zero is less than 0.01,
                                                       # which means that 99.99 of this block is normal tissue before it's zero.
                if (((float(len(numbers) / (grid * grid))) <= 0.01) and (
                        tmp_train.shape[1] == tmp_train.shape[2] == grid) and np.all(tmp_train) > 0):
                    # array[:, :, :] == tmp_train[:, :, :]
                    train.append(tmp_train)
                    label.append([0])
                    flag = False
                times += 1

        return np.array(train), np.array(label)

    # Getting a normal patch
    def getNormalPatchs(self,radius, num=1, save_patch=False):
        trains, labels = [], []
        train_dir = './DataSets/Npy_Datas/'
        paths = [train_dir + '/' + s for s in os.listdir(train_dir)]
        ####################
        random.shuffle(paths)
        numbers = 0
        print("label 0 :" )
        for id,p in enumerate(paths):
            print("%s/%s (label %s)" % (id + 1, len(paths), 0))
            images_t = np.load(p).reshape(5, 240, 240).astype('float')
            images_l = images_t[4, :, :]
            train, label = self.normPatch(images_t, images_l, radius, num=num)
            trains.extend(train)
            labels.extend(label)
            numbers += len(train)
            if numbers > self.total_patchs:
                break
        print('The num of label {} is: {}'.format(0, numbers))
        if save_patch == False:
            return trains, labels
        else:
            patch_train_path = self.patch_train_path
            patch_label_path = self.patch_label_path
            if not os.path.exists(patch_train_path):
                os.makedirs(patch_train_path)
            if not os.path.exists(patch_label_path):
                os.makedirs(patch_label_path)
            np.save(patch_train_path + 'seg_train_' + str(2 * radius) + '.npy', trains)
            np.save(patch_label_path + 'seg_label_' + str(2 * radius) + '.npy', labels)

    # Extraction of features (for flair, t1, t1_ce, t2 respectively)
    def getFeatures(self, arrays, idx=4):
        all_features = {}


        shuffix = ['f', 't1', 't1c', 't2']
        for i in range(idx):
            array = arrays[i]
            fos_f = self.fos.getFirstOrderStatistics(array)
            glcm_f = self.glcm.getGLCMFeatures(array.astype(np.int64))
            glrlm_f = self.glrlm.getGLRLMFeatures(array.astype(np.int64))
            hog_f = self.hog.getHOGFeatures(array)
            lbp_f = self.hog.getHOGFeatures(self.lbp.getLBPFeatures(array), name='lbp')
            features = {**fos_f, **glcm_f, **glrlm_f, **hog_f, **lbp_f}
            tmp_features = {}
            for key in features:
                new_key = key + '_' + shuffix[i]
                tmp_features[new_key] = features[key]
            all_features = {**all_features, **tmp_features}
        return {**all_features}

    # Extract features based on the obtained patches
    def getFeaturesByPatchs(self, train_save, label_save):
        print('Starts get features...')
        patch_scale_path = self.patch_scale_path  # './DataSets/seg_model/ModelWeights/xgb_patch_scale_' + str(grid) + '.npy'
                                                  # 缩Min-Max save path for scaling patches
        excel_raw_path = self.excel_raw_path  # './DataSets/seg_model/features/' + 'xgb_excel_raw_' + str(grid) + '.xlsx'
        if (type(train_save) == str):
            trains = np.load(train_save)
            labels = np.load(label_save)
        else:
            trains = train_save
            labels = label_save
        print(trains.shape)
        print(labels.shape)

        scale_model = []
        for i in range(4):
            min_val = trains[:, i, :, :].min()
            max_val = trains[:, i, :, :].max()
            scale_model.append([min_val, max_val])
            if min_val != max_val:
                trains[:, i, :, :] -= min_val
                trains[:, i, :, :] *= (1.0 / (max_val - min_val))
        scale_model = np.array(scale_model)
        print(scale_model)
        np.save(patch_scale_path, scale_model)

        data_frame = None
        for i in range(len(trains)):
            print('The number {}/{}'.format(i, len(trains)))
            train = trains[i]
            label = labels[i]
            features = self.getFeatures(train, len(train))
            features['zLabel'] = label[0]
            if data_frame is None:
                data_frame = pd.DataFrame(features, index=[0])
            else:
                data_frame.loc[i] = features
        print(len(data_frame.columns), '\n')
        writer = pd.ExcelWriter(excel_raw_path)
        writer.columns = features
        data_frame.to_excel(writer, index=False, encoding='utf-8')
        writer.save()
        print(data_frame.tail())
        print('Save to -->{}<-- successed!\n'.format(excel_raw_path))


# ----------------------------------------------------------------------------------------


#  Processing of raw data
def disposeRawdatas():
    print('Start dispose raw datas...')

    dcm_dir = [dcm_root + '/' + s for s in os.listdir(dcm_root)] ##'E:/unet-Data/2020/HGG/'
    random.shuffle(dcm_dir)

    for d in dcm_dir:
        print(d)
        seg_data_library = SegDataLibrary()
        seg_data_library.disposeRawDatas(d, raw_save_dir)

    print('Dispose raw datas finished!!!')




# Extract the patches and get the corresponding features
def getFeaturesFromPatchs():
    radius = 5
    trains, labels = [], []
    seg_data_library = SegDataLibrary()
    patch_train_path = seg_data_library.patch_train_path  # './DataSets/seg_model/patchs/xgb_patch_train_' + str(grid) + '.npy'
    patch_label_path = seg_data_library.patch_label_path  # './DataSets/seg_model/patchs/xgb_patch_label_' + str(grid) + '.npy'
    train, label = seg_data_library.getDiseasedPatchs( radius=radius)  # Getting 1 2 4 patch paths that produce lesions is useless
    trains.extend(train)
    labels.extend(label)
    train, label = seg_data_library.getNormalPatchs(radius=radius)  # Getting normal brain tissue for patch path is useless
    trains.extend(train)
    labels.extend(label)
    trains = np.array(trains)
    labels = np.array(labels)

    np.save(patch_train_path, trains)  # './DataSets/seg_model/patchs/xgb_patch_train_' + str(grid) + '.npy'
    np.save(patch_label_path, labels)  # './DataSets/seg_model/patchs/xgb_patch_label_' + str(grid) + '.npy'


    seg_data_library.getFeaturesByPatchs(train_save=patch_train_path, label_save=patch_label_path)


if __name__ == '__main__':
    disposeRawdatas()
    getFeaturesFromPatchs()
