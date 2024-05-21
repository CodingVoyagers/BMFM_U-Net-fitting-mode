
'''
    Manage path information for all .py files (including all paths under each python file, all controlled by one class).
'''
workSpace_dir = './DataSets/'

# seg_data_library
class seg_data_library_(object):
    def __init__(self, radius=13):
        grid = 2*radius
        root_dir = './DataSets/seg_model/'
        self.radius = radius
        self.grid = grid
        self.dcm_root = '../train/'
        self.raw_save_dir = './DataSets/Npy_Datas/'    # Conversion to npy's storage directory
        self.patch_train_path = root_dir + '/patchs/' + 'xgb_patch_train_' + str(grid) + '.npy'
        self.patch_label_path = root_dir + '/patchs/' + 'xgb_patch_label_' + str(grid) + '.npy'
        self.patch_scale_path = root_dir + '/ModelWeights/' + 'xgb_patch_scale_' + str(grid) + '.npy'    # Min-Max save path for scaling patches
        self.excel_raw_path = root_dir + '/features/' + 'xgb_excel_raw_' + str(grid) + '.xlsx'    # Data path to store original unscaled features


# seg_tumor
class seg_tumor_(object):
    def __init__(self, radius=13):
        grid = 2*radius
        root_dir = './gainFeatures/DataSets/seg_model/'
        self.patch_scale_path = root_dir + '/ModelWeights/' + 'xgb_patch_scale_' + str(grid) + '.npy'    # Min-Max save path for scaling patches
        self.excel_scale_path = root_dir + '/ModelWeights/' + 'xgb_excel_scale_' + str(grid) + '.csv'    # Min-Max save path for scaling patches
        self.seg_model_path = './model/GBDT.pkl'

        self.radius = radius
        self.grid = grid
        self.dcm_root = '../../dfy-2021-four_slices/train/'
        self.raw_save_dir = './gainFeatures/DataSets/Npy_Datas/'  # Conversion to npy's storage directory
        self.patch_train_path = root_dir + '/patchs/' + 'xgb_patch_train_' + str(grid) + '.npy'
        self.patch_label_path = root_dir + '/patchs/' + 'xgb_patch_label_' + str(grid) + '.npy'
        self.excel_raw_path = root_dir + '/features/' + 'xgb_excel_raw_' + str(grid) + '.xlsx'  # Data path to store original unscaled features
