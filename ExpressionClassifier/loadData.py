import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image


class FERDataSet(Dataset):
    def __init__(self, mode='Training', transform=None):
        self.mode = mode
        self.transform = transform
        self.datas = h5py.File('./data/FER2013/data.h5', 'r', driver='core')
        if self.mode == 'Training':
            self.features = self.datas['Training_features']
            self.labels = self.datas['Training_labels']
        if self.mode == 'PublicTest':
            self.features = self.datas['PublicTest_features']
            self.labels = self.datas['PublicTest_labels']
        if self.mode == 'PrivateTest':
            self.features = self.datas['PrivateTest_features']
            self.labels = self.datas['PrivateTest_labels']
        num = len(self.features)
        self.features = np.asarray(self.features).reshape((num, 48, 48))

    def __getitem__(self, index):
        feature, label = self.features[index], self.labels[index]
        # 转成 PIL Image
        img = feature[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label)

    def __len__(self):
        return len(self.features)


class CKDataSet(Dataset):
    """
        there are 135,177,75,207,84,249,54 images in data
        we choose 123,159,66,186,75,225,48 images for training
        we choose 12,8,9,21,9,24,6 images for testing
    """
    def __init__(self, mode='Training', transform=None):
        self.mode = mode
        self.transform = transform
        self.data = h5py.File('./data/CK+/CK+_data.h5', 'r', driver='core')

        nums = len(self.data['data_label'])  # 共 981 个
        sum_number = [0, 135, 312, 387, 594, 678, 927, 981]  # the sum of class number
        val_number = [12, 15, 7, 18, 8, 23, 5]
        test_number = [12, 18, 9, 21, 9, 24, 6]  # the number of each class

        test_index = []
        val_index = []
        train_index = []

        for j in range(len(test_number)):
            for k in range(test_number[j]):
                test_index.append(sum_number[j + 1] - 1 - k)

        for j in range(len(val_number)):
            for k in range(val_number[j]):
                val_index.append(sum_number[j + 1] - 1 - test_number[j] - k)

        for i in range(nums):
            if i not in test_index:
                train_index.append(i)

        self.features = []
        self.labels = []
        if self.mode == 'Training':
            for idx in range(len(train_index)):
                self.features.append(self.data['data_pixel'][train_index[idx]])
                self.labels.append(self.data['data_label'][train_index[idx]])
        elif self.mode == 'Validating':
            for idx in range(len(val_index)):
                self.features.append(self.data['data_pixel'][val_index[idx]])
                self.labels.append(self.data['data_label'][val_index[idx]])
        elif self.mode == 'Testing':
            for idx in range(len(test_index)):
                self.features.append(self.data['data_pixel'][test_index[idx]])
                self.labels.append(self.data['data_label'][test_index[idx]])

    def __getitem__(self, index):
        feature, label = self.features[index], self.labels[index]
        # 转成 PIL Image
        img = feature[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label)

    def __len__(self):
        return len(self.features)
