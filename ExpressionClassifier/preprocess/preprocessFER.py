import csv
import os
import numpy as np
import h5py
from tqdm import tqdm, trange

fer_data_file = '../data/FER2013/fer2013.csv'
data_path = '../data/FER2013/data.h5'


def main():
    print("Starting Preprocessing FER2013's dataset")
    if not os.path.exists(os.path.dirname(data_path)):
        os.makedirs(os.path.dirname(data_path))

    train_x = []
    train_label = []
    private_test_x = []
    private_test_label = []
    public_test_x = []
    public_test_label = []

    with open(fer_data_file, 'r') as f:
        data = csv.reader(f)
        for row in tqdm(data):
            if row[-1] == 'Training':
                pixel_list = []
                for pixel in row[1].split():
                    pixel_list.append(int(pixel))
                train_label.append(int(row[0]))
                train_x.append(np.asarray(pixel_list).tolist())
            if row[-1] == 'PrivateTest':
                pixel_list = []
                for pixel in row[1].split():
                    pixel_list.append(int(pixel))
                private_test_label.append(int(row[0]))
                private_test_x.append(np.asarray(pixel_list).tolist())
            if row[-1] == 'PublicTest':
                pixel_list = []
                for pixel in row[1].split():
                    pixel_list.append(int(pixel))
                public_test_label.append(int(row[0]))
                public_test_x.append(np.asarray(pixel_list).tolist())

    print("Training: ", np.shape(train_x))
    print("Public Test: ", np.shape(public_test_x))
    print("Private Test: ", np.shape(private_test_x))

    datafile = h5py.File(data_path, 'w')
    datafile.create_dataset("Training_features", dtype='uint8', data=train_x)
    datafile.create_dataset("Training_labels", dtype='int64', data=train_label)
    datafile.create_dataset("PublicTest_features", dtype='uint8', data=public_test_x)
    datafile.create_dataset('PublicTest_labels', dtype='int64', data=public_test_label)
    datafile.create_dataset('PrivateTest_features', dtype='uint8', data=private_test_x)
    datafile.create_dataset('PrivateTest_labels', dtype='int64', data=private_test_label)

    datafile.close()

    print("Finished preprocessing FER2013's dataset")


if __name__ == '__main__':
    main()
