import csv
import matplotlib as mpl
import matplotlib.pyplot as plt

VGG_FER = '../log/vgg_FER2013.csv'
VGG_CK = '../log/vgg_CK+.csv'
RESNET_FER = '../log/resnet_FER2013.csv'
RESNET_CK = '../log/resnet_CK+.csv'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
mpl.use('TkAgg')  # !IMPORTANT


def plotAcc(filename):
    epochs = []
    train_acc_list = []
    valid_acc_list = []
    with open(filename, 'r') as f:
        data = csv.reader(f)
        for line in data:
            if line[0] == 'epoch':
                continue
            if line[0] == '51':
                break
            epoch = int(line[0]) + 1
            train_acc = float(line[1][7:-2])
            valid_acc = float(line[2])
            epochs.append(epoch)
            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)

    plt.figure(figsize=(16, 10), dpi=50)
    plt.plot(epochs, train_acc_list, c='red', label='Training Accuracy')
    plt.plot(epochs, valid_acc_list, c='blue', label='Validation Accuracy')
    # plt.scatter(epochs, train_acc_list, c='red')
    # plt.scatter(epochs, valid_acc_list, c='blue')
    plt.rcParams.update({'font.size': 15})
    plt.legend(loc='upper left')

    plt.xticks(range(0, 51, 5), fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Train Epoch", fontdict={'size': 30})
    plt.ylabel("Accuracy", fontdict={'size': 30})
    plt.title("ResNet Accuracy Performance", fontdict={'size': 30})
    plt.show()


def main():
    plotAcc(RESNET_CK)


if __name__ == '__main__':
    main()
