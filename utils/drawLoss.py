import json

import matplotlib
import matplotlib.pyplot as plt
import os
import matplotlib
import numpy as np

matplotlib.use('Agg')


def show_chart(data, path):
    Loss = data[:, 0]
    F1 = data[:, 1]
    CP = data[:, 2]
    CR = data[:, 3]
    CF1 = data[:, 4]
    OP = data[:, 5]
    OR = data[:, 6]
    OF1 = data[:, 7]
    MAP = data[:, 8]

    x = range(0, 239)
    plt.figure()
    plt.plot(title='Loss', ylabel='loss')
    plt.plot(x, Loss)
    # plt.legend()
    # plt.subplot(311, title='Loss', ylabel='loss')
    # plt.plot(Loss)
    # plt.subplot(322, title='F1', ylabel='loss')
    # plt.plot(F1)
    # plt.subplot(323, title='MAP', ylabel='loss')
    # plt.plot(MAP)
    plt.show()
    # plt.savefig((path + data + '.png'))
    # plt.subplot(424, title='loss_bbox', ylabel='loss')
    # plt.plot(CR)
    # plt.subplot(425, title='total loss', ylabel='loss')
    # plt.plot(CF1)
    # plt.subplot(426, title='accuracy', ylabel='accuracy')
    # plt.plot(OP)
    # plt.subplot(427, title='accuracy', ylabel='accuracy')
    # plt.plot(OR)
    # plt.subplot(428, title='accuracy', ylabel='accuracy')
    # plt.plot(MAP)
    # print(sys.argv[1])
    # plt.suptitle((sys.argv[1][5:] + "\n training result"), fontsize=30)
    # # plt.savefig((sys.argv[1][5:] + '_result.png'))
    # plt.savefig(('output/' + sys.argv[1] + '_result.png'))


def openreadtxt(file_name):
    data = []
    file = open(file_name, 'r')  # 打开文件
    file_data = file.readlines()  # 读取所有行
    for row in file_data:
        tmp_list = row.split(' ')  # 按‘，’切分每行的数据
        a = tmp_list[4]
        data.append([tmp_list[4], tmp_list[6], tmp_list[8], tmp_list[10], tmp_list[12],
                     tmp_list[14], tmp_list[16], tmp_list[18], tmp_list[20]])
    data = np.array(data, dtype=float)
    return data


if __name__ == '__main__':
    run_id = 100
    dir_root = os.path.abspath("..")
    path = os.path.join(dir_root, 'run/run_' + str(run_id))
    train = os.path.join(path + "/train.txt")
    trainData = openreadtxt(train)
    Loss = trainData[:, 0]
    x = range(len(Loss))
    plt.plot(x, Loss)
    plt.legend()
    plt.show()
    # show_chart(trainData, path)

    # x = visualize_mmdetection(sys.argv[1])
    # x.load_data()
    # x.show_chart()
