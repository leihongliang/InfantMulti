import json

import matplotlib

matplotlib.use('Agg')
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
import numpy as np

def get_data(run_id):
    """

    Args:
        run_id: run 的id

    Returns:
        test和train的纯数字数组

    """

    dirRoot = os.path.abspath("..")
    path = os.path.join(dirRoot, 'run/run_' + str(run_id))
    testPath = os.path.join(path + "/test.txt")
    trainPath = os.path.join(path + "/train.txt")
    testData = read_txt(testPath)
    trainData = read_txt(trainPath)
    return testData, trainData

def read_txt(file_name):
    """

    Args:
        file_name: 完整绝对路径

    Returns:
        读出txt中需要的数据，一个纯数字数组

    """
    data = []
    file = open(file_name, 'r')  # 打开文件
    file_data = file.readlines()  # 读取所有行
    for row in file_data:
        tmp_list = row.split(' ')  # 按‘，’切分每行的数据
        # Loss:  Acc:  CP:  CR:  CF1:
        # OP:  OR:  OF1:  MAP:
        data.append([tmp_list[4], tmp_list[6], tmp_list[8], tmp_list[10], tmp_list[12],
                     tmp_list[14], tmp_list[16], tmp_list[18], tmp_list[20]])
    data = np.array(data, dtype=float)
    return data

def show_chart(testData, trainData):
    testloss = list(np.array(testData[:, 0]).flatten())
    testF1 = list(np.array(testData[:, 1]).flatten())
    testCP = list(np.array(testData[:, 2]).flatten())
    testCR = list(np.array(testData[:, 3]).flatten())
    testCF1 = list(np.array(testData[:, 4]).flatten())
    testOP = list(np.array(testData[:, 5]).flatten())
    testOR = list(np.array(testData[:, 6]).flatten())
    testOF1 = list(np.array(testData[:, 7]).flatten())
    testMAP = list(np.array(testData[:, 8]).flatten())

    trainloss = list(np.array(trainData[:, 0]).flatten())
    trainF1 = list(np.array(trainData[:, 1]).flatten())
    trainCP = list(np.array(trainData[:, 2]).flatten())
    trainCR = list(np.array(trainData[:, 3]).flatten())
    trainCF1 = list(np.array(trainData[:, 4]).flatten())
    trainOP = list(np.array(trainData[:, 5]).flatten())
    trainOR = list(np.array(trainData[:, 6]).flatten())
    trainOF1 = list(np.array(trainData[:, 7]).flatten())
    trainMAP = list(np.array(trainData[:, 8]).flatten())

    x = range(len(testloss))
    plt.figure(figsize=(10, 6))


    plt.subplot(221, title='MAP')
    plt.plot(x, testMAP, 'red', linewidth=1, label='test map')
    plt.plot(x, trainMAP, 'green', linewidth=1, linestyle='--', label='train map')
    plt.legend()

    plt.subplot(222, title='CF1')
    plt.plot(x, testCF1, 'red', linewidth=1, label='test CF1')
    plt.plot(x, trainCF1, 'green', linewidth=1, linestyle='--', label='train CF1')
    plt.legend()

    plt.subplot(234, title='LOSS')
    plt.plot(x, testloss, 'red', linewidth=1, label='test loss')
    plt.plot(x, trainloss, 'green', linewidth=1, linestyle='--', label='train loss')
    plt.legend()

    plt.subplot(235, title='CR')
    plt.plot(x, testCF1, 'red', linewidth=1, label='test CR')
    plt.plot(x, trainCF1, 'green', linewidth=1, linestyle='--', label='train CR')
    plt.legend()

    plt.subplot(236, title='CP')
    plt.plot(x, testCF1, 'red', linewidth=1, label='test CP')
    plt.plot(x, trainCF1, 'green', linewidth=1, linestyle='--', label='train CP')
    plt.legend()

    plt.suptitle(run_id)
    svgsave = "E:/HDU/Project/InfantMulti/res_save/" + str(run_id) + ".svg"
    jpgsave = "E:/HDU/Project/InfantMulti/res_save/" + str(run_id) + ".jpg"
    plt.savefig(svgsave, dpi=300, format="svg")
    plt.savefig(jpgsave, dpi=300)
    plt.show()


def find_best(testData, run_id):
    """

    Args:
        testData: 完整的txt

    Returns:

    """
    bestMap = np.argmax(testData[:, 8])
    bestCF1 = np.argmax(testData[:, 4])
    bestCR = np.argmax(testData[:, 3])
    bestCP = np.argmax(testData[:, 2])
    testPath = os.path.join(os.path.join(os.path.abspath(".."), 'run/run_' + str(run_id)) + "/test.txt")
    file = open(testPath, 'r')  # 打开文件
    file_data = file.readlines()

    bestdict = {
        "bestMapEpoch": bestMap + 1,
        "bestCF1Epoch": bestCF1 + 1,
        "bestCREpoch": bestCR + 1,
        "bestCPEpoch": bestCP + 1,
        "bestMap": file_data[bestMap],
        "bestCF1": file_data[bestCF1],
        "bestCR": file_data[bestCR],
        "bestCP": file_data[bestCP]
    }
    return bestdict


if __name__ == '__main__':
    run_id = 60

    testData, trainData = get_data(run_id)
    bestdict = find_best(testData, run_id)
    print("bestMap: " + bestdict["bestMap"],
          "bestCF1: " + bestdict["bestCF1"],
          "bestCR: " + bestdict["bestCR"],
          "bestCP: " + bestdict["bestCP"],)
    show_chart(testData, trainData)
