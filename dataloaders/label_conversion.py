import numpy as np
def label_conversion(probs_01):
    prob = ','.join(str(i) for i in probs_01)
    prob = prob.replace(' ', ',')
    dic = {}
    keys = []
    fr = open('E:\HDU\Project\InfantMuti\dataloaders\infant_multi_labels_2.txt', 'r')
    for line in fr.readlines():
        k = line.split(' ')[0]
        v = line.split(' ')[1]
        dic[k] = v
    fr.close()
    return dic[prob]


def main():
    b=np.array([6.0253913e-11,1.1807488e-06,5.7720683e-08,9.7035384e-01,6.0490111e-06])
    # b=np.float(b)
    b=np.float32(b)
    print(b)


if __name__ == '__main__':
    main()