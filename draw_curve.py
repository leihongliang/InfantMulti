import os
import re
import numpy as np
from matplotlib import pyplot as plt

tra_acc = np.loadtxt(os.path.abspath("."), 'run', 'run_' + str(1))
tra_acc = np.loadtxt(os.path.abspath("."), 'run', 'run_' + str(1))
tra_acc = np.loadtxt(os.path.abspath("."), 'run', 'run_' + str(1))
tra_acc = np.loadtxt(os.path.abspath("."), 'run', 'run_' + str(1))

x = list(range(200))
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_subplot(1, 1, 1)  # ax1是子图的名字

plt.ylim(0,1)
plt.plot(x, val_acc, 'r-', label=u'ResNet 3D')
plt.legend()

plt.xlabel(u'Epoch')
plt.ylabel(u'Acc')
plt.title('Acc')
plt.show()