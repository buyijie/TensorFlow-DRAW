import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

monitor_path=sys.argv[1]
print monitor_path

monitor=pickle.load(open(monitor_path, 'rb'))

monitor.pop('classification_precision', None)

for _k, _v in monitor.items():
    print _k
    print _v
    number_of_samples=len(_v)

x_axis=np.arange(number_of_samples)+1
colors=['b-', 'r-', 'g-', 'y-']
cnt=0

plt.figure(1)

for (_k, _v), _c in zip(monitor.items(), colors):
    cnt+=1
    plt.subplot(2, 2, cnt)
    plt.title(_k)
    plt.plot(x_axis, _v, _c, label=_k)
    if cnt>2:
        plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.axvline(x=20, color='k')

plt.show()
#plt.savefig('../img/loss.png')
