import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
args = parser.parse_args()
data_dir = args.data_dir

t_dir = './logs/' + data_dir + '/datatrain*.npy'
v_dir = './logs/' + data_dir + '/dataval*.npy'

t_loss = []
t_acc = []

v_loss = []
v_acc = []

for f in glob.glob(t_dir):
    stat = np.load(f)
    t_loss.append(stat[0])
    t_acc.append(stat[1])

for f in glob.glob(v_dir):
    stat = np.load(f)
    v_loss.append(stat[0])
    v_acc.append(stat[1])

plt.plot(t_acc)
plt.plot(v_acc)
plt.show()
