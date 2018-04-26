import numpy as np
import matplotlib.pyplot as plt
import glob

model = '/transfer_real'
t_dir = './logs' + model + '/train*.npy'
v_dir = './logs' + model + '/val*.npy'

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

plt.plot(v_acc)
plt.show()