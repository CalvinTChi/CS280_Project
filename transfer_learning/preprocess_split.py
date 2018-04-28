import numpy as np
from sklearn.model_selection import train_test_split
from scipy.misc import imresize, imsave
    
X = np.load('./data/X.npy')
y = np.load('./data/Y.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

count = 0
for x, y in zip(X_train, y_train):
    if y == 0:
        imsave('./data/train/not/' + str(count) + '.jpg', x)
    else:
        imsave('./data/train/cancer/' + str(count) + '.jpg', x)
    count += 1

count = 0
for x, y in zip(X_test, y_test):
    if y == 0:
        imsave('./data/test/not/' + str(count) + '.jpg', x)
    else:
        imsave('./data/test/cancer/' + str(count) + '.jpg', x)
    count += 1