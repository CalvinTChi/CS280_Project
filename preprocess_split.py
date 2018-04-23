import numpy as np
from sklearn.model_selection import train_test_split
from scipy.misc import imresize

def normalize(image):
    # Normalize to pretrained criteria
    shape = (224, 224, 3)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    image = imresize(image, shape)
    image = image/256.0
    for channel in range(3):
        image[:,:,channel] = (image[:,:,channel] - mean[channel]) / std[channel]
    return image
    
X = np.load('./data/X.npy')
y = np.load('./data/Y.npy')

X = np.array([normalize(image).transpose() for image in X])
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.save('./data/train/X_train.npy', X_train)
np.save('./data/train/y_train.npy', y_train)
np.save('./data/test/X_test.npy', X_test)
np.save('./data/test/y_test.npy', y_test)