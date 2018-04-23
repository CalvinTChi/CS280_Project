import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import time
import os
import copy

def de_normalize(image):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    for channel in range(3):
        image[channel] = image[channel] * std[channel] + mean[channel]
    return image
    
def visualize_model(model, dataset, num_images=6):
    was_training = model.training
    model.eval()
    fig = plt.figure()
    
    inputs, labels = dataset['test']
    inputs = inputs[:6]
    labels = labels[:6]
    inputs_og = inputs
    labels_og = labels
    
    inputs = torch.from_numpy(inputs).double()
    labels = torch.from_numpy(labels).double()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inputs_og = np.array([de_normalize(inp).transpose() for inp in inputs_og])
    inputs_og = np.clip(inputs_og, 0, 1)
    
    # wrap them in Variable
    if use_gpu:
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)
    
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
    
    images_so_far = 0
    class_names = {0: 'nc', 1:'c'}
    for i in range(num_images):
        images_so_far += 1
        ax = plt.subplot(num_images//2, 2, images_so_far)
        ax.axis('off')
        ax.set_title('predicted: {}, truth: {}'.format(class_names[preds[i]], class_names[labels_og[i]]))
        plt.imshow(inputs_og[i])
        
    plt.show()
    model.train(mode=was_training)
    
def train_model(model, dataset, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            inputs, labels = dataset[phase]
            
            inputs = torch.from_numpy(inputs).double()
            labels = torch.from_numpy(labels).double()
            
            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
    
X_train = np.load('./data/train/X_train.npy')
y_train = np.load('./data/train/y_train.npy')
X_test = np.load('./data/test/X_test.npy')
y_test = np.load('./data/test/y_test.npy')

dataset = {'train': (X_train, y_train), 'test': (X_test, y_test)}
    
model_ft = models.resnet152(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.double()
use_gpu = False

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# model_ft = train_model(model_ft, dataset, criterion, optimizer_ft, exp_lr_scheduler,
                       # num_epochs=25)
visualize_model(model_ft, dataset)

