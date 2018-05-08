from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import average_precision_score, recall_score
"""
BASELINE real training data, real validation
True negative:  443
False positive:  448
False negative:  135
True positive:  805
Precision = 805/(805+448) = 0.642
Recall = 805/(805+135) = 0.856
Accuracy: 0.682

CCAN synthetic training, real validation
True negative:  0
False positive:  891
False negative:  0
True positive:  940
Precision = 940/(940+891) = 0.513
Recall = 940/940 = 1
Accuracy: 0.513

CCAN synthetic + real training, real validation
True negative:  536
False positive:  355
False negative:  204
True positive:  736
Precision = 736/(736 + 355) = 0.675
Recall = 736/(736 + 204) = 0.783
Accuracy: 0.695

DCGAN synthetic training, real validation
True negative:  842
False positive:  49
False negative:  892
True positive:  48
Precision = 48/(48+49) = 0.495
Recall = 48/(48+892) = 0.051
Accuracy: 0.471

DCGAN synthetic + real training, real validation
True negative:  685
False positive:  206
False negative:  287
True positive:  653
Precision = 653/(653+206) = 0.760
Recall = 653/(653+287) =0.695
Accuracy: 0.731
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
args = parser.parse_args()
data_dir = args.data_dir

truths = np.load('./predictions/' +data_dir +'/truths.npy')
preds = np.load('./predictions/' +data_dir +'/preds.npy')

precision, recall, _ = precision_recall_curve(truths, preds)
average_precision = average_precision_score(truths, preds)
average_recall = recall_score(truths, preds)
tn, fp, fn, tp = confusion_matrix(truths, preds).ravel()

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.3f}, AR={1:0.3f}'.format(
          average_precision, average_recall))
plt.savefig('plots/precision_recall/'+ data_dir + '.jpg')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(truths, preds)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['non-IDC', 'IDC'], normalize=True,
                      title='Normalized confusion matrix')

plt.savefig('./plots/confusion_matrix/'+data_dir+'.jpg')