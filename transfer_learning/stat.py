from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

"""
BASELINE real training data, real validation
True negative:  443
False positive:  448
False negative:  135
True positive:  805

CCAN synthetic training, real validation
True negative:  0
False positive:  891
False negative:  0
True positive:  940

CCAN synthetic + real training, real validation
True negative:  536
False positive:  355
False negative:  204
True positive:  736

CCAN synthetic training, synthetic validation
True negative:  0
False positive:  940
False negative:  0
True positive:  940

CCAN real training, synthetic validation
True negative:  226
False positive:  714
False negative:  322
True positive:  618

CCAN synthetic + real training, synthetic validation
True negative:  861
False positive:  79
False negative:  565
True positive:  375

DCGAN synthetic training, real validation
True negative:  842
False positive:  49
False negative:  892
True positive:  48

DCGAN synthetic + real training, real validation
True negative:  685
False positive:  206
False negative:  287
True positive:  653
"""
truths = np.load('./predictions/transfer_synthetic/truths.npy')
preds = np.load('./predictions/transfer_synthetic/preds.npy')

precision, recall, _ = precision_recall_curve(truths, preds)
average_precision = average_precision_score(truths, preds)
tn, fp, fn, tp = confusion_matrix(truths, preds).ravel()
print("True negative: ", tn)
print("False positive: ", fp)
print("False negative: ", fn)
print("True positive: ", tp)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
plt.savefig('transfer_synthetic.jpg')