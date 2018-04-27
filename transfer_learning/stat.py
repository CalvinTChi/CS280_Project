from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

truths = np.load('./predictions/transfer_combined/truths.npy')
preds = np.load('./predictions/transfer_combined/preds.npy')

precision, recall, _ = precision_recall_curve(truths, preds)
average_precision = average_precision_score(truths, preds)

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
plt.savefig('transfer_combined_pr.jpg')