from utils import metrics
import csv
import os
from pathlib import Path
'''
log file list :
1. weight path - weight  .pth file
2. confusion matrtix log
3. metrics log
4. test result log
'''
def eval_log(log_path, result_log):
    with open(os.path.join(log_path, 'evaluation.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for id, GT, ind, prob in result_log:
            for i in range(len(id)):
                writer.writerow([int(id[i]), GT[i], ind[i], prob[i]])

def pred_log(log_path, result_log, count):
    with open(os.path.join(log_path, 'prediction_detail.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for id, ind, prob in result_log:
            writer.writerow([id ,ind[0], prob[0]])
    with open(os.path.join(log_path, 'prediction_count.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i, num in enumerate(count):
            writer.writerow(["class:"+str(i), str(num)])

def metrics_log(log_path, metrics, conf_mat):
    acc, precision, recall, specificity, F1_score = metrics
    # score log
    with open(os.path.join(log_path, 'metrics.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['Accuracy', '{acc:.2f}'.format(acc=acc*100)])
        writer.writerow(['Precision', '{prec:.2f}'.format(prec=precision*100)])
        writer.writerow(['Recall', '{recall:.2f}'.format(recall=recall*100)])
        writer.writerow(['Specificity', '{spec:.2f}'.format(spec=specificity*100)])
        writer.writerow(['F1 Score', '{f1:.2f}'.format(f1=F1_score*100)])
    
    # matrix log
    with open(os.path.join(log_path, 'conf_mat.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(conf_mat)

    
    
