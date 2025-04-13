import logging
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, confusion_matrix


def calculate_metrics(T, Y, S):
    AUC_dev = roc_auc_score(T, S)
    tpr, fpr, _ = precision_recall_curve(T, S)
    PRC_dev = auc(fpr, tpr)
    accuracy = accuracy_score(T, Y)
    tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc
