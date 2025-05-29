from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,  # Added import
)

def calculate_metrics(y_true, y_pred, probs):
    # (n,), (n,), (n,)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, probs)  # Added AUC calculation
    return accuracy, recall, precision, f1, mcc, auc  # Added auc to return
