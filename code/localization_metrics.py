
def safe_roc_auc_score(y_true, y_pred):
    # AuC cannot be defined if only one class is present
    if len(set(y_true)) == 1:
        return None

    return roc_auc_score(y_true, y_pred)


def computeMetrics(values, gt, th):

    original_values = values.copy()
    values = values > th
    
    values = values.flatten().astype(np.uint8)
    gt = gt.flatten().astype(np.uint8)
    
    if np.max(values) == 0 and np.max(gt) == 0:
        return {
            "F1": 1,
            "Precision": 1,
            "Recall": 1,
            "Specificity": 1,
            "Accuracy": 1,
            "Balanced Accuracy": 1,
            "AuC": None,
            "IoU": 1,
            }  
    
    cm = confusion_matrix(gt, values, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    
    original_values = original_values.flatten()
    auc = safe_roc_auc_score(gt, original_values)
    
    f1 = 2 * TP / np.maximum((2 * TP + FN + FP), 1e-32)
    precision = TP/ (np.maximum(TP + FP, 1e-32))
    recall = TP / np.maximum(TP + FN, 1e-32)
    accuracy = (TP + TN) / np.maximum(TP + TN + FP + FN, 1e-32)
    specificity = TN/ np.maximum(TN + FP, 1e-32)
    balanced_accuracy = (recall + specificity) / 2
    iou = TP / np.maximum(TP + FP + FN, 1e-32)
        
    return {
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "AuC": auc,
        "IoU": iou,
    }  