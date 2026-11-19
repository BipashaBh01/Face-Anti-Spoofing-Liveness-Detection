# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
from Functions.config import *



#=========================================================
# MAIN FUNCTION FOR PAD METRICS
#=========================================================

def compute_pad_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Compute Presentation Attack Detection (PAD) evaluation metrics.

    Args:
        y_true (array-like):
            Ground truth binary labels (0 = real, 1 = fake).
        y_pred_probs (array-like):
            Predicted probabilities for the positive class (attack).
        threshold (float, optional):
            Threshold to convert probabilities into binary predictions. Default = 0.5.

    Returns:
        dict:
            Dictionary containing:
                - APCER: Attack Presentation Classification Error Rate (FP rate)
                - BPCER: Bona Fide Presentation Classification Error Rate (FN rate)
                - ACER : Average Classification Error Rate
                - Acc  : Accuracy
                - AUC  : Area Under ROC Curve
    """
    # Convert probabilities into binary predictions using threshold (>= 0.5 --> 1 (fake))
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Compute confusion matrix: [TN, FP, FN, TP]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    # APCER: proportion of attack samples misclassified as genuine (False Positives)
    # 1e-8 avoids division by zero
    apcer = fn / (fn + tp + 1e-8)    # attack (fake) samples passed as real / total fake

    # BPCER: proportion of genuine samples misclassified as attack (False Negatives)
    bpcer = fp / (fp + tn + 1e-8)    # bona-fide (real) samples rejected as fake / total real

    # ACER: average of APCER and BPCER (overall fairness metric)
    acer  = (apcer + bpcer) / 2

    # Accuracy: overall correct predictions
    acc   = (tp + tn) / (tp + tn + fp + fn)

    # AUC: measures ranking quality of predicted probabilities
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except:
        # If only one class exists in batch --> fallback to 0.5 (random model)
        auc = 0.5

    return {'APCER': apcer, 'BPCER': bpcer, 'ACER': acer, 'Acc': acc, 'AUC': auc}       # Returns all metrics in a dictionary


#=========================================================
# FUNCTION FOR TDR@FDR=1%
#=========================================================

def compute_tdr_at_fdr(y_true, y_probs, target_fdr=0.01):
    """
    TDR @ FDR — spoof catch-rate at a fixed false-alarm budget.

    Answers: "If only `target_fdr` of REAL faces is allowed to be wrongly
    flagged as spoof, what fraction of SPOOFS is still catched?"
    Higher is better. This is the paper's headline metric.

    Args:
        y_true  : ground-truth labels (0 = real, 1 = spoof)
        y_probs : predicted spoof probability per sample (0..1)
        target_fdr : allowed false-detection rate on real faces (default 1%)

    Returns:
        TDR as a percentage (0..100).
    """
    # ROC sweep: with spoof=positive, fpr = real-flagged-as-spoof (= FDR),
    # tpr = spoofs-caught (= TDR). Each index is one threshold/operating point.
    fpr, tpr, _ = roc_curve(y_true, y_probs)

    # Keep only the operating points that meet the false-alarm budget (FDR <= target).
    within_budget = np.where(fpr <= target_fdr)[0]

    # If no threshold is strict enough to hit the target FDR, nothing is caught.
    if len(within_budget) == 0:
        return 0.0

    # Among qualifying points, take the most lenient (largest FDR still <= target),
    # which catches the most spoofs -> read its TDR.
    return tpr[within_budget[-1]] * 100.0