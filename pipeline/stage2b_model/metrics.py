"""Stage 2b evaluation metrics — copied from pipeline/scripts/train_stage2.py.

Source lines: 588-647 (confusion_matrix, macro_f1_from_cm, qwk_from_cm, ece_score)
DO NOT modify without also updating train_stage2.py.
"""

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def macro_f1_from_cm(cm):
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        den = (2 * tp + fp + fn)
        f1 = (2 * tp / den) if den > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), [float(x) for x in f1s]


def qwk_from_cm(cm):
    n = cm.sum()
    if n == 0:
        return 0.0
    k = cm.shape[0]
    w = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            w[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)
    act = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    e = np.outer(act, pred) / max(1, n)
    num = (w * cm).sum()
    den = (w * e).sum()
    if den <= 0:
        return 0.0
    return float(1.0 - num / den)


def ece_score(probs, y_true, n_bins=15):
    # probs: [N,K], y_true: [N]
    if len(y_true) == 0:
        return 0.0
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        acc_bin = acc[m].mean()
        conf_bin = conf[m].mean()
        ece += float(m.sum()) / n * abs(float(acc_bin) - float(conf_bin))
    return float(ece)
