import numpy as np

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)  # 如果是2类的话 2*2矩阵 [0预测对的数量 0预测成1的数量]\[1预测成0的数量 1预测对的数量]
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()  # 预测对的占总数
    acc_cls = np.diag(hist) / hist.sum(axis=1) # 各类预测的正确率
    acc_cls = np.nanmean(acc_cls)  # 算术平均值
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))  # 即IoU交并比
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()  # 即frequency weighted IU，频权交并比
    return acc, acc_cls, mean_iu, fwavacc
