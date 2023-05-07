from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, \
    roc_auc_score, precision_recall_curve, auc
import torch
import torch.nn.functional as func


def null_metrics():
    return {
        'acc': 0.0,
        'f1-score': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'mcc': 0.0,
        'roc-auc': 0.0,
        'pr-auc': 0.0
    }


# 输入：真实标签，预测数值（维度：用户数量 * 2）
# 输出：metrics:{七个指标}，plog:str(三个指标)
def calc_metrics(y, pred):
    assert y.dim() == 1 and pred.dim() == 2
    if torch.any(torch.isnan(pred)):    # 有空值返回0
        metrics = null_metrics()
        plog = ''
        for key, value in metrics.items():
            plog += ' {}: {:.6}'.format(key, value)
        return metrics, plog
    pred = func.softmax(pred, dim=-1)           # 对每一行求softmax
    pred_label = torch.argmax(pred, dim=-1)     # 返回每一行最大值的序号index，即 第0列->0:bot,第1列->1:human
    pred_score = pred[:, -1]                    # 取第二列
    y = y.to('cpu').numpy().tolist()
    pred_label = pred_label.to('cpu').tolist()
    pred_score = pred_score.to('cpu').tolist()
    precision, recall, _thresholds = precision_recall_curve(y, pred_score)
    metrics = {
        'acc': accuracy_score(y, pred_label),
        'f1-score': f1_score(y, pred_label),
        'precision': precision_score(y, pred_label),
        'recall': recall_score(y, pred_label),
        'mcc': matthews_corrcoef(y, pred_label),
        'roc-auc': roc_auc_score(y, pred_score),
        'pr-auc': auc(recall, precision)
    }
    plog = ''
    for key in ['acc', 'f1-score', 'mcc']:
        plog += ' {}: {:.6}'.format(key, metrics[key])
    return metrics, plog    # metrics:{七个指标}，plog:str(三个指标)


# 判断当前模型在训练集上的表现是否优于历史最佳
def is_better(now, pre):
    if now['acc'] != pre['acc']:
        return now['acc'] > pre['acc']
    if now['f1-score'] != pre['f1-score']:
        return now['f1-score'] > pre['f1-score']
    if now['mcc'] != pre['mcc']:
        return now['mcc'] > pre['mcc']
    if now['pr-auc'] != pre['pr-auc']:
        return now['pr-auc'] > pre['pr-auc']
    if now['roc-auc'] != pre['roc-auc']:
        return now['roc-auc'] > pre['roc-auc']
    if now['precision'] != pre['precision']:
        return now['precision'] > pre['precision']
    if now['recall'] != pre['recall']:
        return now['recall'] > pre['recall']
    return False
