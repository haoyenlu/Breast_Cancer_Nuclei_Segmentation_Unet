
import torch

def confusion_matrix(predict, target):
    predict = predict.view(-1)
    target = target.view(-1)
    tp = (predict * target).sum().to(torch.float32)
    tn = ((1 - target) * (1 - predict)).sum().to(torch.float32)
    fp = ((1 - target) * predict).sum().to(torch.float32)
    fn = (target * (1 - predict)).sum().to(torch.float32)

    return tp, tn, fp, fn


def iou(tp, tn, fp, fn, epsilon=1e-3):
    return tp / (tp + fp + fn + epsilon)



def dice_score(tp, tn, fp, fn, epsilon=1e-3):
    return 2 * tp / (2 * tp + fp + fn + epsilon)



def f1_score(tp, tn, fp, fn, epsilon=1e-3):
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2* (precision*recall) / (precision + recall + epsilon)

    return f1


def TPR(tp, tn, fp, fn , epsilon=1e-3):
    return tp / (tp + fn + epsilon)

def FPR(tp, tn, fp, fn , epsilon=1e-3):
    return fp / (fp + tn + epsilon)

