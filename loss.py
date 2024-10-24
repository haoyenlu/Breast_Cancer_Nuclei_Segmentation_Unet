import torch


def dice_loss(predict, target, smooth = 1.):

    dice_score = dice_coef(predict, target, smooth)

    return 1 - dice_score


def dice_coef(predict, target , smooth = 1. , batch=True):
    
    # Calculate dice per batch
    predict = predict.view(predict.shape[0],-1)
    target = target.view(target.shape[0],-1)


    intersection = torch.sum(predict * target,dim=1)
    union = torch.sum(predict.pow(2) ,dim=1) + torch.sum(target.pow(2),dim=1)

    score = (2 * intersection + smooth) / (union + smooth)
    score = torch.mean(score,dim=0)


    return  score
