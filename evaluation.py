import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt



from metric import iou, confusion_matrix, f1_score, TPR, FPR, dice_score


def model_predict(model,test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_prediction = []

    model.eval()
    for image , mask in test_dataloader:

        image = image.to(device).float()
        mask = mask.to(device).float().squeeze()
        pred = model(image).detach().cpu().numpy()

        if len(pred.shape) == 2: pred = np.expand_dims(pred,axis=0)

        all_prediction.append(pred)


        del image
        del mask
        del pred

        torch.cuda.empty_cache()

    all_prediction = np.concatenate(all_prediction,axis=0)

    return all_prediction


def calculate_metrics(predict,target):
    tp, tn, fp, fn = confusion_matrix(predict, target)
    
    iou_s = iou(tp, tn, fp, fn).item()
    dice_s = dice_score(tp, tn, fp, fn).item()
    f1_s = f1_score(tp, tn, fp, fn).item()
    tpr = TPR(tp, tn, fp, fn).item()
    fpr = FPR(tp, tn, fp, fn).item()

    return dice_s, iou_s, f1_s, tpr , fpr




def evaluate_model(model, valid_dataset , batch_size=16):
    threshold = np.arange(0.000,1+0.005,0.005)

    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size,shuffle=False)
    valid_prediction = model_predict(model,valid_dataloader)

    # print(valid_prediction.shape, len(valid_dataset))

    metrics = {'dice':[],'iou':[],'f1_score':[],'TPR':[],'FPR':[]}

    for t in tqdm(threshold):
        total_dice = 0
        total_iou = 0
        total_f1 = 0
        total_tpr = 0
        total_fpr = 0

        for i in range(len(valid_dataset)):
            curr_pred , curr_target = valid_prediction[i], valid_dataset[i][1].squeeze().float()

            curr_pred_t = torch.from_numpy(curr_pred >= t).float()

            dice_s, iou_s, f1_s, tpr , fpr = calculate_metrics(curr_pred_t, curr_target)

            total_dice += dice_s
            total_iou += iou_s
            total_f1 += f1_s
            total_tpr += tpr
            total_fpr += fpr


        total_dice = total_dice / len(valid_dataset)
        total_iou = total_iou / len(valid_dataset)
        total_f1 = total_f1 / len(valid_dataset)
        total_tpr = total_tpr / len(valid_dataset)
        total_fpr = total_fpr / len(valid_dataset)

        metrics['dice'].append(total_dice)
        metrics['iou'].append(total_iou)
        metrics['f1_score'].append(total_f1)
        metrics['TPR'].append(total_tpr)
        metrics['FPR'].append(total_fpr)

    return metrics , threshold