import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import torchvision.transforms as T

from sklearn.metrics import auc


from patching import divide2group, sliding_window
from dataset import NucleiDataset



def visualize(images,masks):
    n = images.shape[0]
    idx = np.random.choice(n,1)[0]

    img = images[idx]
    mask = masks[idx]

    fig, axs = plt.subplots(1,3,figsize=(10,10))
    axs[0].set_title('Original image')
    axs[0].imshow(img)

    axs[1].set_title('Mask')
    axs[1].imshow(mask)

    axs[2].set_title('Masked Image')
    axs[2].imshow(img)
    axs[2].imshow(mask,alpha=0.5)
    plt.show()



def visualize_pixel_intensity(image, range = (0,256)):
    hist, bins = np.histogram(image.flatten(),256,range=range)

    cdf = hist.cumsum()  / 100

    plt.plot(cdf,label='cumulative')
    plt.stairs(hist,label='histogram')
    plt.grid()
    plt.legend()
    plt.savefig('pixel.png',transparent=True)
    plt.show()


def recover_from_patch(patches , H_step, W_step):
    N , H , W = patches.shape

    assert N == H_step * W_step

    sv = []
    for i in range(H_step):
        sh = []
        for j in range(W_step):
            sh.append(patches[i*H_step + j])
        sv.append(np.hstack(sh))
    recover = np.vstack(sv)

    return recover

def visualize_prediction(model, image, mask, threshold=0.05, method = 'slide', save=True , group_num = 4 , window_size = 256, step_size = 256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    H , W , C = image.shape



    if method == 'GROUP':
        patch_image, patch_mask = divide2group(image, mask, group=group_num, discard=False)

    elif method == 'SLIDE':
        patch_image , patch_mask = sliding_window(image, mask, window_size=window_size, step_size=step_size, discard=False)

    else:
        raise ValueError("Method Not Allow")


    # Make Prediction
    model.eval()
    prediction = []
    temp_dataset = NucleiDataset(patch_image, patch_mask, T.ToTensor(),augmentation=None)
    temp_dataloader = torch.utils.data.DataLoader(temp_dataset,batch_size=16,shuffle=False)

    for image , mask in temp_dataloader:
        image = image.to(device).float()
        pred  = model(image).detach().cpu().numpy()
        prediction.append(pred)

    prediction = np.concatenate(prediction,axis=0)


    if method =='GROUP':
        if group_num == 1:
            recover_pred_mask = prediction
        else:
            recover_pred_mask = recover_from_patch(prediction , H_step=group_num , W_step=group_num)

    else:
        recover_pred_mask = recover_from_patch(prediction , H_step=(H//step_size) + 1 , W_step=(W//step_size) + 1)

    recover_pred_mask =  cv2.resize(recover_pred_mask,(H,W),interpolation=cv2.INTER_NEAREST)

    fig,axs = plt.subplots(1,3,figsize=(15,15))

    axs[0].imshow(image)
    axs[0].imshow(recover_pred_mask >= threshold, alpha=0.7)
    axs[0].set_title(f"Prediction with threshold:{threshold}")
    axs[0].axis('off')
    axs[1].imshow(mask)
    axs[1].set_title(f"Ground Truth Label")
    axs[1].axis('off')
    axs[2].imshow(image)
    axs[2].set_title("Image")
    axs[2].axis('off')

    if save:
      plt.subplots_adjust(wspace=0, hspace=0)
      fig.savefig(f"Prediction_Result_{threshold}.png",transparent=True,bbox_inches='tight',pad_inches=0)

    plt.show()


def plot_roc_curve(metrics):
    TPR = metrics['TPR']
    FPR = metrics['FPR']

    plt.plot(FPR,TPR)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.show()

    print(f"AUC:{auc(FPR,TPR)}")