import numpy as np
from tqdm import tqdm
import PIL.Image
import os
import skimage
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt

from preprocess import sliding_window, divide2group
from visualization import visualize , visualize_prediction
from augmentation import RandomRotation , RandomHorizontalFlip , RandomVerticalFlip
from dataset import NucleiDataset
from model import UNet
from train import train_model
from evaluation import evaluate_model

'''Pre-Processing Image'''
image_dir = './Original/'
mask_dir  = './Mask/'

mask_files = os.listdir(mask_dir)
img_files = os.listdir(image_dir)

# Remove images that have no mask
img_files.remove('10269_00022.tif')

all_filename = [x.split('_mask')[0] for x in mask_files]
img_filename = [x + '_original.tif' for x in all_filename]
mask_filename = [x + '_mask.png' for x in all_filename]


'''Prameter setting'''
# METHOD = 'GROUP' 
GROUP_NUM = 2


METHOD = 'SLIDE'
WINDOW_SIZE = 256
STEP_SIZE = 256

CONTRAST = False

image_list , mask_list = [], []

'''Processe each iamge'''
for i in tqdm(range(len(all_filename))):
    img, mask = PIL.Image.open(image_dir + img_filename[i]), PIL.Image.open(mask_dir + mask_filename[i])
    img, mask = np.array(img), np.array(mask)
    img = img / 255.0 # Normalization

    if CONTRAST:
        img = skimage.exposure.equalize_hist(img,nbins=256)

    if METHOD == 'GROUP':
        image_patch , mask_patch = divide2group(img, mask , group=GROUP_NUM,resize_shape=(256,256),discard=True)

    elif METHOD == 'SLIDE':
        image_patch , mask_patch = sliding_window(img, mask, WINDOW_SIZE, STEP_SIZE, discard=True)

    image_list.append(image_patch)
    mask_list.append(mask_patch)

all_images = np.concatenate(image_list, axis=0)
all_masks = np.concatenate(mask_list,  axis=0)

# print(all_images.shape, all_masks.shape)
# visualize(all_images, all_masks)

num_samples = all_images.shape[0]
split_ratio = 0.8
batch_size = 16
train_size = int(num_samples * split_ratio)

'''Dataset'''
augmentation = [
    RandomRotation(p=0.5),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5)
]

transformation = T.ToTensor()

train_dataset = NucleiDataset(all_images[:train_size],all_masks[:train_size],transformation, augmentation )
test_dataset =  NucleiDataset(all_images[train_size:],all_masks[train_size:],transformation, augmentation  = None)

train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

# print(len(train_dataset),len(test_dataset))
# print(test_dataset[0][0].shape,test_dataset[0][1].shape)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
num_classes = 1
EPOCHS = 50
BCE_WEIGHT = 0.5
model = UNet(num_classes).to(device)

TRAIN = False
CKPT  = True


if TRAIN:
    dataloaders = {'train':train_dataloader,'val':test_dataloader}
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=1e-4)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.8)
    model , history = train_model(model, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs=EPOCHS, bce_weight=BCE_WEIGHT)

    plt.plot(history['train']['loss'],label='train')
    plt.plot(history['val']['loss'],label='valid')
    plt.show()

elif CKPT:
    model_name = 'best_weight_256_256'
    checkpoint = f'./checkpoint/{model_name}.pth'
    ckpt = torch.load(checkpoint,map_location=device)
    model.load_state_dict(ckpt)


EVALUATE = True


if EVALUATE:
    valid_dataset = NucleiDataset(all_images[train_size:],all_masks[train_size:],transformation, augmentation  = None)
    metrics , threshold = evaluate_model(model, valid_dataset)
    fig,axs = plt.subplots(1,3,figsize=(15,8))

    scale_fn = lambda x: x * 100

    axs[0].plot(threshold,list(map(scale_fn,metrics['dice'] )))
    axs[1].plot(threshold,list(map(scale_fn,metrics['iou'])))
    axs[2].plot(threshold,list(map(scale_fn,metrics['f1_score'])))

    fig.savefig('Evaluation.png',bbox_inches='tight')
    plt.show()
    print(f"Max Dice Score:{max(metrics['dice'])}, Max IOU Score:{max(metrics['iou'])}, Max F1_Score:{max(metrics['f1_score'])}")
    print(f"Max Dice Threshold:{threshold[np.argmax(metrics['dice'])]}, Max IOU Threshold:{threshold[np.argmax(metrics['iou'])]}, Max F1_score Threshold:{threshold[np.argmax(metrics['f1_score'])]}")


SEE_PREDICTION = True
if SEE_PREDICTION:
    idx = np.random.randint(0,len(img_filename))
    img, mask = PIL.Image.open(image_dir + img_filename[i]), PIL.Image.open(mask_dir + mask_filename[i])
    img, mask = np.array(img), np.array(mask)
    img = img / 255.0 # Normalization

    if CONTRAST:
        img = skimage.exposure.equalize_hist(img,nbins=256)

    visualize_prediction(model,img , mask , threshold = threshold[np.argmax(metrics['dice'])] , method = METHOD , group_num = GROUP_NUM , window_size = WINDOW_SIZE, step_size = STEP_SIZE , save=True)