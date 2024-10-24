import numpy as np
import cv2
import skimage


def divide2group(image,mask,group=2,resize_shape = (256,256), discard=True):
    H,W,C = image.shape
    qH , qW = H // 2**(group-1), W // 2**(group-1)


    img_list = []
    mask_list = []

    for i in range(2**(group-1)):
        Hs , He = qH*i, qH*(i+1)
        for j in range(2**(group-1)):
            Ws, We =  qW*j, qW * (j+1)

            cur_image = image[Hs:He, Ws:We,:]
            cur_mask = mask[Hs:He, Ws:We ].astype(float)

            # Discard image with no mask
            if discard and not np.any(cur_mask): continue

            # Resize
            cur_image = cv2.resize(cur_image, dsize=resize_shape, interpolation=cv2.INTER_NEAREST)
            cur_mask = cv2.resize(cur_mask, dsize=resize_shape, interpolation=cv2.INTER_NEAREST)

            img_list.append(cur_image)
            mask_list.append(cur_mask)

    if len(img_list) == 0: return None, None

    return np.stack(img_list,axis=0), np.stack(mask_list,axis=0)

def sliding_window(image, mask , window_size, step_size, discard=True):
    H , W , C = image.shape


    H_step = (H // step_size)
    W_step = (W // step_size)

    H_pad = window_size - (H - step_size * H_step)
    W_pad = window_size - (W - step_size * W_step)

    H_left_pad, H_right_pad = H_pad // 2 , H_pad // 2
    if H_pad % 2 != 0: H_right_pad += 1

    W_left_pad , W_right_pad = W_pad // 2, W_pad // 2
    if W_pad % 2 != 0: W_right_pad += 1

    image_pad = np.pad(image,[(H_left_pad,H_right_pad),(W_left_pad,W_right_pad),(0,0)], mode='constant')
    mask_pad  = np.pad(mask, [(H_left_pad,H_right_pad),(W_left_pad,W_right_pad)],       mode='constant')


    list_image = []
    list_mask = []

    for i in range(H_step + 1):
        Hs , He = i*step_size, i*step_size + window_size

        for j in range(W_step + 1):
            Ws, We = j*step_size , j*step_size + window_size

            curr_image = image_pad[Hs:He,Ws:We,:]
            curr_mask = mask_pad[Hs:He,Ws:We]

            
            if curr_image.shape[:2] != (window_size , window_size): continue

            # Discard image with no mask
            if discard and not np.any(curr_mask): continue

            list_image.append(curr_image)
            list_mask.append(curr_mask)

    if len(list_image) == 0: return None, None

    return np.stack(list_image,axis=0), np.stack(list_mask,axis=0)