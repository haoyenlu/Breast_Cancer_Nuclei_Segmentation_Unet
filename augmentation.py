import torch
import cv2
import torchvision.transforms as T


def randomly_apply(p=0.5):
    num = torch.rand(1)

    if num >= p: return True
    return False


class RandomRotation():
    def __init__(self,p=0.5):
        self.p = p

    def __call__(self, image , mask = None):
        if randomly_apply(self.p):
            angle = torch.randint(0, 360, (1,)).item() # Get random angle from 0 to 360

            rotate_image = T.functional.rotate(image, angle)
            rotate_mask  = T.functional.rotate(mask , angle) if mask is not None else None

            return rotate_image, rotate_mask

        return image , mask

class RandomHorizontalFlip():
    def __init__(self,p=0.5):
        self.p = p
        self.axflip = -2

    def __call__(self, image , mask = None):
        if randomly_apply(self.p):

            flip_image =  torch.flip(image, (self.axflip,))
            flip_mask  =  torch.flip(mask , (self.axflip,)) if mask is not None else None

            return flip_image, flip_mask

        return image , mask

class RandomVerticalFlip():
    def __init__(self,p=0.5):
        self.p = p
        self.axflip = -1

    def __call__(self, image , mask = None):
        if randomly_apply(self.p):

            flip_image =  torch.flip(image, (self.axflip,))
            flip_mask  =  torch.flip(mask , (self.axflip,)) if mask is not None else None

            return flip_image, flip_mask

        return image , mask

