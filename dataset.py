import torch


class NucleiDataset(torch.utils.data.Dataset):
  def __init__(self,images,masks,transformation = None,augmentation = None):
      assert images.shape[0] == masks.shape[0], "Images and Masks have to be same size"

      self.n = images.shape[0]
      self.images = images
      self.masks = masks
      self.transformation = transformation
      self.augmentation = augmentation


  def __len__(self):
      return self.n

  def __getitem__(self,idx):
      curr_image = self.images[idx]
      curr_mask  = self.masks[idx].astype(float)


      if self.transformation is not None:
          curr_image = self.transformation(curr_image) # (C,H,W)
          curr_mask = self.transformation(curr_mask)

      if self.augmentation is not None:
          for aug_fn in self.augmentation:
              curr_image, curr_mask = aug_fn(curr_image,curr_mask)

      return [curr_image, curr_mask]
