# Breast Cancer Nuclei Segmentation using Unet

### Dataset
[Breast Cancer Nuclei Dataset](https://drive.google.com/file/d/1OYOuEM4MEpaMEydF7iutRb197lF5iZnP/view)

### Preprocessing
1. Load Image and Mask
2. Remove Image with no Mask
3. Save as npy file

### Patching
Original Resolution of Image: (2000,2000)
- 2 group: 4 * (1000,1000) Image
- 3 group: 8 * (500,500) Image
- 4 group: 16 * (250,250) Image

Remove Patched Image with No Mask for training

Resize to (256,256) for model training

### Loss Function
- Dice Loss
- Binary Cross-Entropy Loss

### Evaluation Metric
- Dice Coefficience
- IOU


#### Result
- Dice: 36.1%
- IOU: 25.4%
![Prediction_Result_0 28](https://github.com/user-attachments/assets/2d4fc93b-41d7-4a41-a781-fa6b678b3fab)
![Prediction_Result_0 25](https://github.com/user-attachments/assets/235d50ca-0d0b-4b1a-bb97-c417e2f95628)




