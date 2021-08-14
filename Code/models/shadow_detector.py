"""
MODEL
"""
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

args = {
    'snapshot': '3000',
    'scale': 416
}

img_transform = transforms.Compose([
    transforms.Resize(args['scale']),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

import os
dirFile = os.path.dirname(__file__)

if __name__ == '__main__':
    rel_path_img = "../../Data/ISTD_Dataset/train/train_A/11-1.png"
    path = os.path.join(dirFile,rel_path_img)

    rel_path_model = "../../Data/BDRAR.pth"
    path_model = os.path.join(dirFile,rel_path_model)
    img = Image.open(path)
    preprocess_img = img_transform(img)
    preprocess_img = preprocess_img.unsqueeze(0)

    model = torch.load(path_model)
    model = model.to("cpu")
    model.eval()
    out = model(preprocess_img).to("cpu").detach().numpy()

    fig,axs = plt.subplots(1,2)

    axs[0].imshow(np.moveaxis(preprocess_img.numpy().squeeze(), 0, 2))
    axs[1].imshow(out.squeeze(),cmap='gray')

    plt.show()

