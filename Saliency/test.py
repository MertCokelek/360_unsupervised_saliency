import numpy as np
from skimage.transform import resize
from model_2 import SalGAN_4
import os
import cv2
import torch


model = SalGAN_4()
weight = torch.load('./trained_models/random_decoder/final_models/finetuned.pth')

model.load_state_dict(weight['model'], strict=True)


model.cuda()

IMG_Path = "/home/yasser/Desktop/DATA360/images/"

image_list = os.listdir(IMG_Path)

if not os.path.isdir('./V'):
    os.makedirs('./V')

if not os.path.isdir('./saliency'):
    os.makedirs('./saliency')

print(image_list)
# i =80
for img in image_list:
    image_path = IMG_Path + img
    print(img)
    ori = cv2.imread(image_path)
    inpt = cv2.resize(ori, (320, 160))

    inpt = np.float32(inpt)
    # inpt-=[0.485, 0.456, 0.406]
    inpt = torch.cuda.FloatTensor(inpt)

    inpt = inpt.permute(2, 0, 1)

    inpt = torch.cuda.FloatTensor(inpt)

    with torch.no_grad():
        saliency_map = model(inpt.unsqueeze(0))

    Output = saliency_map
    Output = (Output.cpu()).detach().numpy()
    Output = Output.squeeze()
    Output = resize(Output, (1024, 2048))
    np.save('./V/' + img[:-4] + '.npy', Output)
    cv2.imwrite('./saliency/' + img[:-4] + '.png', (Output - Output.min()) * 255 / (Output.max() - Output.min()))
