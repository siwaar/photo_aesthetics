import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from model.model import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to pretrained model')
#parser.add_argument('--test_csv', type=str, help='test csv file')
parser.add_argument('--test_images', type=str, help='path to folder containing images')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--predictions', type=str, help='output file to store predictions')
args = parser.parse_args()

mobilenet = models.mobilenet_v2(pretrained=True)
model = NIMA(mobilenet)

try:
    device = torch.device('cpu')
    model.load_state_dict(torch.load(args.model, map_location=device), strict=False)
    print('successfully loaded model')
except:
    raise

seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

model.eval()

test_transform = transforms.Compose([
    transforms.Resize((256,256)), 
    transforms.RandomCrop(224), 
    transforms.ToTensor()
    ])

testing_imgs  = args.test_images
print("testing_imgs:", os.listdir(testing_imgs))
images_list = os.listdir(testing_imgs)
mean, std = 0.0, 0.0


for i,img in enumerate(images_list):
    print("here:", i,img)
    path_im = os.path.join(args.test_images, str(img))
    print(path_im)
    im = Image.open(path_im) #the path to the folder with images
    im = im.convert('RGB')
    imt = test_transform(im)
    imt = imt.to(device)
    with torch.no_grad():
        imt = imt.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        out = model(imt)
    out = out.view(10, 1)
    for j, e in enumerate(out, 1):
        mean += j * e
    for k, e in enumerate(out, 1):
        std += e * (k - mean) ** 2
    std = std ** 0.5
    if not os.path.exists(args.predictions):
        os.makedirs(args.predictions)
    with open(os.path.join(args.predictions, 'my_pred.txt'), 'a') as f:
          f.write(str(img) + ' mean: %.3f | std: %.3f\n' % (mean, std))

    mean, std = 0.0, 0.0
    #pbar.update()