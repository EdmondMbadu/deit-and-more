from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt
import requests
from PIL import Image
import torch
import os
from os import listdir
# check you have the right version of timm
import numpy as np
import timm
from setuptools import sic
from timm.utils import ModelEma

assert timm.__version__ == "0.3.2"


torch.set_grad_enabled(False)

with open("imagenet_classes.txt", "r") as f:
    imagenet_categories = [s.strip() for s in f.readlines()]
# create the data transform that DeiT expects
transform = T.Compose([
    T.Resize(256, interpolation=3),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

model = torch.hub.load('facebookresearch/deit:main',
                       'deit_base_patch16_224', pretrained=True)
model.eval()

# Url of images taken on the internet
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# url = 'https://media.vanityfair.com/photos/5e27310def889c00087c7928/2:3/w_887,h_1331,c_limit/taylor-swift-cats.jpg'
# url = 'https://i.stack.imgur.com/UYYqo.jpg'
im = Image.open(requests.get(url, stream=True).raw)

# transform the original image and add a batch dimension
img = transform(im).unsqueeze(0)

# compute the predictions
out = model(img)

# and convert them into probabilities
scores = torch.nn.functional.softmax(out, dim=-1)[0]

# finally get the index of the prediction with highest score
topk_scores, topk_label = torch.topk(scores, k=5, dim=-1)

for i in range(5):
    pred_name = imagenet_categories[topk_label[i]]
    print(
        f"Prediction index {i}: {pred_name:<25}, score: {topk_scores[i].item():.3f}")
