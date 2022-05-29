#!/usr/bin/env python

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from hm2hm_center_train import Heatmap2FeatNetwork
from img2heatmap_train import Img2Heatmap

x1_path = './data_/original_img'
x2_path = './data_/heatmap'

x1_file_names = os.listdir(x1_path)
x1_file_names.sort()
x1_file_list = [os.path.join(x1_path, filename) for filename in x1_file_names]
x2_file_names = os.listdir(x2_path)
x2_file_names.sort()
x2_file_list = [os.path.join(x2_path, filename) for filename in x2_file_names]

model = torch.load('best_model_hm2feat.pt')
model.to(device)
model.eval()

for idx in range(len(x1_file_list)):
    orig_pano = cv2.imread(x1_file_list[idx], cv2.IMREAD_COLOR)[75:-75, :, :]
    heatmap = np.expand_dims(cv2.imread(x2_file_list[idx], cv2.IMREAD_GRAYSCALE)[75:-75, :], axis=-1)

    masked_img = (orig_pano * (heatmap >= 1).astype(np.int)).transpose(2, 0, 1)
    masked_img = torch.tensor(np.expand_dims(masked_img, 0), dtype=torch.float)
    masked_img = masked_img.to(device)

    pred = np.array(model(masked_img).cpu().detach().numpy() * 255., dtype=np.uint8)[0, 0, :, :]
    pred = cv2.resize(pred, (1608, 362))
    p_inds = np.where(pred >= 128)
    img = orig_pano.copy().astype(np.float32)
    #img[:, :, 0] += pred * 255.
    #img[:, :, 2] += pred
    #img[img >= 255.0] = 255.0
    x_inds, y_inds = p_inds
    for x, y in zip(x_inds, y_inds):
        cv2.circle(img, (y, x), 5, (255, 0, 0), -1)

    img = img.astype(np.uint8)
    cv2.imshow('img', img)
    cv2.imshow('heatmap', pred)
    if cv2.waitKey(0) & 0xFF == 27:
        break

model = torch.load('best_model_normal_heatmap.pt')
model.to(device)
model.eval()

video_file = 'testset/testvideo.avi'
cap = cv2.VideoCapture(video_file)
with torch.no_grad():
    while True:
        ret, img = cap.read()
        X = torch.tensor(np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0), dtype=torch.float)
        X = X.to(device)
        pred = np.array(model(X).cpu().detach().numpy(), dtype=np.float)
        pred = pred[0, 0, :, :]
        red_mask = (pred > 0.7).astype(np.float)
        #result = cv2.resize(np.squeeze(np.array(pred.cpu())), dsize=None, fx=4, fy=4)
        img = np.array(img, dtype=np.float32)
        img[:, :, 0] += pred * 255.
        img[:, :, 2] += pred * 255.
        img[img >= 255.0] = 255.0
        img = np.array(img, dtype=np.uint8)
        cv2.imshow('img', img)
        cv2.imshow('heatmap', pred)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
