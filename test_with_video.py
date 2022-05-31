import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os

from hm2hm_center_train import Heatmap2FeatNetwork
from img2heatmap_train import Img2Heatmap

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

original_img_path = './original_img'
heatmap_path = './heatmap'

orig_img_file_names = os.listdir(original_img_path)
orig_img_file_names.sort()
orig_img_file_list = [os.path.join(original_img_path, filename) for filename in orig_img_file_names]
hm_file_names = os.listdir(heatmap_path)
hm_file_names.sort()
hm_file_list = [os.path.join(heatmap_path, filename) for filename in hm_file_names]

img2heatmap = torch.load('best_model_normal_heatmap.pt')
img2heatmap.to(device)
img2heatmap.eval()

hm2hm_center = torch.load('best_model_hm2feat.pt')
hm2hm_center.to(device)
hm2hm_center.eval()

hm_threshold = 0.7
ct_threshold = 0.7

for idx in range(len(orig_img_file_list)):
    orig_img = np.expand_dims(cv2.imread(orig_img_file_list[idx], cv2.IMREAD_COLOR)[75:-75, :, :], axis=0)
    orig_img = torch.tensor(orig_img, dtype=torch.float).to(device)
    hm_pred = img2heatmap(orig_img).cpu().detach().numpy()[0, 0, :, :] # value : 0 ~ 1, shape : (362, 1608)

    heatmap = np.expand_dims(cv2.imread(hm_file_list[idx], cv2.IMREAD_GRAYSCALE)[75:-75, :], axis=-1)
    masked_img = (orig_img * (heatmap >= 1).astype(np.int)).transpose(2, 0, 1)
    masked_img = torch.tensor(np.expand_dims(masked_img, 0), dtype=torch.float).to(device)

    hm_ct_pred = hm2hm_center(masked_img).cpu().detach().numpy()[0, 0, :, :] # value : 0 ~ 1, shape : (362 // 4, 1608 // 4)

    # prediction end

    heat_x_inds, heat_y_inds = np.where(hm_pred >= hm_threshold)
    orig_img[heat_x_inds, heat_y_inds, :] = 100

    p_inds = np.where(hm_ct_pred >= ct_threshold)
    x_inds, y_inds = p_inds
    for x, y in zip(x_inds, y_inds):
        cv2.circle(orig_img, (y*4, x*4), 7, (255, 0, 0))

    cv2.imshow('img', orig_img)
    if cv2.waitKey(0) & 0xFF == 27:
        break
