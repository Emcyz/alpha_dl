import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from basicblock import *
from hm2hm_center_train import Heatmap2FeatNetwork
from img2heatmap_train import Img2Heatmap

# never mind this
class MyCNN(Img2Heatmap):
    def __init__(self):
        super().__init__()

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

img2heatmap = torch.load('best_model_img2heatmap.pt')
img2heatmap.to(device)
img2heatmap.eval()

hm2hm_center = torch.load('best_model_hm2feat.pt')
hm2hm_center.to(device)
hm2hm_center.eval()

hm_threshold = 0.7
ct_threshold = 0.7

for idx in range(len(orig_img_file_list)):
    orig_img = cv2.imread(orig_img_file_list[idx], cv2.IMREAD_COLOR)[75:-75, :, :]
    _orig_img = np.expand_dims(orig_img.transpose(2, 0, 1), axis=0)
    _orig_img = torch.tensor(_orig_img, dtype=torch.float).to(device)
    hm_pred = img2heatmap(_orig_img).cpu().detach().numpy()[0, 0, :, :] # value : 0 ~ 1, shape : (362, 1608)

    heatmap = cv2.imread(hm_file_list[idx], cv2.IMREAD_GRAYSCALE)[75:-75, :]
    masked_img = (orig_img * (heatmap[:, :, np.newaxis] >= 1).astype(np.int32)).transpose(2, 0, 1)
    masked_img = torch.tensor(masked_img[np.newaxis, :, :], dtype=torch.float).to(device)

    hm_ct_pred = hm2hm_center(masked_img).cpu().detach().numpy()[0, 0, :, :] # value : 0 ~ 1, shape : (362 // 4, 1608 // 4)
    # prediction end

    # paint heatmap on original_img with gray color (100, 100, 100)
    heat_x_inds, heat_y_inds = np.where(hm_pred >= hm_threshold)
    orig_img[heat_x_inds, heat_y_inds, :] = 100  

    # mark circle on predicted vehicle center
    p_inds = np.where(hm_ct_pred >= ct_threshold)
    x_inds, y_inds = p_inds
    for x, y in zip(x_inds, y_inds):
        cv2.circle(orig_img, center=(y*4, x*4), radius=7, color=(255, 0, 0), thickness=3)

    cv2.imshow('img', orig_img)
    if cv2.waitKey(0) & 0xFF == 27:
        break
