import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
import torchsummary


from basicblock import BasicBlock
import os
import cv2
import numpy as np
import time
from datetime import timedelta

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

ORIGINAL_PANORAMA_PATH = 'data_/original_img'
HEATMAP_PATH = 'data_/heatmap'
HM_CENTER_PATH = 'data_/hm_center'
LIDAR_PATH = 'data_/lidar'
BBOX_2D_PATH = 'data_/2D_BBOX'
BBOX_3D_PATH = 'data_/3D_BBOX'

class Heatmap2Feat_Dataset_(Dataset):
    def __init__(self, x1_path, x2_path, y_path): # original_img, heatmap, hm_center
        super(Heatmap2Feat_Dataset_, self).__init__()
        self.x1_file_names = os.listdir(x1_path)
        self.x1_file_names.sort()
        self.x1_file_list = [os.path.join(x1_path, filename) for filename in self.x1_file_names]

        self.x2_file_names = os.listdir(x2_path)
        self.x2_file_names.sort()
        self.x2_file_list = [os.path.join(x2_path, filename) for filename in self.x2_file_names]

        self.y_file_names = os.listdir(y_path)
        self.y_file_names.sort()
        self.y_file_list = [os.path.join(y_path, filename) for filename in self.y_file_names]
        
    def __len__(self):
        return len(self.x1_file_list)

    def __getitem__(self, idx):
        orig_pano = cv2.imread(self.x1_file_list[idx], cv2.IMREAD_COLOR)[75:-75, :, :]
        heatmap = np.expand_dims(cv2.imread(self.x2_file_list[idx], cv2.IMREAD_GRAYSCALE)[75:-75, :], axis=-1)

        masked_img = (orig_pano * (heatmap >= 1).astype(np.int)).transpose(2, 0, 1)
        masked_img = torch.tensor(masked_img, dtype=torch.float)

        hm_center_ = cv2.imread(self.y_file_list[idx], cv2.IMREAD_GRAYSCALE)[75:-75, :]
        center_inds = np.where(hm_center_ == 255)
        hm_center = cv2.resize(hm_center_, (1608//4, 362//4)).astype(np.float32) / 255.0
        hm_center[center_inds[0] // 4, center_inds[1] // 4] = 1.
        hm_center = torch.tensor(hm_center, dtype=torch.float).unsqueeze(0)

        return masked_img, hm_center

class Heatmap2FeatNetwork(nn.Module): # 모델 정의 My Convolutional Neural Network
    def __init__(self):
        super(Heatmap2FeatNetwork, self).__init__()
        # 3, 362, 1608
        self.layer1 = nn.Sequential(
            nn.ReplicationPad2d((0, 0, 3, 3)),
            nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=(0, 3), padding_mode='circular', bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        ) # 64, 181, 804
        
        self.layer2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        ) # 64, 90, 402
        
        self.block64 = nn.Sequential(
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        )

        self.block128 = nn.Sequential(
            BasicBlock(64, 128, 2, self.downsample1),
            BasicBlock(128, 128, 1)
        ) # 128, 45, 201



        

        self.layer5 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)
        self.layer6 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        _, _, h, w = x.size() # [N, C, H, W]  3, 362, 1608
        h, w = h // 4, w // 4 # 3, 90, 402
        x = self.layer1(x) # 64, 181, 804
        x = self.layer2(x) # 64, 90, 402
        skip1 = self.block64(x) # 64, 90, 402
        skip2 = self.block128(skip1) # 128, 45, 201
        
        # out3
        out3 = F.interpolate(skip2, scale_factor=2, mode='bilinear', align_corners=True) # 128, 90, 402
        concat_out3 = torch.cat((out3, skip1), dim=1) # concat_out3 [N, 192, 90, 404], C = 128 + 64

        # out4
        output = F.interpolate(self.layer5(concat_out3), size=(h, w), mode='bilinear', align_corners=True) # 64, 90, 402

        output = self.layer6(output) # 1, 90, 402

        output = self.sigmoid(output)
        hmax = F.max_pool2d(output, (3, 3), stride=1, padding=1)
        output = output * (hmax == output).float()

        output = torch.clamp(output, torch.finfo(torch.float32).eps, 1-torch.finfo(torch.float32).eps)
        return output
      
if __name__ == '__main__':
    C, H, W = 3, 362, 1608
    image_data_path = 'data_/original_img'

    dataset_len = len(os.listdir(image_data_path))

    train_dataset, val_dataset = random_split(Heatmap2Feat_Dataset_(ORIGINAL_PANORAMA_PATH, HEATMAP_PATH, HM_CENTER_PATH), [round(dataset_len * 0.9), round(dataset_len * 0.1)])

    train_dataloader = DataLoader(train_dataset, batch_size=12, num_workers=0, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=12, num_workers=0)

    def FL_of_CornerNet(X, y, alpha=2, beta=4):
        p_inds = y.eq(1).float()
        n_inds = (-p_inds + 1.)

        p_loss = (torch.log(X) * torch.pow(1 - X, alpha) * p_inds).sum()
        n_loss = (torch.log(1 - X) * torch.pow(X, alpha) * torch.pow(1 - y, beta) * n_inds).sum()

        p_num = p_inds.sum()
        
        return -(p_loss + n_loss) / p_num

    NEW_MODEL = True

    if NEW_MODEL:
        model = Heatmap2FeatNetwork()
        print('Training New Model')
    else:
        model = torch.load('best_model1.pt')
        print('load model')

    model.to(device)

    torchsummary.summary(model, (C, H, W), batch_size=16, device=device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch = 100
    preval_loss, val_loss = 0.0, 0.0
    total_time, epoch_time, batch_time = time.time(), 0.0, 0.0
    MSE_funcion = nn.MSELoss()
    for i in range(epoch):
        epoch_time = time.time()
        print('epoch: {}'.format(i+1))

        model.to(device)
        model.train()
        batch_time = time.time()
        for batch, (X, Y) in enumerate(train_dataloader):
            X, Y = X.to(device), Y.to(device)

            pred = model(X)
            loss = FL_of_CornerNet(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                current = batch * len(X)
                batch_time = time.time() - batch_time
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{round(dataset_len * 0.9):>5d}] --- time: {timedelta(seconds=round(batch_time))}")
                batch_time = time.time()
        print('train epoch done')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch, (X, Y) in enumerate(val_dataloader):
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                val_loss += FL_of_CornerNet(pred, Y).item()
            if i == 0 or preval_loss > val_loss:
                torch.save(model, 'best_model_hm2feat.pt')
                preval_loss = val_loss
                print(f'val_loss: {val_loss} --- val_loss decreased, best model saved.')
            else:
                print(f'val_loss: {val_loss} --- model not saved')
        epoch_time = time.time() - epoch_time
        print(f'time spent {timedelta(seconds=round(epoch_time))} per epoch')

    print('\n')
    print(f'total learning time: {timedelta(seconds=round(time.time() - total_time))}')
