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

# image size == 3, 1608, 362
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class Img2Heatmap(nn.Module):
    def __init__(self):
        super(Img2Heatmap, self).__init__()
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
        self.downsample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )

        self.block256 = nn.Sequential(
            BasicBlock(128, 256, 2, self.downsample2),
            BasicBlock(256, 256, 1)
        ) # 256, 23, 101

        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU())


        self.out1_layer = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.out2_layer = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.out3_layer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        _, _, h, w = x.size() # [N, C, H, W]  3, 362, 1608
        #h, w = h // 4, w // 4
        x = self.layer1(x) # 64, 181, 804
        x = self.layer2(x) # 64, 90, 402
        skip1 = self.block64(x) # 64, 90, 402
        skip2 = self.block128(skip1) # 128, 45, 201
        skip3 = self.block256(skip2) # 256, 23, 101

        # out1
        out1 = F.interpolate(skip3, scale_factor=2, mode='bilinear', align_corners=True) # 256, 46, 202
        skip2 = F.pad(skip2, pad=(1, 0, 1, 0), mode='circular') # 128, 45 + 1, 201 + 1 
        concat_out1 = torch.cat((out1, skip2), dim=1) # concat_out2 = [N, 384, 46, 202], C = 256 + 128
        
        # out2
        out2 = F.interpolate(self.layer4(concat_out1), scale_factor=2, mode='bilinear', align_corners=True) # 128, 92, 404
        skip1 = F.pad(skip1, pad=(1, 1, 1, 1), mode='circular') # 64, 90 + 2, 402 + 2
        concat_out3 = torch.cat((out2, skip1), dim=1) # concat_out3 [N, 192, 92, 404], C = 128 + 64

        # out3
        out3 = F.interpolate(self.layer5(concat_out3), size=(h//2, w//2), mode='bilinear', align_corners=True) # 64, 181, 804
        
        output = torch.cat(
            [F.interpolate(self.out1_layer(out1), size=(h, w)).unsqueeze(-1), 
            F.interpolate(self.out2_layer(out2), size=(h, w)).unsqueeze(-1),
            F.interpolate(self.out3_layer(out3), size=(h, w)).unsqueeze(-1)], dim=-1).sum(dim=-1)
    
        output = torch.clamp(self.tanh(output), torch.finfo(torch.float32).eps, 1-torch.finfo(torch.float32).eps)
        return output
      
class MyDataset2(Dataset): 
    def __init__(self, x_path, y_path):
        super(MyDataset2, self).__init__()
        self.x_file_names = os.listdir(x_path) 
        self.x_file_names.sort()
        self.x_file_list = [os.path.join(x_path, filename) for filename in self.x_file_names]

        self.y_file_names = os.listdir(y_path)
        self.y_file_names.sort()
        self.y_file_list = [os.path.join(y_path, filename) for filename in self.y_file_names]

    def __len__(self):
        return len(self.x_file_list)

    def __getitem__(self, idx):
        x = np.transpose(cv2.imread(self.x_file_list[idx], cv2.IMREAD_COLOR)[75:-75, :, :], (2, 0, 1)) 
        x = torch.tensor(x, dtype=torch.float)
        
        y = np.expand_dims(cv2.imread(self.y_file_list[idx], cv2.IMREAD_GRAYSCALE)[75:-75, :], axis=0)
        y = torch.tensor(y > 0, dtype=torch.float)
        return x, y 


if __name__ == '__main__':
    C, H, W = 3, 362, 1608
    image_data_path = 'data_/original_img'
    heatmap_data_path = 'data_/heatmap'

    dataset_len = len(os.listdir(image_data_path))
    train_dataset, val_dataset = random_split(MyDataset2(image_data_path, heatmap_data_path), [round(dataset_len * 0.9), round(dataset_len * 0.1)])

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
        model = Img2Heatmap()
        print('Training New Model')
    else:
        model = torch.load('best_model_img2heatmap.pt')
        print('load model')

    model.to(device)

    torchsummary.summary(model, (C, H, W), batch_size=12, device=device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch = 200
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
                torch.save(model, 'best_model_img2heatmap.pt')
                preval_loss = val_loss
                print(f'val_loss: {val_loss} --- val_loss decreased, best model saved.')
            else:
                print(f'val_loss: {val_loss} --- model not saved')
        epoch_time = time.time() - epoch_time
        print(f'time spent {timedelta(seconds=round(epoch_time))} per epoch')

    print('\n')
    print(f'total learning time: {timedelta(seconds=round(time.time() - total_time))}')
