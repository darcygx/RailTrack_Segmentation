import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
# import matplotlib.pyplot as plt
import cv2
from onehot import onehot

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.46820313, 0.4618998, 0.46163976], std=[0.2412799, 0.24061733, 0.2378084])])  # 本地图像[640,480]
    transforms.Normalize(mean=[0.43274517, 0.45175052, 0.4350528], std=[0.29908684, 0.27952745, 0.26577608])])  # rs图像[640,480]


class RailDataset(Dataset):

    def __init__(self, transform=None, trainornot=True):
        self.transform = transform
        if trainornot:
            self.path = 'mydataset/rail_data_train/train'
            self.path_msk = 'mydataset/rail_data_train/train_msk'
        else:
            self.path = 'mydataset/rail_data_test/test'
            self.path_msk = 'mydataset/rail_data_test/test_msk'
    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, idx):
        img_name = os.listdir(self.path)[idx]
        img_name_msk = os.listdir(self.path_msk)[idx]
        imgA = cv2.imread(self.path + '/' + img_name)
        imgA = cv2.resize(imgA, (640,480))
        # plt.imshow(imgA)
        # plt.show()
        # imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
        # plt.imshow(imgA)
        # plt.show()
        imgB = cv2.imread(self.path_msk + '/' + img_name_msk, 0)
        imgB = cv2.resize(imgB, (640,480))
        # plt.imshow(imgB)
        # plt.show()
        imgB = imgB / 255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.transpose(2, 0, 1)
        imgB = torch.FloatTensor(imgB)
        # print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)

        return imgA, imgB


bag_tr = RailDataset(transform, trainornot=True)
bag_te = RailDataset(transform, trainornot=False)

train_size = int(len(bag_tr))
test_size = int(len(bag_te))

train_dataloader = DataLoader(bag_tr, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(bag_te, batch_size=4, shuffle=False, num_workers=4)

if __name__ == '__main__':

    for train_batch in train_dataloader:
        print(train_batch)

    for test_batch in test_dataloader:
        print(test_batch)