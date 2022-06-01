from datetime import datetime

import numpy as np
import torch
import os
import cv2
from torchvision import transforms

# 将本地图片分割


# def untransform(img):
#     img = img.transpose(1, 2, 0)
#     img *= np.array([0.29908684, 0.27952745, 0.26577608])
#     img += np.array([0.43274517, 0.45175052, 0.4350528])
#     img *= np.array([255, 255, 255])
#     img = img.astype(np.uint8)
#     b, g, r = cv2.split(img)
#     img = cv2.merge([r, g, b])
#     return img

def test(n_class = 2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # start timing
    prev_time = datetime.now()
    # fcn_model = torch.load('settings1/checkpoints/FCN8s_640_480_rs/fcn_model_{}.pt'.format(50))
    fcn_model = torch.load('result/checkpoints_8s/fcn_model_{}.pt'.format(50))
    fcn_model = fcn_model.to(device)
    file_dir = 'local_scene_dataset/vali'
    save_dir = 'local_scene_dataset/vali_msk'
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.46820313, 0.4618998, 0.46163976], std=[0.2412799, 0.24061733, 0.2378084])])  # 本地图像[640,480]
        transforms.Normalize(mean=[0.43274517, 0.45175052, 0.4350528],std=[0.29908684, 0.27952745, 0.26577608])])  # rs图像[640,480]

    # fcn_model.eval()
    with torch.no_grad():
        for file in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file)
            img = cv2.imread(file_path)
            # img = cv2.resize(img, (640, 480))

            img = transform(img)
            img = img.to(device)
            img = img.unsqueeze(0)

            output = fcn_model(img)
            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])

            output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
            output_np = np.argmax(output_np, axis=1).squeeze()  # 这里argmin可以使标签0,1反过来
            output_np = np.uint8(output_np*255)

            cc = file.split('.')
            cv2.imwrite(save_dir+'/'+cc[0]+'.png', output_np)
            # plt.pause(0.5)


if __name__ == "__main__":

    test(n_class = 2)

