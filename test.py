from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch, os
import evaluate
import cv2
from  torch.utils.data import DataLoader
from RailData import bag_te
import time

# 统计Railsem数据集上的测试准确率


def untransform(img):
    img = img.transpose(1, 2, 0)
    img *= np.array([0.29908684, 0.27952745, 0.26577608])
    img += np.array([0.43274517, 0.45175052, 0.4350528])
    img *= np.array([255, 255, 255])
    img = img.astype(np.uint8)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    return img

def test(n_class = 2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # start timing
    prev_time = datetime.now()
    fcn_model = torch.load('result_maybe_paper_model/checkpoints/FCN8s_640_480_rs/fcn_model_{}.pt'.format(50))
    # fcn_model = torch.load('result/checkpoints_32s/fcn_model_{}.pt'.format(50))
    fcn_model = fcn_model.to(device)

    fcn_model.eval()
    visualizations = []
    test_dataloader = DataLoader(bag_te, batch_size=1, shuffle=False, num_workers=0)  #0
    label_trues, label_preds = [], []
    with torch.no_grad():
        for index, (rail, rl_msk) in enumerate(test_dataloader):  # 注意要把RailData里DataLoader的batchsize改为1

            rail = rail.to(device)
            rl_msk = rl_msk.to(device)

            torch.cuda.synchronize()
            start = time.time()

            output = fcn_model(rail)
            output = torch.sigmoid(output)

            output_np = output.cpu().detach().numpy().copy()
            output_np = np.argmin(output_np, axis=1)

            torch.cuda.synchronize()
            end = time.time()
            print(end - start)

            rl_msk_np = rl_msk.cpu().detach().numpy().copy()
            rl_msk_np = np.argmin(rl_msk_np, axis=1)
            rail_np = rail.cpu().detach().numpy().copy()

            # img = untransform(rail_np[0,:,:,:])
            # plt.subplot(1, 3, 1)
            # plt.imshow(img)
            # plt.subplot(1, 3, 2)
            # plt.imshow(np.squeeze(rl_msk_np[0, ...]), 'gray')
            # plt.subplot(1, 3, 3)
            # plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
            # plt.pause(0.5)

            for img, lt, lp in zip(rail, rl_msk_np, output_np):
                label_trues.append(lt)
                label_preds.append(lp)
                # img, lt = untransform(img, lt)
            #     if len(visualizations) < 9:
            #         viz = fcn_utils.visualize_segmentation(
            #             lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
            #             label_names=test_dataloader.dataset.class_names)
            #         visualizations.append(viz)

    metrics = evaluate.label_accuracy_score(
        label_trues, label_preds, n_class=2)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
                Accuracy: {0}
                Accuracy Class: {1}
                Mean IU: {2}
                FWAV Accuracy: {3}'''.format(*metrics))
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)


if __name__ == "__main__":

    # os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    test(n_class = 2)
