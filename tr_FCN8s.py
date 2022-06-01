from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from RailData import test_dataloader, train_dataloader
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
import evaluate

# 在Railsem数据集上训练


def train(epo_num=50, show_vgg_params=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = FCN8s(pretrained_net=vgg_model, n_class=2)
    fcn_model = fcn_model.to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # start timing
    prev_time = datetime.now()
    for epo in range(epo_num):

        train_loss = 0
        fcn_model.train()
        for index, (rail, rail_msk) in enumerate(train_dataloader):

            rail = rail.to(device)
            rail_msk = rail_msk.to(device)

            optimizer.zero_grad()
            output = fcn_model(rail)
            output = torch.sigmoid(output)
            loss = criterion(output, rail_msk)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy()  # detach的效果与data效果一样，但是更安全 copy指完全复制，与原来不影响
            output_np = np.argmin(output_np, axis=1)
            rail_msk_np = rail_msk.cpu().detach().numpy().copy()
            rail_msk_np = np.argmin(rail_msk_np, axis=1)

            if np.mod(index, 50) == 0:
                # print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
                doc = open('result/out_iter_8s.txt', 'a+')
                doc.write('\nepoch {}, {}/{},train loss, {}'.format(epo, index, len(train_dataloader), iter_loss))
                doc.close()

        test_loss = 0
        label_trues, label_preds = [], []

        fcn_model.eval()
        with torch.no_grad():
            for index, (rail, rail_msk) in enumerate(test_dataloader):
                rail = rail.to(device)
                rail_msk = rail_msk.to(device)

                optimizer.zero_grad()
                output = fcn_model(rail)
                output = torch.sigmoid(output)
                loss = criterion(output, rail_msk)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy()
                output_np = np.argmin(output_np, axis=1)
                rail_msk_np = rail_msk.cpu().detach().numpy().copy()
                rail_msk_np = np.argmin(rail_msk_np, axis=1)
                for img, lt, lp in zip(rail, rail_msk_np, output_np):
                    label_trues.append(lt)
                    label_preds.append(lp)
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
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        # print('epoch train loss = %f, epoch test loss = %f, %s'
        #        %(train_loss/len(train_dataloader), test_loss/len(test_dataloader), time_str))
        doc = open('result/out_epoch_8s.txt', 'a+')
        doc.write('\nepoch train loss = %f, epoch test loss = %f, %s'
                  % (train_loss / len(train_dataloader), test_loss / len(test_dataloader), time_str))
        doc.close()

        if np.mod(epo, 10) == 0:
            torch.save(fcn_model, 'result/checkpoints_8s/fcn_model_{}.pt'.format(epo))
            # print('saveing checkpoints/fcn_model_{}.pt'.format(epo))


if __name__ == "__main__":
    train(epo_num=51, show_vgg_params=False)
