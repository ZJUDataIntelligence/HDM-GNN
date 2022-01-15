import torch
import torch.nn as nn

import time
from sklearn.metrics import r2_score
from utils import *
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import global_var


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--output", type=str, help='output path')
args = parser.parse_args()

config_filename = args.config
output_path = args.output


with open(config_filename, 'r') as f:
    config = json.loads(f.read())

seed = config['seed']
setup_seed(seed)

global_var._init()
global_var.set_value('config',config)

from model import STHGNN
from model_NO_N import STHGNN_N
from model_E import STHGNN_E
from model_NO_S import STHGNN_S
from model_NO_T import STHGNN_T
from model_NO_M import STHGNN_noM
from model_NO_F import STHGNN_noF
from model_NO_EMF import STHGNN_c_g


from data_load import *


def train(model, train_data_loader, optimizer):
    model.train()
    # loss_fn = nn.MSELoss()
    loss_fn = nn.HuberLoss(reduction='mean', delta=config['DELTA'])
    st = time.time()
    output_list = []
    label_list = []
    for i, input_data in enumerate(train_data_loader):
        optimizer.zero_grad()
        input_data = input_data.to(device)

        label = input_data.y
        out = model(input_data)

        loss = loss_fn(out, label)

        pred = np.floor(out.cpu().reshape(263).detach().numpy())
        y_true = label.cpu().reshape(263).detach().numpy()
        output_list.append(pred)
        label_list.append(y_true)

        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        if i%50 == 0:
            print(f'{time.time()-st:.2f}: loss: {loss}.  {i}/{len(train_data_loader)}')

    output = np.array(output_list)
    label = np.array(label_list)

    train_mse_loss = mse_np(output, label)
    train_rmse_loss = rmse_np(output, label)
    train_mae_loss = mae_np(output, label)
    train_mape_loss = mape_np(output, label)
    train_r2_loss = r2_score(output.reshape((-1, 1)), label.reshape((-1, 1)))

    return train_mse_loss, train_rmse_loss, train_mae_loss, train_mape_loss, train_r2_loss


def test(model, test_data_loader):
    model.eval()
    output_list = []
    label_list = []
    with torch.no_grad():
        for i, input_data in enumerate(test_data_loader):
            input_data = input_data.to(device)
            label = input_data.y
            out = model(input_data)

            pred = np.floor(out.cpu().reshape(263).numpy())
            y_true = label.cpu().reshape(263).numpy()

            output_list.append(pred)
            label_list.append(y_true)

            torch.cuda.empty_cache()
            if i % 50 == 0:
                print(f'testing...  {i}/{len(test_data_loader)}')

        output = np.array(output_list)
        label = np.array(label_list)

        test_mse_loss = mse_np(output,label)
        test_rmse_loss = rmse_np(output,label)
        test_mae_loss = mae_np(output,label)
        test_mape_loss = mape_np(output,label)
        test_r2_loss = r2_score(np.array(output_list).reshape((-1,1)), np.array(label_list).reshape((-1,1)))

    return test_mse_loss, test_rmse_loss, test_mae_loss, test_mape_loss, test_r2_loss


best_rmse_list = []
best_mae_list = []
best_mape_list = []
for r in range(config['REPEAT_TIMES']):
    plt.cla()
    path = output_path + 'times_{}'.format(r)
    if not os.path.exists(path):
        os.mkdir(path)

    # setup_seed(20+r)
    train_data_loader, test_data_loader = load_data(config['CRIME_LABEL_DATA_PATH'], config['CRIME_DATA_PATH'],
                                                    config['A311_DATA_PATH'],config['POI_DATA_PATH'],
                                                    config['TAXI_DATA_PATH'], config['BIKE_DATA_PATH'],
                                                    config['GEO_DATA_PATH'],config['SIMI_DATA_PATH'], config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    if 'WITHOUT_N' in config and config['WITHOUT_N'] == 1:
        model = STHGNN_N(time_series_length=config['TIME_SERIES_LENGTH'], hidden_channels=config['HIDDEN_CHANNELS'],
                       out_channels=1, num_layers=config['NUM_LAYERS'])
    elif 'WITHOUT_E' in config and config['WITHOUT_E'] == 1:
        model = STHGNN_E(time_series_length=config['TIME_SERIES_LENGTH'], hidden_channels=config['HIDDEN_CHANNELS'],
                         out_channels=1, num_layers=config['NUM_LAYERS'])
    elif 'WITHOUT_S' in config and config['WITHOUT_S'] == 1:
        model = STHGNN_S(time_series_length=config['TIME_SERIES_LENGTH'], hidden_channels=config['HIDDEN_CHANNELS'],
                         out_channels=1, num_layers=config['NUM_LAYERS'])
    elif 'WITHOUT_T' in config and config['WITHOUT_T'] == 1:
        model = STHGNN_T(time_series_length=config['TIME_SERIES_LENGTH'], hidden_channels=config['HIDDEN_CHANNELS'],
                         out_channels=1, num_layers=config['NUM_LAYERS'])
    elif 'WITHOUT_M' in config and config['WITHOUT_M'] == 1:
        model = STHGNN_noM(time_series_length=config['TIME_SERIES_LENGTH'], hidden_channels=config['HIDDEN_CHANNELS'],
                         out_channels=1, num_layers=config['NUM_LAYERS'])
    elif 'WITHOUT_F' in config and config['WITHOUT_F'] == 1:
        model = STHGNN_noF(time_series_length=config['TIME_SERIES_LENGTH'], hidden_channels=config['HIDDEN_CHANNELS'],
                         out_channels=1, num_layers=config['NUM_LAYERS'])
    elif 'CRIME_DATA' in config:
        if config['CRIME_DATA']==1 and config['POI_DATA']==0 and config['311_DATA']==0 and config['GEO_DATA']==1 \
                and config['TAXI_DATA']==0 and config['BIKE_DATA']==0 and config['SIMI_DATA']==0:
            model = STHGNN_c_g(time_series_length=config['TIME_SERIES_LENGTH'],
                               hidden_channels=config['HIDDEN_CHANNELS'],
                               out_channels=1, num_layers=config['NUM_LAYERS'])
        elif config['CRIME_DATA']==1 and config['POI_DATA']==1 and config['311_DATA']==1 and config['GEO_DATA']==1 \
                and config['TAXI_DATA']==1 and config['BIKE_DATA']==1 and config['SIMI_DATA']==1:
            model = STHGNN(time_series_length=config['TIME_SERIES_LENGTH'],
                               hidden_channels=config['HIDDEN_CHANNELS'],
                               out_channels=1, num_layers=config['NUM_LAYERS'])
    else:
        print('invalid config!')
        break
        # model = STHGNN(time_series_length=config['TIME_SERIES_LENGTH'], hidden_channels=config['HIDDEN_CHANNELS'],
        #                out_channels=1, num_layers=config['NUM_LAYERS'])
    model = model.to(device)
    # model = torch.load('/home/zh/temp/new_20nodefeatures_threshold5/model_10_epoch.pth')

    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=config['WEIGHT_DECAY'])

    train_mse_loss_list = []
    train_rmse_loss_list = []
    train_mae_loss_list = []
    train_mape_loss_list = []
    train_r2_loss_list = []

    test_mse_loss_list = []
    test_rmse_loss_list = []
    test_mae_loss_list = []
    test_mape_loss_list = []
    test_r2_loss_list = []

    best = []
    min_rmse = 1000000000.0
    for epoch in range(0, config['EPOCH']):
        start_time = time.time()
        train_mse_loss, train_rmse_loss, train_mae_loss, train_mape_loss, train_r2_loss = train(model, train_data_loader, optimizer)
        test_mse_loss, test_rmse_loss, test_mae_loss, test_mape_loss, test_r2_loss = test(model, test_data_loader)

        if test_rmse_loss < min_rmse:
            min_rmse = test_rmse_loss
            best = [test_mse_loss, test_rmse_loss, test_mae_loss, test_mape_loss, test_r2_loss, epoch]

        train_mse_loss_list.append(train_mse_loss)
        train_rmse_loss_list.append(train_rmse_loss)
        train_mae_loss_list.append(train_mae_loss)
        train_mape_loss_list.append(train_mape_loss)
        train_r2_loss_list.append(train_r2_loss)

        test_mse_loss_list.append(test_mse_loss)
        test_rmse_loss_list.append(test_rmse_loss)
        test_mae_loss_list.append(test_mae_loss)
        test_mape_loss_list.append(test_mape_loss)
        test_r2_loss_list.append(test_r2_loss)

        print(f'{time.time()-start_time:.2f}: Epoch: {epoch:03d}, Train loss: '
              f' MAE {train_mae_loss:.4f};  MAPE {train_mape_loss:.4f}; R2 {train_r2_loss:.4f}')
        print(f'{time.time()-start_time:.2f}: Epoch: {epoch:03d}, Test loss: '
              f' MAE {test_mae_loss:.4f};  MAPE {test_mape_loss:.4f}; R2 {test_r2_loss:.4f}')
        print(f'{time.time() - start_time:.2f}:Best Epoch: {best[5]:03d}, '
              f' MAE {best[2]:.4f};  MAPE {best[3]:.4f}; R2 {best[4]:.4f}')
        if epoch%10==0:
            torch.save(model, path + '/' +'model_{}_epoch.pth'.format(epoch))

    plt.plot(train_rmse_loss_list, label="train rmse")
    plt.plot(test_rmse_loss_list, label="test rmse")
    plt.legend()
    plt.savefig(path + '/' + 'figure.jpg')
    plt.show()

    best_mae_list.append(best[2])
    best_mape_list.append(best[3])
    print('best tesing results: MAE: {:.2f}\ntesting: RMSE: {:.2f}\ntesting: MAPE: {:.2f}\n'.format(best[2], best[1], best[3]))


save_variable(best_mae_list,output_path + 'best_mae_list')
save_variable(best_mape_list,output_path + 'best_mape_list')
print('best MAE list:')
print(best_mae_list)
print('best MAPE list:')
print(best_mape_list)
print('final results mean: MAE: {:.2f}\ntesting: MAPE: {:.2f}\n'.format(np.mean(best_mae_list),np.mean(best_mape_list)))
print('final results range: MAE: {:.2f}\ntesting: MAPE: {:.2f}\n'
      .format((np.max(best_mae_list)-np.min(best_mae_list)),
              (np.max(best_mape_list)-np.min(best_mape_list))))