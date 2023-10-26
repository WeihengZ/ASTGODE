import os, sys

from lib.data import load_graphdata_channel1
from lib.adj_mat import get_adjacency_matrix, scaled_Laplacian, cheb_polynomial, distanceA_to_adj, load_graph_data, diffusion_polynomial
from lib.metrics import *

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse


# set the random seed
torch.manual_seed(10)

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--case', type=str, default = None)
parser.add_argument('--adjoint', type=int, default=0)
args = parser.parse_args()

if args.adjoint == 0:
    print('Using adjoint model training...')
    from lib.model_adjoint import Net
else:
    print('Using normal model training...')
    from lib.model import Net

# define the device
if torch.cuda.is_available():
    device = torch.device('cuda:0') 
else:
    device = torch.device('cpu')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ implementation starts here ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``

if args.case == 'PEMS04':
# load the data and construct graph adj matrix
    [train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std] = \
        load_graphdata_channel1(graph_signal_matrix_filename = r'./data/PEMS04/PEMS04', \
            num_of_hours = 1, num_of_days = 1, num_of_weeks=1, DEVICE=torch.device("cpu"), batch_size=32, shuffle=True)
    # load the adj matrix
    A, distA = get_adjacency_matrix(distance_df_filename=r'./data/PEMS04/PEMS04.csv', num_of_vertices=307, id_filename=None)
    W = distanceA_to_adj(A, distA)
    node_size = 307
    used_feature_dim = 3


if args.case == 'PEMS_bay':
    [train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std] = \
        load_graphdata_channel1(graph_signal_matrix_filename = r'./data/PEMS_bay/pems-bay', \
            num_of_hours = 1, num_of_days = 1, num_of_weeks=1, DEVICE=torch.device("cpu"), batch_size=24, shuffle=True)
    sensor_ids, sensor_id_to_ind, W = load_graph_data(r'./data/PEMS_bay/adj_mx_bay.pkl')
    node_size = 325
    used_feature_dim = 1

# define the chebyshev polynomial 
L_tile = scaled_Laplacian(W)
cheb_polys = cheb_polynomial(L_tile, 3)

# define training function
def train(loader, model, optimizer, criterion, device):

    '''
    p.s. input is (batch, #node, #time_step, #feature)
         output is (batch, #node, #time_step)
    '''

    batch_loss = 0 
    for idx, (inputs, targets) in enumerate(tqdm(loader)):

        model.train()
        optimizer.zero_grad()

        inputs = inputs.to(device)[:,:,:,:]    # (B,N,F,T)
        targets = targets.to(device)
        predict = model(inputs)

        loss = criterion(predict, targets) # + criterion(recon, (inputs[:,:,:,0:12]).squeeze(-2))
        loss.backward()
        optimizer.step()

        batch_loss += loss.detach().cpu().item()
    return batch_loss / (idx + 1)

# define evaluation function
@torch.no_grad()
def eval(loader, model, std, mean, device):
    batch_rmse_loss = np.zeros(12)
    batch_mae_loss = np.zeros(12)
    batch_mape_loss = np.zeros(12)
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.eval()

        inputs = (inputs[:,:,:,:]).to(device) # (B,N,F,T)
        targets = targets.to(device) # (B,N,T)
        output = model(inputs) # (B,N,T)
        
        out_unnorm = output.detach().cpu().numpy()
        target_unnorm = targets.detach().cpu().numpy()

        mae_loss = masked_mae_np(target_unnorm, out_unnorm, 0)
        rmse_loss = masked_rmse_np(target_unnorm, out_unnorm, 0)
        # mape_loss = masked_mape_np(target_unnorm, out_unnorm, 0)
        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        # batch_mape_loss += mape_loss
    
    print('rmse loss:', batch_rmse_loss / (idx + 1))
    print('mae loss:', batch_mae_loss / (idx + 1))

    return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1) # , batch_mape_loss / (idx + 1)

# change the location of polynomials to device
new_chev_polys = []
for i in range(len(cheb_polys)):
    new_chev_polys.append(torch.from_numpy(cheb_polys[i]).to(device))

# defien the model
# by now, nb_chev_filter must be same as the feature hidden state because of NODE
model = Net(DEVICE=device, cheb_polys=new_chev_polys, in_channels=used_feature_dim, K=diff_step, nb_chev_filter=64, hidden_feature_dim=64, time_strides=1,\
    num_of_vertices=node_size, num_of_timesteps=12, t_span=torch.tensor([0,1,2,3])).float().to(device)

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# define loss function
# criterion = nn.MSELoss() # default delta == 1
criterion = nn.L1Loss()

# store error_history
mae_hist = []
rmse_hist = []

for epoch in range(30):

    torch.save(model, './model_{}.pkl'.format(args.case))

    # print the important parameter

    rm, ma = eval(val_loader, model, std, mean, device)

    print(epoch)

    train_loss = train(train_loader, model, optimizer, criterion, device)

    print('training loss:', train_loss)

    mae_hist.append(np.mean(ma))
    rmse_hist.append(np.mean(rm))
    plt.figure()
    x = np.arange(len(mae_hist))
    plt.plot(x, mae_hist, label="MAE")
    plt.plot(x, rmse_hist, label="RMSE")
    plt.legend(loc=0)
    plt.xlabel("training epoch")
    plt.ylabel("error")

    plt.title("convergence test of ASTGODE on {}".format(args.case))
    plt.savefig(r'./error_{}.png'.format(args.case))
    np.save(r'./mae_err_non_adjoint.npy', mae_hist)
    np.save(r'./rmse_err_non_adjoint.npy', rmse_hist)









