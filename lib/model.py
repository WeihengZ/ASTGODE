__all__ = ['Net']

from typing_extensions import final
import torch
# from torch._C import float32, float64
import torch.nn as nn
import torch.nn.functional as F

from lib.adj_mat import cheb_polynomial

from torchdiffeq import odeint



class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.rand(num_of_timesteps).float().to(DEVICE))
        self.W2 = nn.Parameter(torch.rand(in_channels, num_of_timesteps).float().to(DEVICE))
        self.W3 = nn.Parameter(torch.rand(in_channels).float().to(DEVICE))
        self.bs = nn.Parameter(torch.rand(1, num_of_vertices, num_of_vertices).float().to(DEVICE))
        self.Vs = nn.Parameter(torch.rand(num_of_vertices, num_of_vertices).float().to(DEVICE))


    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        # print(lhs)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        # print(rhs)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        # print(product)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        # print(S)

        S_normalized = F.softmax(S, dim=1)

        # print(S_normalized)

        return S_normalized


class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, DEVICE, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = DEVICE
        self.Theta = nn.ParameterList([nn.Parameter(torch.rand(in_channels, out_channels).float().to(self.DEVICE)) for _ in range(K)])    # for each chebeshev, we will have one parameter

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = (self.cheb_polynomials[k]).to(self.DEVICE)  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = (T_k_with_at.double()).permute(0, 2, 1).matmul(graph_signal.double())  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                '''Vincent modified'''
                output = output + rhs.matmul(theta_k.double())  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)

class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.rand(num_of_vertices).float().to(DEVICE))
        self.U2 = nn.Parameter(torch.rand(in_channels, num_of_vertices).float().to(DEVICE))
        self.U3 = nn.Parameter(torch.rand(in_channels).float().to(DEVICE))
        self.be = nn.Parameter(torch.rand(1, num_of_timesteps, num_of_timesteps).float().to(DEVICE))
        self.Ve = nn.Parameter(torch.rand(num_of_timesteps, num_of_timesteps).float().to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        x = x.float()

        # print('TAT x:', torch.max(torch.isnan(x)))

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        # print('TAT lhs:', torch.max(torch.isnan(lhs)))

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        # print('TAT rhs:', torch.max(torch.isnan(rhs)))

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        # print('TAT product:', torch.max(torch.isnan(product)))

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        # print('TAT E:', torch.max(torch.isnan(E)))

        E_normalized = F.softmax(E, dim=1)

        # print('TAT E_normalized:', torch.max(torch.isnan(E_normalized)))

        return E_normalized

class ODEFunc(nn.Module):

    '''
    Description: Define the ODE function.

    Args:
    # feature_dim: feature dimension of the signals (e.g. out_channels[-1])
    # temporal_dim: temporal dimension of the signals (e.g. 12)
    # adj: adjacent matrix (#node, #node)

    Input:
    # --- t: A tensor with shape [] (scalar), meaning the current time.
    # --- x: A tensor with shape (B,N,T,F_hidden), meaning the value of x at t.

    Output:
    # --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
    
    '''

    def __init__(self, DEVICE, cheb_polys, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_of_vertices, num_of_timesteps):
        super(ODEFunc, self).__init__()

        self.x0 = None
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(DEVICE, K, cheb_polys, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))

    def forward(self, t, x):

        x = x.permute(0,1,3,2) # (B,N,T,F_hidden) -> (B,N,F_hidden,T)
        x = x.float()

        # print('x:', torch.max(torch.isnan(x)))

        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        temporal_At = self.TAt(x) # (b, T, T)

        # print('temporal_At:', torch.max(torch.isnan(temporal_At)))

        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        # x_TAt = torch.relu(x_TAt)

        # print('x_TAt:', torch.max(torch.isnan(x_TAt)))

        spatial_At = self.SAt(x_TAt)

        # print('spatial_At:', torch.max(torch.isnan(spatial_At)))

        spatial_gcn = self.cheb_conv_SAt(x_TAt, spatial_At) # (b,N,F,T)

        # spatial_gcn = torch.relu(spatial_gcn)

        # print('spatial_gcn:', torch.max(torch.isnan(spatial_gcn)))

        # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)->(b,N,F,T) 
        # time_conv_output = self.time_conv(spatial_gcn.double().permute(0, 2, 1, 3).float()).double().permute(0, 2, 1, 3) 

        final_output = spatial_gcn.permute(0,1,3,2)
        # final_output = time_conv_output.double().permute(0,1,3,2) # (b,N,F,T)-> (b,N,T,F) 

        

        return final_output


class ODEblock(nn.Module):

    '''
    input x: (B,N,T,F_hidden)

    Args:
    t: 1D tensot

    output:
    z: new hidden variable (B,N,T,F_hidden)
    '''

    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method='euler')[-1]    # output the result of last step
        return z

# Define the ODEGCN model.
class ODEG(nn.Module):
    '''
    description: graph neural ODE model

    input:
    x: graph signal (B,N,T,F_hidden)

    output:
    return: graph signal (B,N,T,F_hidden)
    '''
    def __init__(self, DEVICE, cheb_polys, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,\
                        num_of_vertices, num_of_timesteps, t_span):
        super(ODEG, self).__init__()
        self.odeblock = ODEblock(ODEFunc(DEVICE, cheb_polys, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_of_vertices, num_of_timesteps), t=t_span)

    def forward(self, x):

        self.odeblock.set_x0(x)
        z = self.odeblock(x)
        return z

class Net(nn.Module):

    def __init__(self, DEVICE, cheb_polys, in_channels, K, nb_chev_filter, hidden_feature_dim, time_strides,\
                       num_of_vertices, num_of_timesteps, t_span):
        super(Net, self).__init__()


        self.fc_encoder = nn.Sequential(nn.Linear(in_channels, 256), nn.ReLU(),  nn.Linear(256, hidden_feature_dim))
        self.fc_decoder = nn.Sequential(nn.Linear(3*hidden_feature_dim*num_of_timesteps, num_of_timesteps*64), nn.ReLU(), nn.Linear(num_of_timesteps*64, num_of_timesteps*64), nn.ReLU(), nn.Linear(num_of_timesteps*64, num_of_timesteps))
        self.GODE1 = ODEG(DEVICE, cheb_polys, hidden_feature_dim, K, nb_chev_filter, hidden_feature_dim, time_strides,\
                          num_of_vertices, num_of_timesteps, t_span)
        self.GODE2 = ODEG(DEVICE, cheb_polys, hidden_feature_dim, K, nb_chev_filter, hidden_feature_dim, time_strides,\
                          num_of_vertices, num_of_timesteps, t_span)
        self.GODE3 = ODEG(DEVICE, cheb_polys, hidden_feature_dim, K, nb_chev_filter, hidden_feature_dim, time_strides,\
                          num_of_vertices, num_of_timesteps, t_span)

    def forward(self, x):

        x = x.permute(0,1,3,2)  # (B,N,F,T) -> (B,N,T,F)

        # expand the dimension of the feature: (B,N,T,F)->(B,N,T,F_hidden)
        H1_1 = self.fc_encoder((x[:,:,0:12,:]))
        H1_2 = self.fc_encoder((x[:,:,12:24,:]))
        H1_3 = self.fc_encoder((x[:,:,24:36,:]))

        # forward of Neural ODE (B,N,T,F_hidden)
        H1_1 = self.GODE1(H1_1)
        H1_2 = self.GODE2(H1_2)
        H1_3 = self.GODE3(H1_3)

        # aggregrate the information of 3 ODE and apply maxpooling
        H1 = torch.cat((H1_1, H1_2, H1_3), -1)
        

        # H1 = torch.max(outs, dim=0)[0]

        # print('H1:', torch.max(torch.isnan(H1)))
        
        # (B,N,T,F_hidden) -> (B,N,T*F)
        H1 = H1.reshape((x.shape[0], x.shape[1], -1))
        # H1 = self.fc_decoder2(H1)

        # decoder: (B,N,T*F) -> (B,N,T)
        xprime = self.fc_decoder(H1)
        # x_recon = self.fc_decoder(H1_recon)

        return xprime
