This is implementation of ASTGODE of the paper: Attention-based Spatial-Temporal Graph Neural ODE for Traffic Prediction( https://arxiv.org/abs/2305.00985 ), which was accepted by 25th IEEE Intelligent Transportation Systems Conference (ITSC 2022).

# Data preparation

Download the dataset from goolge drive: https://drive.google.com/drive/folders/1CtFU_peMpiu0u3w9NhxZfRLyBQsb-kcz. Put them in the folder named "data".

# Experiment

To reproduce the results of this work, you can simply run main.py. If you are interested in model training without adjoint sensitivity method, simply run:
'''
python main.py --case='PEMS04' --adjoint=1
python main.py --case='PEMS_bay' --adjoint=1
'''

If you are interested in model training with adjoint sensitivity method, simply run:

'''
python main.py --case='PEMS04' --adjoint=1
python main.py --case='PEMS_bay' --adjoint=1
'''

Please consider cite our paper if our codes help
