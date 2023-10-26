This is implementation of ASTGODE of the paper: Attention-based Spatial-Temporal Graph Neural ODE for Traffic Prediction( https://arxiv.org/abs/2305.00985 ), which was accepted by 25th IEEE Intelligent Transportation Systems Conference (ITSC 2022).

# Data preparation

Download the dataset from goolge drive: https://drive.google.com/drive/folders/1CtFU_peMpiu0u3w9NhxZfRLyBQsb-kcz. Put them in the folder named "data".

# Experiment

To reproduce the results of this work, you can simply run main.py. If you are interested in model training without adjoint sensitivity method, simply run:
```
python main.py --case='PEMS04' --adjoint=1
python main.py --case='PEMS_bay' --adjoint=1
```

If you are interested in model training with adjoint sensitivity method, simply run:
```
python main.py --case='PEMS04' --adjoint=0
python main.py --case='PEMS_bay' --adjoint=0
```

If you make advantage of the ASTGODE in your research, please consider citing our paper in your manuscript:
```
@misc{zhong2023attentionbased,
      title={Attention-based Spatial-Temporal Graph Neural ODE for Traffic Prediction}, 
      author={Weiheng Zhong and Hadi Meidani and Jane Macfarlane},
      year={2023},
      eprint={2305.00985},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
