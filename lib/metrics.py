__all__ = ['mask_np', 'masked_mape_np', 'masked_rmse_np', 'masked_mae_np']

import numpy as np


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_rmse_np(y_true, y_pred, null_val=np.nan):
    'description'
    # y_true: (T,B,N)
    # y_pred: (T,B,N)

    mask = mask_np(y_true[:,:,0], null_val)
    mask /= mask.mean()
    
    time_steps = y_true.shape[2]
    error_each_steps = []
    for k in range(time_steps):
        err = (y_true[:,:,k] - y_pred[:,:,k]) ** 2
        err = np.sqrt(np.mean(np.nan_to_num(mask * err)))
        error_each_steps.append(err)

    # print('root mean square error of each steps:', error_each_steps)
    return np.array(error_each_steps)


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true[:,:,0], null_val)
    mask /= mask.mean()

    time_steps = y_true.shape[2]
    error_each_steps = []
    for k in range(time_steps):
        err = np.abs(y_true[:,:,k] - y_pred[:,:,k])
        err = np.mean(np.nan_to_num(mask * err))
        error_each_steps.append(err)

    # print('mean absolute error of each steps:', error_each_steps)

    return np.array(error_each_steps)