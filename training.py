import pickle
import numpy as np
import torch
import model
import os
import simple_utils as utils
from parameters import parms_dic

device = torch.device('cpu')

niter = 1500 # Number of training steps
total_runs = 1 # Number of train-test splits
eval_every = 50 # How frequently to evaluate on validation and test sets

torch.set_default_tensor_type(torch.DoubleTensor)

use_set, use_vec, use_pers = parms_dic['use_set'], parms_dic['use_vec'], parms_dic['use_pers']
def totensor(x):
    # Converts numpy array to tensor
    return torch.Tensor(x).to(device=device)

def make_max(x, training_maxis):
    # takes max training values as input and outputs a tensor with same shape as x
    maxis = np.repeat(training_maxis, x.shape[0], axis=0)
    maxis[np.isnan(maxis)] = 1
    maxis = torch.Tensor( maxis ).to(device=device)
    return maxis

for cur_run in range(total_runs):
    print('************************ run ************************')
    print(cur_run)
    print('************************     ************************')

    # Load Data
    fname = 'data/' + str(cur_run) + '.pickle'
    with open(fname, 'rb') as f:
        data_dic = pickle.load(f)
    x_train, g_train, y_train = data_dic['x_train'], data_dic['g_train'], data_dic['y_train']
    x_val, g_val, y_val = data_dic['x_val'], data_dic['g_val'], data_dic['y_val']
    x_test, g_test, y_test = data_dic['x_test'], data_dic['g_test'], data_dic['y_test']

    # Convert data to tensor
    x_train, g_train, y_train, x_val, g_val, y_val, x_test, g_test, y_test = totensor(x_train), totensor(g_train), totensor(y_train), totensor(x_val), totensor(g_val), totensor(y_val), totensor(x_test), totensor(g_test), totensor(y_test)

    # Create model and optimizer
    model = model.Combiner(ds_parms=parms_dic['ds_parms'], cnn_parms=parms_dic['cnn_parms']).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parms_dic['lr'])
    
    # Compute the maximum value that occurs at each location during the training period
    maxis_train_simple = np.nanmax(y_train.cpu(), axis=(0, 1))[np.newaxis]

    # Repeat these maximum values so that their shape matches excess value tensor shape for train, validation, test
    maxis_train = make_max(y_train, maxis_train_simple)
    maxis_val = make_max(y_val, maxis_train_simple)
    maxis_test = make_max(y_test, maxis_train_simple)

    best_llk = 999999
    for i in range(niter):
        optimizer.zero_grad()
        # Compute training loss
        loss, llk_loss, zero_prob, pred = utils.forward(model, x_train, g_train, y_train, maxis_train, use_vec, use_set, use_pers)
        
        # Periodically record validation and test loss
        if i % eval_every == 0:
            loss_val, llk_loss_val, zero_prob_val, pred_val = utils.forward(model, x_val, g_val, y_val, maxis_val, use_vec, use_set, use_pers)
            loss_test, llk_loss_test, zero_prob_test, pred_test = utils.forward(model, x_test, g_test, y_test, maxis_test, use_vec, use_set, use_pers)

        # Compute gradients and apply a step of Adam
        loss.backward()
        optimizer.step()

        # If this is the lowest validation loss encountered so far, then save predictions
        if (i % eval_every == 0) & (llk_loss_val < best_llk) & (zero_prob_val < 0.01):
            print('*******best run so far*******')
            best_llk = llk_loss_val
            best_pred = [pred, pred_val, pred_test]

        # If we computed validation and test loss then we should print all losses
        if i % eval_every == 0:
            print(i, 'train loss: ', loss.cpu().item(), 'val llk: ', llk_loss_val, 'test llk: ', llk_loss_test, zero_prob_val)