
import numpy as np
import torch
import torch.nn as nn
import itertools
import pdb
import weightedstats as ws

'''
This file containts the implementation of the proposed architecture.
The Combiner class is the primary class with the other classes being
components of this model.
'''

class DeepSet(torch.nn.Module):
    '''
    implementation of deep set
    '''
    def __init__(self, ndim=1, hdim=16, sodim=16, odim=8, nlayers=3, snlayers=3):
        super(DeepSet, self).__init__()
        '''
        ndim: the dimension of each set element
        hdim: hidden dimension of the element processing network
        sodim: dimension of the set representation
        odim: output dimension
        nlayers: number of layers of element processor
        snlayers: number of layers in set processor
        '''
        
        self.ndim = ndim
        self.hdim = hdim
        self.sodim = sodim
        self.nlayers = nlayers
        self.snlayers = snlayers
        self.odim = odim

        nset_dim = sodim
        elm_mids = list(itertools.chain(
            *[[torch.nn.Linear(hdim, hdim), torch.nn.ELU(),] for i in range(self.nlayers - 2)]))
        self.elements = torch.nn.Sequential(
            torch.nn.Linear(ndim, hdim),
            torch.nn.ELU(),
            *elm_mids,
            torch.nn.Linear(hdim, sodim)
        )

        fc_mids = list(itertools.chain(
            *[[torch.nn.Linear(sodim, sodim), torch.nn.ELU(),] for i in range(self.snlayers - 2)]))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(nset_dim, sodim),
            torch.nn.ELU(),
            *fc_mids,
            torch.nn.Linear(sodim, odim)
        )
        
    def forward(self, x):
        n_sets = x.shape[0]
        set_size = x.shape[1]
        
        # store information about which inputs are nan
        pre_mask = torch.isnan(x)
        out = x.reshape([n_sets * set_size, 1])
        new_mask = ~torch.isnan(out)
        
        # Extract a representation of each set element
        new_out = torch.zeros([n_sets * set_size, self.sodim]).to(x.device)
        el_out = self.elements(out[new_mask][:, np.newaxis])
        new_out[torch.squeeze(new_mask)] += el_out
        out = new_out
        
        # Construct an initial vector representation for each set by averaging the representations of its elements
        mask = (~torch.isnan(out))
        out = out.reshape([n_sets, set_size, self.sodim])
        mask = mask.reshape([n_sets, set_size, self.sodim])
        counts = torch.sum( (~pre_mask)*1., dim=(1))
        counts[counts == 0] = 1
        sums = torch.sum(out, dim=1)
        avgs = sums / counts[:, np.newaxis]
        out = avgs
        mask = (torch.sum(mask, dim=(1, 2)) > 0) * 1.
        
        # Use fully connected net to compute higher level features from initial set representation
        out = self.fc(out)
        return torch.squeeze(out, dim=1), mask[:, np.newaxis]
    
class CNNResLayer(torch.nn.Module):
    '''
    Implements a single layer of a cnn with residual connections
    '''
    def __init__(self, ndim, odim, kernel_size, use_res=True, is_linear=False):
        '''
        ndim: number of input channels
        odim: number of output channels, in this application it will be 2 for the GP shape and scale parameters
        kernel_size: convolution kernel size
        use_res: whether or not to make this layer a residual layer or ordinary CNN layer
        is_linear: whether or not to make this just a simple convolutional layer with no activation function or residual connections
        '''
        super(CNNResLayer, self).__init__()
        
        # We have two cnn's in case this is a residual layer
        self.cnn1 = torch.nn.Conv2d(ndim, odim, kernel_size, padding=int((kernel_size - 1)/2))
        self.cnn2 = torch.nn.Conv2d(odim, odim, kernel_size, padding=int((kernel_size - 1)/2))
        if not (ndim == odim):
            self.skip = torch.nn.Conv2d(ndim, odim, 1)
        else:
            self.skip = torch.nn.Identity()
        self.act = torch.nn.ELU()
        self.ndim = ndim
        self.odim = odim
        self.kernel_size = kernel_size
        self.use_res = use_res
        self.is_linear = is_linear

    def forward(self, x):
        out = self.cnn1(x)
        if not self.is_linear:
            out = self.act(out)
            if self.use_res:
                out = self.cnn2(out)
                out += self.skip(x)
        return out
    
class CNN(torch.nn.Module):
    '''
    Standard CNN implementation
    '''
    def __init__(self, ndim, hdim=8, odim=3, nlayers=4, kernel_size=3, use_res=True):
        super(CNN, self).__init__()
        layers = [CNNResLayer(ndim, hdim, kernel_size, use_res=use_res)]
        layers += [CNNResLayer(hdim, hdim, kernel_size, use_res=use_res) for i in range(nlayers - 2)]
        layers += [CNNResLayer(hdim, odim, 1, use_res=False, is_linear=True)]
        self.convs = torch.nn.Sequential(*layers)
        self.ndim = ndim
        self.hdim = hdim
        self.odim = odim
        
    def forward(self, x):
        return self.convs(x)
    
class Constrainer(torch.nn.Module):
    '''
    Processes the output of a NN to make sure it satisfies the generalized Pareto constraints
    '''
    def __init__(self):
        super(Constrainer, self).__init__()
        
    def forward(self, all_ks, maxis=None):
        self.all_ks = all_ks

        # iterate through batches -- could be vectorized
        outs = list()
        for batch in range(all_ks.shape[0]):
            ks = all_ks[batch, :, :]
            k1, k2 = torch.exp(ks[:, 0]), torch.exp(ks[:, 1])
            sigma = k1[:, np.newaxis]
            k2 = k2[:, np.newaxis]
            xi = k2 - sigma / (maxis[batch, :, np.newaxis] + 1e-6)
            outs.append(torch.cat((xi, sigma), axis=1))
        return torch.stack(outs, dim=0)
        
class Combiner(torch.nn.Module):
    '''
    This is the proposed model
    '''
    def __init__(self, ds_parms=None, cnn_parms=None):
        '''
        ds_parms: deep set parameters
        cnn_parms: cnn parameters
        '''
        super(Combiner, self).__init__()

        # Create cnn
        self.cnn = CNN(**cnn_parms)
        
        # Create deep set parameters
        if not ds_parms is None:
            self.ds = DeepSet(**ds_parms)
        else:
            self.ds = None

        # create constrainer which will enforce GP constraints
        self.constrainer = Constrainer()
        
        # save cnn output dimension which will also be the overall model output dimension
        self.odim = cnn_parms['odim']
    
    def forward(self, vecs=None, sets=None, maxis=None):
        # Save shape information
        if not vecs is None:
            bsize, ndim_v, h, w = vecs.shape
        else:
            bsize, ndim_s, h, w = sets.shape
            
        # Compute set representations
        if not sets is None:
            ndim_s = sets.shape[1]
            sets = sets.reshape([bsize, ndim_s, h * w])
            sets = sets.permute([0, 2, 1])
            sets = sets.reshape([-1, ndim_s])
            set_out, mask = self.ds(sets)
            set_out = torch.cat([set_out, mask], dim=1)
            cur_feat_dim = set_out.shape[1]
            set_out = set_out.reshape([bsize, h * w, cur_feat_dim])
            set_out = set_out.permute([0, 2, 1])
            set_out = set_out.reshape([bsize, cur_feat_dim, h, w])
        
        # Concatenate vector predictors and set representations
        if not sets is None and not vecs is None:
            out = torch.cat([set_out, vecs], dim=1)
        elif not vecs is None:
            out = vecs
        else:
            out = set_out

        # Pass through CNN
        out = self.cnn(out)

        out_dim = out.shape[1]
        out = out.reshape([bsize, out_dim, h * w])
        out = out.permute([0, 2, 1])
        
        # Constrain output
        final_out = self.constrainer(out[:, :, :], maxis=maxis)  

        final_out = final_out.reshape(bsize, h, w, self.odim)
        return final_out