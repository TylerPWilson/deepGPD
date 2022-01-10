parms_dic = {
    'use_pers': True, # Wether or not to use persistence features (i.e. GP parameters from previous time step)
    'use_vec': True, # Wether or not to use vector features
    'use_set': True, # Wether or not to use set features
    'lr': 0.0001, # learning rate
    # Parameters for deep set
    'ds_parms':{
        'odim': 12, # output dimension
        'hdim': 4, # hidden dimension of FCN that processes set elements
        'sodim': 8, # dimension of set representation
        'nlayers': 7, # number of layers for processing elements
        'snlayers': 3, # number of layers for processing set representation
    },
    'cnn_parms': {
        'ndim': 41, # Input dimension to CNN = Deep Set Odim + # Persistence Features + # Vector Features 
        'hdim': 50, # hidden dimension
        'odim': 2, # output dimension -- 1 dimension for GP shape and 1 for scale
        'nlayers': 7, # number of layers
        'kernel_size': 3, # Kernel size
        'use_res': True # Use residual connections
    }
}