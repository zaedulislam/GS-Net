from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import time
import logging
import torch
import torch.utils.data
import torch.nn as nn
import time
import numpy as np
import torch.optim as optim

from tokenize import String
from arguments import opts
from six.moves import xrange  # pylint: disable=redefined-builtin
from utils.data_utils import define_actions
from utils.utils1 import save_model
from nets.post_refine import post_refine
from train_graph_time import train, val
from data.common.data_utils import read_3d_data
from data.common.graph_utils import adj_mx_from_skeleton
from data.load_data_hm36 import Fusion  # Data fusion to prepare data 
from models.gsnet_gcn import GSNetGCN


model = {} # Model list 
opt = opts().parse() # Import args    

lr = opt.learning_rate
opt.manualSeed = 1
print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

try:
    os.makedirs(opt.save_dir)
except OSError:
    pass

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(opt.save_dir, 'train_test.log'), level=logging.INFO)
logging.info('======================================================')

# 1. Load data
root_path = opt.root_path
if opt.dataset == 'h36m':
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'
    from data.common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path, opt)
else:
    raise KeyError('Invalid dataset')

actions = define_actions(opt.actions)

p_dropout = (None if opt.dropout == 0.0 else opt.dropout)

# Adjacency Matrix
adj = adj_mx_from_skeleton(dataset.skeleton())

# 2. Load model
model['gsnet_gcn'] = GSNetGCN(adj, opt.hid_dim, beta = opt.beta, num_layers=opt.num_layers, p_dropout=p_dropout).cuda()
model['post_refine'] = post_refine(opt).cuda()

if opt.pro_train:
    train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
                                                   shuffle=True, num_workers=int(opt.workers), pin_memory=False)
if opt.pro_test:
    test_data = Fusion(opt=opt, train=False,dataset=dataset, root_path =root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=False)

# 3. Set optimizer
total_param=0
all_param = []

for i_model in model:
    all_param += list(model[i_model].parameters())
    total_param += sum(p.numel() for p in model[i_model].parameters())
if opt.optimizer == 'SGD':
    optimizer_all = optim.SGD(all_param, lr=opt.learning_rate, momentum=0.9, nesterov=True, weight_decay=opt.weight_decay)
elif opt.optimizer == 'Adam':
    optimizer_all = optim.Adam(all_param, lr=lr, amsgrad=True)

optimizer_all_scheduler = optim.lr_scheduler.StepLR(optimizer_all, step_size=5, gamma=0.1)

# Print parameters 
print("==> Total parameters: {:.2f}M".format(total_param / 1000000.0))

# 4. Reload model
gsnet_gcn_dict = model['gsnet_gcn'].state_dict()

if opt.gsnet_gcn_reload == 1: 
    
    pre_dict_gsnet_gcn = torch.load(os.path.join(opt.previous_dir, opt.module_gsnet_model))
    # For name, key in stgcn_dict.items():
    for name, key in gsnet_gcn_dict.items(): 
        if name.startswith('A') == False:
           
            gsnet_gcn_dict[name] = pre_dict_gsnet_gcn[name]
    
    model['gsnet_gcn'].load_state_dict(gsnet_gcn_dict)

post_refine_dict = model['post_refine'].state_dict()

if opt.post_refine_reload == 1:
    pre_dict_post_refine = torch.load(os.path.join(opt.previous_dir, opt.post_refine_model))

    for name, key in post_refine_dict.items():
        post_refine_dict[name] = pre_dict_post_refine[name]

    model['post_refine'].load_state_dict(post_refine_dict)


# 5.Set criterion
criterion = {}
# L2 loss
criterion['MSE'] = nn.MSELoss(reduction='mean').cuda()
# L1 loss
criterion['L1'] = nn.L1Loss(reduction='mean').cuda()


# Training process
training_start_time = time.time()

for epoch in range(1, opt.nepoch):
    print('======>>>>> Online epoch: #%d <<<<<======' % (epoch))
    torch.cuda.synchronize()

    # Switch to train
    if opt.pro_train == 1:
        timer = time.time()
        print('======>>>>> Train <<<<<======')
        print('Frame number: %d' %(opt.pad*2+1))
        print('Processing file %s:' %opt.model_doc)
        print('Learning rate %f' % (lr))
        mean_error = train(opt, actions, train_dataloader, model, criterion, optimizer_all)
        timer = time.time() - timer
        timer = timer / len(train_data)
        print('==> Time to learn 1 sample = %f (ms)' % (timer * 1000))

    # Switch to test
    if opt.pro_test == 1:
        timer = time.time()
        print('======>>>>> Test <<<<<======')
        print('Frame number: %d' %(opt.pad*2+1))
        print('Processing file %s:' %opt.model_doc)
        mean_error = val(opt, actions, test_dataloader, model, criterion)
        timer = time.time() - timer
        timer = timer / len(test_data)
        print('==> Time to learn 1 sample = %f (ms)' % (timer * 1000))

        if opt.save_out_type == 'xyz':
            data_threshold = mean_error['xyz']

        elif opt.save_out_type == 'post':
            data_threshold = mean_error['post']

        if opt.save_model and data_threshold < opt.previous_best_threshold:
            opt.previous_gsnet_gcn_name = save_model(opt.previous_gsnet_gcn_name, opt.save_dir, epoch, opt.save_out_type, data_threshold, model['gsnet_gcn'], 'gsnet_gcn')

            if opt.post_refine:
                opt.previous_post_refine_name = save_model(opt.previous_post_refine_name, opt.save_dir, epoch, opt.save_out_type,
                                                      data_threshold, model['post_refine'], 'post_refine')
                
            opt.previous_best_threshold = data_threshold

    if epoch % opt.large_decay_epoch == 0:
        for param_group in optimizer_all.param_groups:
            param_group['lr'] *= opt.lr_decay
            lr *= opt.lr_decay


# Training/testing time calculation
training_end_time = time.time()
total_training_time_in_seconds = training_end_time - training_start_time

hours = "{:.0f}".format(total_training_time_in_seconds // 3600)
seconds = total_training_time_in_seconds % 3600

minutes = "{:.0f}".format(seconds // 60)
seconds = "{:.2f}".format(seconds % 60)

trainOrTestText: String = "Training" if opt.pro_train==1 else "Testing"
print("Total " + trainOrTestText + " time: " + hours + " hour(s), " + minutes + " minute(s), " + seconds + " second(s)")
