
import os
import datetime
import numpy as np
import pickle
import tensorboard_logger as tb_logger

import torch
from torch.utils import data
from torchvision import transforms, utils
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import torch.backends.cudnn as cudnn
from torchvision import utils
import sys
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss
from utils import adjust_learning_rate, pil_loader
from model import SalEncoder, Encoder, Normalize
from loader import ImageData, Projection
import argparse
# from utils

try:
    from apex import amp, optimizers
except ImportError:
    # print
    pass

def parse_options():
    parser = argparse.ArgumentParser('argument for training')

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1000, help='print frequency in steps')
    parser.add_argument('--tb_freq', type=int, default=50, help='tb frequency in steps')
    parser.add_argument('--save_freq', type=int, default=1, help='save frequency in epoch')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=250, help='number of training epochs')

    # optimization
    parser.add_argument('--optimizer_type', type=str, default='Adam', choices=['SGD', 'Adam'])
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=16000, help='# negative samples')
    parser.add_argument('--nce_t', type=float, default=0.07, help='temperature parameters')
    parser.add_argument('--nce_m', type=float, default=0.5, help='momentum for updates in memory bank')
    parser.add_argument('--feat_dim', type=int, default=1024, help='dim of feat for inner product')

    # dataset
    parser.add_argument('--dataset', type=str, default='Train')

    # specify folder
    parser.add_argument('--data_folder', type=str, default="data", help='path to data')
    parser.add_argument('--model_path', type=str, default="models", help='path to save model')
    parser.add_argument('--tb_path', type=str, default="runs", help='path to tensorboard')


    # mixed precision setting
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])

    # data crop threshold
    # parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    opt = parser.parse_args()

    if (opt.data_folder is None) or (opt.model_path is None) or (opt.tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'softmax' if opt.softmax else 'nce'
    opt.model_name = 'memory_{}_{}_lr_{}_decay_{}_bsz_{}'.format(opt.method, opt.nce_k, opt.learning_rate,
                                                                    opt.weight_decay, opt.batch_size)

    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    if not os.path.isdir(opt.data_folder):
        raise ValueError('data path not exist: {}'.format(opt.data_folder))

    return opt


def get_data_loader(args):

    #path to image folders
    data_folder = os.path.join(args.data_folder, 'Train', 'image')

    means = [0.485, 0.456, 0.406]
    stds =  [0.229, 0.224, 0.225]
    
    normalize = transforms.Normalize(mean=means, std=stds)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_dataset = ImageData(pil_loader, 
                             data_folder, 
                             transform=train_transform)
    train_sampler = None

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    # num of samples
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data
#------------------------------------------------------------------

def init_model(args, n_data):
    #set device
    #need to change in case of multiple-GPUS
    #=====================================================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    #-----------remove this--------
    #device = torch.device('cpu')

    #set model and NCE criterion
    net =  Encoder(device, args.feat_dim)
    contrast = NCEAverage(args.feat_dim,\
                            n_data, args.nce_k,\
                            args.nce_t, args.nce_m,\
                            args.softmax)

    nce_l1 = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    nce_l2 = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    #-------------------------
    net = net.to(device)
    contrast = contrast.to(device)
    nce_l1 = nce_l1.to(device)
    nce_l2 = nce_l2.to(device)
    
    if device !='cpu':
        cudnn.benchmark = True
    #======================================================================    

    return net, contrast, nce_l1, nce_l2, device
#-------------------------------------------------------------------------
def init_optimizer(args, net):
    #set optimizer 

    if args.optimizer_type =='SGD':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    
    if args.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=args.learning_rate, 
                                     betas=(args.beta1, args.beta2),
                                     weight_decay=args.weight_decay)    

    return optimizer

#-----------------------------------------------------------------------
def train(train_loader, net, contrast, nce_l1, nce_l2, optimizer, args):
    
    net.train()
    contrast.train()
    Transf_ = Projection()

    steps = args.start_steps

    for epoch in range(args.start_epoch, args.epochs+1):

        #adjust learning rate
        adjust_learning_rate(epoch, args, optimizer)

        t1 = time.time()

        for idx, (batch_data, index) in enumerate(train_loader):

            net.zero_grad()
            batch_size = batch_data.size()[0]

            batch_data = batch_data.to(args.device)
            index = index.to(args.device).long()
            
            #Projection 
            tf1, tf2 = Transf_(batch_data)
            
            
            #-------forward---------------
            
            feat1, feat2, feat3 = net(batch_data, tf1, tf2)
            
            #--------loss-----------------
            index.cuda()
            l1, l2 = contrast(feat1, feat2, feat3, index)
            nce_1, nce_2 = nce_l1(l1), nce_l2(l2)

            #----------batch_loss---------
            loss = nce_1 + nce_2

            #-----------backward----------
            optimizer.zero_grad()

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            optimizer.step()

            steps+=1
            #-------------print and tensorboard logs ------------------
            if (steps%args.print_freq==0):
                print('Epoch: {}/{} || steps: {}/{} || total loss: {:.3f} || loss_nce1: {:.3f} || loss_nce2: {:.3f}'\
                    .format(epoch, args.epochs, steps, args.total_steps,
                    loss.item(), nce_1.item(), nce_2.item()))
                sys.stdout.flush()
            #logs
            if (steps%args.tb_freq==0):
                args.logger.log_value('total_loss', loss.item(), steps)
                args.logger.log_value('loss_nce1', nce_1.item(), steps)
                args.logger.log_value('loss_nce2', nce_2.item(), steps)

            
           #------------------------------------------------------------- 
        t2 = time.time()
        print('epoch {}, total time {:.2f}s'.format(epoch, (t2 - t1)))

        #-------------------save model (after every epoch)-------------------------------   
        if(epoch%args.save_freq==0):
            print('==> Saving...')
            state = {
                'model': net.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'steps':steps,
            }
            if args.amp:
                state['amp'] = amp.state_dict()

            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

            # help release GPU memory
            del state
        torch.cuda.empty_cache()


def main():

    #set parser
    args = parse_options()
    #data loader
    
    train_loader, n_data = get_data_loader(args)
   
       
    args.total_steps = args.epochs*int(n_data/args.batch_size)\
                      if (n_data%args.batch_size)==0\
                      else args.epochs*(int(n_data/args.batch_size)+1)  

    #get model
    net, contrast, nce_l1, nce_l2, args.device = init_model(args,n_data)

    #get optimizer
    optimizer = init_optimizer(args, net)

    #mixed_precsion
    if args.amp:
        model, optimizer = amp.initialize(net, optimizer, opt_level=args.opt_level)

    #checkpoint
    args.start_epoch = 1
    args.start_steps = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            args.start_steps  = checkpoint['steps'] + 1
            net.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            if args.amp and checkpoint['opt'].amp:
                print('==> resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #tensorboard

    args.logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    
    #start the training loop
    train(train_loader, net, contrast, nce_l1, nce_l2, optimizer, args)
    
if __name__=='__main__':
    main()