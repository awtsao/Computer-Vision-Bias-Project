import os
import argparse
import torch
from models import simclr as base 
from models import dataset
import utils


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',
                        choices=[
                                 'cifar-s_baseline',
                                 'cifar-color',
                                'celeba'
                            #Put other options later
                                ])

    parser.add_argument('--experiment_name', type=str, default='cifar-s')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.set_defaults(cuda=True)
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for dataloader')
    parser.add_argument('--save',action='store_true')

    opt = vars(parser.parse_args())
    return create_train_setting(opt)

def create_train_setting(opt):
    # common experiment setting
    if opt['experiment'].startswith('cifar'):
        opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
        opt['save_folder'] = os.path.join('record/'+opt['experiment'],
                                          opt['experiment_name'])
        utils.creat_folder(opt['save_folder'])
        optimizer_setting = {
            'optimizer': torch.optim.Adam,
            'lr': 1e-3,
            'weight_decay': 1e-6,
        }
        opt['optimizer_setting'] = optimizer_setting
        opt['dataset'] = dataset.CifarTrainPair
    elif opt['experiment'].startswith('celeb'):
        opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
        opt['save_folder'] = os.path.join('record/' + opt['experiment'],
                                          opt['experiment_name'])
        utils.creat_folder(opt['save_folder'])
        optimizer_setting = {
            'optimizer': torch.optim.Adam,
            'lr': 1e-3,
            'weight_decay': 1e-6,
        }
        opt['optimizer_setting'] = optimizer_setting
        opt['dataset'] = dataset.CelebTrainPair
    # experiment-specific setting
    ############ cifar color vs gray ##############
    if opt['experiment'] == 'cifar-s_baseline':

        opt['train_data_path'] = '../data/cifar-s/p95.0/train_imgs'
    elif opt['experiment'] == 'cifar-color':
        opt['train_data_path'] = '../data/cifar_color_train_imgs'
    elif opt['experiment']=='celeba':
        opt['train_data_path'] = '../data/celeba/train_imgs'
    model = base.SimClr(opt)

    ###OTHER EXPERIMENTS HERE####


    return model, opt