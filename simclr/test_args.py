import os
import argparse
import torch
from models.cifar_base import cifar_model
import utils
from models import dataloader
import torch.nn.functional as F

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',
                        choices=[
                                 'cifar-s_baseline' ,
                            'cifar-s_domain_independent'
                            #Put other options later
                                ])
    parser.add_argument('--experiment_name',type=str, default='cifar-s_baseline')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.set_defaults(cuda=True)
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')

    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for dataloader')
    parser.add_argument('--model_path',type=str)

    opt = vars(parser.parse_args())
    return create_test_setting(opt)

def create_test_setting(opt):
    # common experiment setting
    if opt['experiment'].startswith('cifar'):
        opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
        opt['save_folder'] = os.path.join('test',
                                          opt['experiment_name'])
        opt['print_freq'] = 50
        opt['batch_size'] = 128
        utils.creat_folder(opt['save_folder'])
        opt['dataloader'] = dataloader.CifarDataset

    # experiment-specific setting
    ############ cifar color vs gray ##############
    if opt['experiment'] == 'cifar-s_baseline':
        opt['test_color_path'] = '../data/cifar_color_test_imgs'
        opt['test_gray_path'] = '../data/cifar_gray_test_imgs'
        opt['test_label_path'] = '../data/cifar_test_labels'
        opt['train_data_path'] = '../data/cifar-s/p95.0/train_imgs'
        opt['train_label_path']= '../data/cifar_train_labels'
        opt['num_classes'] = 10
        opt['criterion'] = F.cross_entropy

        model = cifar_model.CifarBase(opt['model_path'], testing=True)

    ###OTHER EXPERIMENTS HERE####


    return model, opt