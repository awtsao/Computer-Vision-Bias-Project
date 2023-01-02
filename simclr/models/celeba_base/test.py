import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from thop import profile, clever_format
from torch.utils.data import random_split, Subset
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader
import os
import sys
import numpy as np
from tensorboardX import SummaryWriter
import celeb_model
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import dataset
import simclr
import utility
import pickle



def _test(net, loader):
        """Compute model output on test set"""
        
        net.eval()

        output_list = []
        feature_list = []
        data_bar = tqdm(loader)
        with torch.no_grad():
            for images, targets in data_bar:
                images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                outputs, features = net(images)
                

                output_list.append(outputs)
                feature_list.append(features)

        return torch.cat(output_list), torch.cat(feature_list)


def inference(output):
        predict_prob = torch.sigmoid(output)
        return predict_prob.cpu().numpy()
def log_result(log_writer, name, result, step):
         log_writer.add_scalars(name, result, step)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name',type=str, default='celeb_baseline')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.set_defaults(cuda=True)

    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for dataloader')
    parser.add_argument('--frozen', action='store_true')
    opt = vars(parser.parse_args())
    print(opt['frozen'])
    print('testing_base')

    opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
    if opt['frozen']:
        opt['save_path'] = 'test/frozen'
        path = 'results/frozen'
    else:
        opt['save_path'] = 'test/unfrozen'
        path = 'results/unfrozen'
    opt['batch_size'] = 128
    if not os.path.exists('test'):
      os.makedirs('test')
    if not os.path.exists(opt['save_path']):
        os.makedirs(opt['save_path'])
    test_data_path = '../../../data/celeba/test_imgs'
    test_label_path = '../../../data/celeba/test_labels'
    
    with open(test_label_path, 'rb') as f:
      test_labels = pickle.load(f)
    
    test_class_weight = utility.compute_class_weight(test_labels)

    test_class_weight = utility.compute_class_weight(test_labels)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    log_writer = SummaryWriter(os.path.join(opt['save_path'], 'logfile'))

    test_data = dataset.CelebDataset(test_data_path, test_label_path, transform_test)
    test_loader = DataLoader(test_data, batch_size=opt['batch_size'], shuffle=False, num_workers=opt['num_workers'], pin_memory=True)
    subclass_idx = list(set(range(39)) - {0,16,21,29,37})
    print(f'path to models: {path}')
    for x in os.listdir(path):
        if x.endswith('.pth'):
            epoch_number=x[x.find('h')+1:x.find('.')]
            print(f"testing {x}")
            save_path = os.path.join(opt['save_path'], x[:-4])
            model = celeb_model.CelebBase(pretrained_path=os.path.join(path,x), testing=True)

            device = opt['device']
              #Setup model


            model = model.to(device)
            torch.cuda.empty_cache()
            test_output, test_feature = _test(model, test_loader)
            test_predict_prob = inference(test_output)
            test_per_class_AP = utility.compute_weighted_AP(test_labels, test_predict_prob, 
                                                      test_class_weight)
            test_mAP = utility.compute_mAP(test_per_class_AP, subclass_idx)
            test_result = f'per_class_AP: {test_per_class_AP}\nmAP: {test_mAP}'
            print(f'test mAP: {test_mAP}')
            utility.write_info(f'{save_path}_test_result.txt', test_result)
            if epoch_number.isnumeric():
              log_result(log_writer, 'test epoch', {'mAP': test_mAP}, epoch_number)


 
