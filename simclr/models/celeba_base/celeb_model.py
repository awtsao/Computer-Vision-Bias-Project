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
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import dataset
import simclr
import utility
import pickle

class CelebBase(nn.Module):
    def __init__(self, pretrained_path, testing):
        super(CelebBase, self).__init__()
        self.encoder = simclr.SimClr({"feature_dim": 128}).encoder
        self.fc = nn.Linear(2048, 39, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        if testing:
          checkpoint = torch.load(pretrained_path, map_location='cpu')

          state_dict = []
          for n,p in checkpoint.items():
            if 'total_ops' not in n and 'total_params' not in n:
              state_dict.append((n,p))
            
          self.load_state_dict(dict(state_dict))
        else:
          print('here')
          self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=testing)

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(self.dropout(self.relu(feature)))
        return out, feature


def _train(net, data_loader, train_optimizer):
    net.train()

    total_loss, total_num, data_bar = 0.0,  0, tqdm(data_loader)
    with (torch.enable_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out, features = net(data)
            loss = criterion(out, target)

            
            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} '
                                     .format('Train' , epoch, epochs, total_loss / total_num))

    return total_loss / total_num
def _test(net, loader):
        """Compute model output on test set"""
        
        net.eval()

        test_loss = 0
        output_list = []
        feature_list = []
        data_bar = tqdm(loader)
        with torch.no_grad():
            for images, targets in data_bar:
                images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                outputs, features = net(images)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                output_list.append(outputs)
                feature_list.append(features)

        return test_loss, torch.cat(output_list), torch.cat(feature_list)

def criterion(output, target):
        return F.binary_cross_entropy_with_logits(output, target[:, :-1])
def inference(output):
        predict_prob = torch.sigmoid(output)
        return predict_prob.cpu().numpy()
def log_result(log_writer, name, result, step):
         log_writer.add_scalars(name, result, step)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='results/128_0.5_200_512_500_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--split', type=float, default=0.9)
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--encoder_lr', type=float, default=5e-5)
    parser.add_argument('--fc_lr', type=float, default=1e-3)

    args = parser.parse_args()
    model_path, batch_size, epochs, split, frozen = args.model_path, args.batch_size, args.epochs, args.split, args.frozen
    print(frozen)
    encoder_lr, fc_lr = args.encoder_lr, args.fc_lr

    print(model_path)
    if not os.path.exists('results'):
        os.mkdir('results')
    finetune_data_path = "../../../data/celeba/train_imgs"
    finetune_label_path = "../../../data/celeba/train_labels"

    val_data_path = '../../../data/celeba/val_imgs'
    val_label_path = '../../../data/celeba/val_labels'
    
    with open(val_label_path, 'rb') as f:
      val_labels = pickle.load(f)
    
    val_class_weight = utility.compute_class_weight(val_labels)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_finetune = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize])

    transform_validate = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    finetune_data = dataset.CelebDataset(finetune_data_path, finetune_label_path, transform_finetune)
    valid_data = dataset.CelebDataset(val_data_path, val_label_path, transform_validate)
   

    finetune_loader = DataLoader(finetune_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    subclass_idx = list(set(range(39)) - {0,16,21,29,37})
    

    
    model = CelebBase(pretrained_path=model_path, testing=False).cuda()
    if frozen:
      for param in model.encoder.parameters():
        param.requires_grad = False
      optimizer =    optimizer = optim.Adam([
        {'params':model.fc.parameters(), 'lr':fc_lr},
        {'params':model.relu.parameters(), 'lr':fc_lr},
        {'params':model.dropout.parameters(),'lr':fc_lr}

      ],
      lr=1e-3,
      weight_decay=1e-6
      )
      save_path = 'results/frozen'
        
    else:

      optimizer = optim.Adam([
        {'params':model.encoder.parameters(), 'lr':encoder_lr}, 
        {'params':model.fc.parameters(), 'lr':fc_lr},
        {'params':model.relu.parameters(), 'lr':fc_lr},
        {'params':model.dropout.parameters(),'lr':fc_lr}

      ],
      lr=1e-3,
      weight_decay=1e-6
      )
      save_path = f'results/unfrozen'

    log_writer = SummaryWriter(os.path.join(save_path, 'logfile'))
    print(optimizer)
    results = {'train_loss': [],'test_loss': [], 'val_output':[],
    'val_feature':[], 'val_per_class_AP':[], 'val_mAP':[]}
    print(f'epochs:{epochs}')
    print(save_path)
    best_val_mAP = 0
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for epoch in range(1, epochs + 1):
        torch.cuda.empty_cache()
        train_loss= _train(model, finetune_loader, optimizer)
        #results['train_loss'].append(train_loss)
        
        if epoch % 10 == 0:
          val_loss, val_output, val_feature = _test(model, val_loader)
          val_predict_prob = inference(val_output)
          val_per_class_AP = utility.compute_weighted_AP(val_labels, val_predict_prob, 
                                                      val_class_weight)
          val_mAP = utility.compute_mAP(val_per_class_AP, subclass_idx)
          #results['val_output'].append(val_output.cpu().numpy())
          #results['val_feature'].append(val_feature.cpu().numpy())
          #results['val_per_class_AP'].append(val_per_class_AP)
          #results['val_mAP'].append(val_mAP)
          #print(f'Epoch {epoch}: {results}')
          log_result(log_writer, 'Val epoch', {'loss': val_loss/len(val_loader), 'mAP': val_mAP}, epoch)
          print(f"mAP: {val_mAP}")
          torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch{epoch}.pth'))
          if val_mAP > best_val_mAP:
              best_val_mAP = val_mAP
              torch.save(model.state_dict(), os.path.join(save_path, 'linear_model.pth'))
    torch.save(model.state_dict(),os.path.join(save_path, 'final_model.pth'))