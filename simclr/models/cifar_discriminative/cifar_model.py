import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from thop import profile, clever_format
from torch.utils.data import random_split, Subset
from torch.utils.data.dataloader import DataLoader
import os
import sys
import numpy as np
import torch.nn.functional as F

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import dataset
import simclr
import utility


class CifarDiscriminative(nn.Module):
    def __init__(self, pretrained_path, testing):
        super(CifarDiscriminative, self).__init__()
        self.encoder = simclr.SimClr({"feature_dim": 128}).encoder
        self.fc = nn.Linear(2048, 20, bias=True)
        if testing:
          checkpoint = torch.load(pretrained_path, map_location='cpu')

          state_dict = []
          for n,p in checkpoint.items():
            if 'total_ops' not in n and 'total_params' not in n:
              state_dict.append((n,p))
            
          self.load_state_dict(dict(state_dict))
        else:
          self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=testing)

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out, feature


def finetune_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out, features = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum(
                (prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum(
                (prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/content/drive/MyDrive/6670/simclr/results/cifar-s_baseline/cifar-s_baseline_128_0.5_256_100_cifar.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=250, help='Number of sweeps over the dataset to train')
    parser.add_argument('--fine_tune', type=str, default='color')
    parser.add_argument('--split', type=float, default=0.9)
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--fc_lr', type=float, default=1e-3)

    args = parser.parse_args()
    model_path, batch_size, epochs, split, frozen = args.model_path, args.batch_size, args.epochs, args.split, args.frozen
    print(frozen)
    
    encoder_lr, fc_lr = args.encoder_lr, args.fc_lr
    if not frozen:
      print('NOT FROZEN')
      print(encoder_lr)
      print(fc_lr)
    print(model_path)
    
    color = 'cifar-s'
    finetune_data_path = "../../../data/cifar-s/p95.0/train_imgs"
    finetune_label_path = "../../../data/cifar-s/p95.0/train_2n_labels"


    print(finetune_data_path)
    print(epochs)
    ###RIGHT NOW NORMALIZE USING FINETUNED DATA, CAN SWITCH TO JUST THE ORIGINAL TRAINED 95.0 DATA like in the orignal paper"
    normalize = utility.getNormalize(finetune_data_path)

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

    # Split finetune_data into training and validate
    finetune_data = dataset.CifarDataset(finetune_data_path, finetune_label_path, transform_finetune)
    valid_data = dataset.CifarDataset(finetune_data_path, finetune_label_path, transform_validate)
    indices = list(range(len(finetune_data)))
    finetune_size = int(split * len(finetune_data))
    np.random.shuffle(indices)
    finetune_idx = indices[:finetune_size]
    valid_idx = indices[finetune_size:]
    finetune_dataset = Subset(finetune_data, indices=finetune_idx)
    valid_dataset = Subset(valid_data, indices=valid_idx)
    # finetune_ds, val_ds = random_split(finetune_data, [finetune_size, len(finetune_data)-finetune_size])

    finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    

    
    model = CifarDiscriminative(pretrained_path=model_path, testing=False).cuda()
    print('===================MODEL ENCODER=================')
    print(model.encoder.parameters())
    print('=====================FC================')
    print(model.fc.parameters())
    if frozen:
      for param in model.encoder.parameters():
        param.requires_grad = False
      optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
      save_path = 'results/frozen'
    else:
      optimizer = optim.Adam([
        {'params':model.encoder.parameters(), 'lr':encoder_lr}, 
        {'params':model.fc.parameters(), 'lr':fc_lr},

      ],
      lr=1e-3,
      weight_decay=1e-6
      )
      save_path = f'results/unfrozen'
    print(optimizer)

    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))

    loss_criterion = F.cross_entropy
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    best_acc = 0.0
    if not os.path.exists('results'):
      os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = finetune_val(model, finetune_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = finetune_val(model, val_loader, None)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(os.path.join(save_path,'linear_statistics.csv'), index_label='epoch')
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch{epoch}.pth'))
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), os.path.join(save_path, 'linear_model.pth'))
    torch.save(model.state_dict(),os.path.join(save_path, 'final_model.pth'))