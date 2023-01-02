import train_argparse
from tqdm import tqdm
import torch
# from pytorch_metric_learning.losses import NTXentLoss
import pandas as pd
import pickle
import numpy as np
import torchvision.transforms as transforms
import os 
import sys
def trainEpoch(net, data_loader, train_optimizer, epoch_num, opt, device):
    # https://github.com/leftthomas/SimCLR/blob/master/main.py
    batch_size = opt['batch_size']
    temperature = opt['temperature']
    epochs = opt['epochs']
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    for pos_1, pos_2 in train_bar:
        pos_1, pos_2 = pos_1.to(device, non_blocking=True), pos_2.to(device, non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch_num, epochs, total_loss / total_num))

    return total_loss / total_num


def getDataLoaderCifar(opt):
  print('USING CIFAR DATALOADER')
  with open(opt['train_data_path'], 'rb') as f:
      train_array = pickle.load(f)
  
  mean = tuple(np.mean(train_array / 255., axis=(0, 1, 2)))
  std = tuple(np.std(train_array / 255., axis=(0, 1, 2)))
  normalize = transforms.Normalize(mean=mean, std=std)


  transform_train = transforms.Compose([
      transforms.RandomResizedCrop(32),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)],p=0.8),
      transforms.RandomGrayscale(p=0.2),
      transforms.ToTensor(),
      normalize])


  train_data = opt['dataset'](opt['train_data_path'],transform_train)

  return torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'],
      shuffle=True, num_workers=opt['num_workers'], pin_memory=True, drop_last=True)


def getDataLoaderCeleb(opt):
    print('USING CELEBA DATALOADER')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize])

    train_data = opt['dataset'](opt['train_data_path'], transform_train)

    return torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'],
                                       shuffle=True, num_workers=opt['num_workers'], pin_memory=True, drop_last=True)


if __name__ == '__main__':
    
    model, opt = train_argparse.parse()

    device = opt['device']
    model = model.to(device)
    print(opt)
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(os.path.join('results', opt['experiment'])):
        os.mkdir(os.path.join('results', opt['experiment']))
    #Dataloader
    torch.cuda.empty_cache()
    if opt['experiment'].startswith('cifar'):
        train_loader = getDataLoaderCifar(opt)
    else:
        train_loader = getDataLoaderCeleb(opt)


    opt_settings = opt['optimizer_setting']
    results = {'train_loss': []}
    experiment = opt['experiment']
    feature_dim, temperature, batch_size, epochs = opt['feature_dim'], opt['temperature'], opt['batch_size'], opt[
        'epochs']

    save_name_pre = '{}_{}_{}_{}_{}'.format(experiment, feature_dim, temperature, batch_size, epochs)
    optimizer = opt_settings['optimizer'](params=model.parameters(), lr=opt_settings['lr'],
                                          weight_decay=opt_settings['weight_decay'])
    for epoch in range(opt['epochs']):
        train_loss = trainEpoch(model, train_loader, optimizer, epoch, opt, device)
        results['train_loss'].append(train_loss)
        # test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        # results['test_acc@1'].append(test_acc_1)
        # results['test_acc@5'].append(test_acc_5)
        # save statistics
        # data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        # data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if opt['save']:
          if epoch % 10 == 0:
              torch.save(model.state_dict(), f'./results/{opt["experiment"]}/{save_name_pre}_epoch{epoch}.pth')
      
        # Need to save stats later
    if opt['save']:
      torch.save(model.state_dict(), f'./results/{opt["experiment"]}/{save_name_pre}_cifar.pth')
