import argparse
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torch.nn.functional as F

import os
import sys
import cifar_model
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import dataloader
import utility as utils

def set_data(opt):
    """Set up the dataloaders"""

    with open(opt['train_data_path'], 'rb') as f:
        train_array = pickle.load(f)

    mean = tuple(np.mean(train_array / 255., axis=(0, 1, 2)))
    std = tuple(np.std(train_array / 255., axis=(0, 1, 2)))
    normalize = transforms.Normalize(mean=mean, std=std)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_data = opt['dataloader'](opt['train_data_path'],
                                   opt['train_label_path'],
                                   transform_train)

    test_color_data = opt['dataloader'](opt['test_color_path'],
                                        opt['test_label_path'],
                                        transform_test)
    test_gray_data = opt['dataloader'](opt['test_gray_path'],
                                       opt['test_label_path'],
                                       transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=opt['batch_size'],
        shuffle=True, num_workers=opt['num_workers'])
    test_color_loader = torch.utils.data.DataLoader(
        test_color_data, batch_size=opt['batch_size'],
        shuffle=False, num_workers=opt['num_workers'])

    test_gray_loader = torch.utils.data.DataLoader(
        test_gray_data, batch_size=opt['batch_size'],
        shuffle=False, num_workers=opt['num_workers'])
    return train_loader,test_color_loader, test_gray_loader

def log_result(log_writer, name, result, step):
    log_writer.add_scalars(name, result, step)





def _test(loader):
    """Test the model performance"""

    model.eval()

    total = 0
    correct = 0
    test_loss = 0
    output_list = []
    feature_list = []
    predict_list = []
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images, targets = images.to(device), targets.to(device)
            outputs, features = model.forward(images)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            predict_list.extend(predicted.tolist())
            output_list.append(outputs.cpu().numpy())
            feature_list.append(features.cpu().numpy())

    test_result = {
        'accuracy': correct * 100. / total,
        'predict_labels': predict_list,
        'outputs': np.vstack(output_list),
        'features': np.vstack(feature_list)
    }
    return test_result


def test(name):
    # Test and save the result
    test_color_result = _test(test_color_loader)
    test_gray_result = _test(test_gray_loader)
    #utils.save_pkl(test_color_result, os.path.join(save_path, 'test_color_result.pkl'))
    #utils.save_pkl(test_gray_result, os.path.join(save_path, 'test_gray_result.pkl'))

    # Output the classification accuracy on test set
    info = ('Test on color images accuracy: {}\n'
            'Test on gray images accuracy: {}'.format(test_color_result['accuracy'],
                                                      test_gray_result['accuracy']))
    prefix = name[:name.find('.')]
    epoch_number=name[name.find('h')+1:name.find('.')]

    utils.write_info(os.path.join(opt["save_path"], f"{prefix}.txt"), info)
    if epoch_number.isnumeric():
        log_result(log_writer, 'test epoch', {'color':test_color_result['accuracy'], 'gray':test_gray_result['accuracy']}, epoch_number) 
if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_name',type=str, default='cifar_baseline')
  parser.add_argument('--no_cuda', dest='cuda', action='store_false')
  parser.add_argument('--random_seed', type=int, default=0)
  parser.set_defaults(cuda=True)
  parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')

  parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
  parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for dataloader')
  parser.add_argument('--frozen', action='store_true')

  opt = vars(parser.parse_args())
  opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
  if opt['frozen']:
      opt['save_path'] = 'test/frozen'
      path = 'results/frozen'
  else:
      opt['save_path'] = 'test/unfrozen'
      path = 'results/unfrozen'
  opt['print_freq'] = 50
  opt['batch_size'] = 128
  if not os.path.exists(opt['save_path']):
      os.makedirs(opt['save_path'])
  opt['dataloader'] = dataloader.CifarDataset
  opt['test_color_path'] = '../../../data/cifar_color_test_imgs'
  opt['test_gray_path'] = '../../../data/cifar_gray_test_imgs'
  opt['test_label_path'] = '../../../data/cifar_test_labels'
  opt['train_data_path'] = '../../../data/cifar-s/p95.0/train_imgs'
  opt['train_label_path']= '../../../data/cifar_train_labels'
  opt['num_classes'] = 10
  opt['criterion'] = F.cross_entropy
  log_writer = SummaryWriter(os.path.join(opt['save_path'], 'logfile'))

  for x in os.listdir(path):
      if x.endswith('.pth'):
        print(f"testing {x}")
        #save_path = os.path.join(opt['save_path'], x[:-4])
        model = cifar_model.CifarBase(os.path.join(path,x), testing=True)

        device = opt['device']
          #Setup model


        model = model.to(device)
        train_loader, test_color_loader, test_gray_loader  = set_data(opt)
        criterion = opt['criterion']
        print_freq = opt['print_freq']
        torch.cuda.empty_cache()

        test(x)
