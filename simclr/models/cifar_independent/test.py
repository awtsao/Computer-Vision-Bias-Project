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



def criterion(output, target):
    class_num = output.size(1) // 2
    logprob_first_half = F.log_softmax(output[:, :class_num], dim=1)
    logprob_second_half = F.log_softmax(output[:, class_num:], dim=1)
    output = torch.cat((logprob_first_half, logprob_second_half), dim=1)
    return F.nll_loss(output, target)
    
def _test(loader, test_on_color=True):
    """Test the model performance"""
    
    model.eval()

    total = 0
    correct = 0
    test_loss = 0
    output_list = []
    feature_list = []
    target_list = []
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images, targets = images.to(device), targets.to(device)
            outputs, features = model.forward(images)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            output_list.append(outputs)
            feature_list.append(features)
            target_list.append(targets)
            
    outputs = torch.cat(output_list, dim=0)
    features = torch.cat(feature_list, dim=0)
    targets = torch.cat(target_list, dim=0)
    
    accuracy_conditional = compute_accuracy_conditional(outputs, targets, test_on_color)
    accuracy_sum_out =compute_accuracy_sum_out(outputs, targets)
    
    test_result = {
        'accuracy_conditional': accuracy_conditional,
        'accuracy_sum_out': accuracy_sum_out,
        'outputs': outputs.cpu().numpy(),
        'features': features.cpu().numpy()
    }
    return test_result

def compute_accuracy_conditional(outputs, targets, test_on_color):
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()
    
    class_num = outputs.shape[1] // 2
    if test_on_color:
        outputs = outputs[:, :class_num]
    else:
        outputs = outputs[:, class_num:]
    predictions = np.argmax(outputs, axis=1)
    accuracy = (predictions == targets).mean() * 100.
    return accuracy

def compute_accuracy_sum_out(outputs, targets):
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()
    
    class_num = outputs.shape[1] // 2
    predictions = np.argmax(outputs[:, :class_num] + outputs[:, class_num:], axis=1)
    accuracy = (predictions == targets).mean() * 100.
    return accuracy

def test(name):
    # Test and save the result

    test_color_result = _test(test_color_loader, test_on_color=True)
    test_gray_result =_test(test_gray_loader, test_on_color=False)
    #utils.save_pkl(test_color_result, os.path.join(save_path, 'test_color_result.pkl'))
    #utils.save_pkl(test_gray_result, os.path.join(save_path, 'test_gray_result.pkl'))
    
    # Output the classification accuracy on test set for different inference
    # methods
    info = ('Test on color images accuracy conditional: {}\n' 
            'Test on color images accuracy sum out: {}\n'
            'Test on gray images accuracy conditional: {}\n'
            'Test on gray images accuracy sum out: {}\n'
            .format(test_color_result['accuracy_conditional'],
                    test_color_result['accuracy_sum_out'],
                    test_gray_result['accuracy_conditional'],
                    test_gray_result['accuracy_sum_out']))

    prefix = name[:name.find('.')]
    epoch_number=name[name.find('h')+1:name.find('.')]

    utils.write_info(os.path.join(opt["save_path"], f"{prefix}.txt"), info)
    if epoch_number.isnumeric():
        log_result(log_writer, 'test epoch', {'color conditional':test_color_result['accuracy_conditional'], 'gray conditional':test_gray_result['accuracy_conditional'],\
       + 'color sum out':test_color_result['accuracy_sum_out'], 'gray sum out':test_gray_result['accuracy_sum_out']}, epoch_number) 

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_name',type=str, default='cifar_independent')
  parser.add_argument('--no_cuda', dest='cuda', action='store_false')
  parser.add_argument('--random_seed', type=int, default=0)
  parser.set_defaults(cuda=True)
  parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')

  parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
  parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for dataloader')
  parser.add_argument('--model_path',type=str)
  parser.add_argument('--frozen', action='store_true')

  opt = vars(parser.parse_args())
  opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
  if opt['frozen']:
        opt['save_path'] = 'test/frozen'
        path = 'results/frozen'
  else:
        opt['save_path'] = 'test/unfrozen'
        path = 'results/unfrozen'
  print(opt['save_path'])
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
  opt['num_classes'] = 20
  opt['criterion'] = criterion
  log_writer = SummaryWriter(os.path.join(opt['save_path'], 'logfile'))

  for x in os.listdir(path):
    if x.endswith('.pth'):
      print(f"testing {x}")
      #save_path = os.path.join(opt['save_path'], x[:-4])
      model = cifar_model.CifarIndependent(os.path.join(path,x), testing=True)

      device = opt['device']
        #Setup model


      model = model.to(device)
      train_loader, test_color_loader, test_gray_loader  = set_data(opt)
      criterion = opt['criterion']
      print_freq = opt['print_freq']
      torch.cuda.empty_cache()

      test(x)