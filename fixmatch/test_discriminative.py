import argparse
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torch.nn.functional as F

import os
import sys
from utils import pickle_utils
from dataset.cifar import CIFARDataset
import models.wideresnet as models

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

prior_shift_weight = np.array([1/5 if i % 2 == 0 else 1 /
                               95 for i in range(10)] + [1/95 if i % 2 == 0 else 1/5 for i in range(10)])


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
    return train_loader, test_color_loader, test_gray_loader


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
    target_list = []
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            output_list.append(outputs)
            # feature_list.append(features)
            target_list.append(targets)

    outputs = torch.cat(output_list, dim=0)
    # features = torch.cat(feature_list, dim=0)
    targets = torch.cat(target_list, dim=0)

    accuracy_sum_prob_wo_prior_shift = compute_accuracy_sum_prob_wo_prior_shift(
        outputs, targets)
    accuracy_sum_prob_w_prior_shift = compute_accuracy_sum_prob_w_prior_shift(
        outputs, targets)
    accuracy_max_prob_w_prior_shift = compute_accuracy_max_prob_w_prior_shift(
        outputs, targets)

    test_result = {
        'accuracy_sum_prob_wo_prior_shift': accuracy_sum_prob_wo_prior_shift,
        'accuracy_sum_prob_w_prior_shift': accuracy_sum_prob_w_prior_shift,
        'accuracy_max_prob_w_prior_shift': accuracy_max_prob_w_prior_shift,
        'outputs': outputs.cpu().numpy(),
        # 'features': features.cpu().numpy()
    }
    return test_result


def compute_accuracy_sum_prob_wo_prior_shift(outputs, targets):
    probs = F.softmax(outputs, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    predictions = np.argmax(probs[:, :10] + probs[:, 10:], axis=1)
    accuracy = (predictions == targets).mean() * 100.
    return accuracy


def compute_accuracy_sum_prob_w_prior_shift(outputs, targets):
    probs = F.softmax(outputs, dim=1).cpu().numpy() * prior_shift_weight
    targets = targets.cpu().numpy()
    predictions = np.argmax(probs[:, :10] + probs[:, 10:], axis=1)
    accuracy = (predictions == targets).mean() * 100.
    return accuracy


def compute_accuracy_max_prob_w_prior_shift(outputs, targets):
    probs = F.softmax(outputs, dim=1).cpu().numpy() * prior_shift_weight
    targets = targets.cpu().numpy()
    predictions = np.argmax(
        np.stack((probs[:, :10], probs[:, 10:])).max(axis=0), axis=1)
    accuracy = (predictions == targets).mean() * 100.
    return accuracy


def test():
    # Test and save the result
    test_color_result = _test(test_color_loader)
    test_gray_result = _test(test_gray_loader)
    pickle_utils.save_pkl(test_color_result, os.path.join(
        save_path, 'test_color_result.pkl'))
    pickle_utils.save_pkl(test_gray_result, os.path.join(
        save_path, 'test_gray_result.pkl'))

    # Output the classification accuracy on test set
    info = ('Test on color images accuracy: {}\n'
            'Test on gray images accuracy: {}'.format(test_color_result['accuracy'],
                                                      test_gray_result['accuracy']))
    pickle_utils.write_info(os.path.join(save_path, 'test_result.txt'), info)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str,
                        default='cifar_baseline')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.set_defaults(cuda=True)
    parser.add_argument('--feature_dim', default=128,
                        type=int, help='Feature dim for latent vector')

    parser.add_argument('--batch_size', default=128, type=int,
                        help='Number of images in each mini-batch')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers for dataloader')
    parser.add_argument('--frozen', action='store_true')

    opt = vars(parser.parse_args())
    opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
    if opt['frozen']:
        opt['save_path'] = 'test/frozen'
        path = 'results/frozen'
    else:
        opt['save_path'] = 'test/cifar10s@1000.423'
        path = 'results/cifar10s@1000.423'
    opt['print_freq'] = 50
    opt['batch_size'] = 128
    if not os.path.exists(opt['save_path']):
        os.makedirs(opt['save_path'])
    opt['dataloader'] = CIFARDataset
    opt['test_color_path'] = '../data/cifar_color_test_imgs'
    opt['test_gray_path'] = '../data/cifar_gray_test_imgs'
    opt['test_label_path'] = '../data/cifar_test_labels'
    opt['train_data_path'] = '../data/cifar-s/p95.0/train_imgs'
    opt['train_label_path'] = '../data/cifar_train_labels'
    opt['num_classes'] = 20
    opt['criterion'] = F.cross_entropy

    for x in os.listdir(path):
        if x.endswith('.tar'):
            print(f"testing {x}")
            save_path = os.path.join(opt['save_path'], x[:-4])

            load_path = 'results/cifar10s@1000.423/' + x
            model_state_dict = torch.load(
                load_path, map_location=torch.device('cpu'))['state_dict']
            model = models.build_wideresnet(
                depth=28, widen_factor=2, dropout=0, num_classes=opt['num_classes'])
            model.load_state_dict(model_state_dict)

            device = opt['device']
            # Setup model

            model = model.to(device)
            train_loader, test_color_loader, test_gray_loader = set_data(opt)
            criterion = opt['criterion']
            print_freq = opt['print_freq']
            log_writer = SummaryWriter(os.path.join(save_path, 'logfile'))
            torch.cuda.empty_cache()

            test()
