import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import tqdm
from thop import profile, clever_format
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from .. import dataloader
from .. import simclr
from .. import utils


class CifarBase(nn.Module):
    def __init__(self, pretrained_path):
        super(CifarBase, self).__init__()
        self.encoder = simclr.Simclr().encoder
        self.fc = nn.Linear(2048, 10, bias=True)
        self.load_state_Dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


def finetune_val(net, data_loader, train_optimizer, loss_criterion, epoch, epochs):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
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
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='results/128_0.5_200_512_500_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--test_color', type=bool, default=True)
    parser.add_argument('--split', type=float, default=0.9)
    args = parser.parse_args()
    model_path, batch_size, epochs, split = args.model_path, args.batch_size, args.epochs, args.split
    if args.test_color:
        finetune_data = "../../data/cifar-s/cifar_color_train_imgs"
        finetune_label = "../../data/cifar-s/cifar_train_labels"
    else:
        finetune_data = "../../data/cifar-s/cifar_color_train_imgs"
        finetune_label = "../../data/cifar-s/cifar_train_labels"

    ###RIGHT NOW NORMALIZE USING FINETUNED DATA, CAN SWITCH TO JUST THE ORIGINAL TRAINED 95.0 DATA like in the orignal paper"
    normalize = utils.getNormalize(finetune_data)

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
    finetune_data = dataloader.CifarDataset(finetune_data, finetune_label)
    finetune_ds, val_ds = random_split(finetune_data, [split * len(finetune_data), len(finetune_data) * (1 - split)])

    finetune_loader = DataLoader(finetune_ds, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = CifarBase(num_class=10, pretrained_path=model_path).cuda()
    for param in model.encoder.parameters():
        param.requires_grad = False

    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    best_acc = 0.0
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
        data_frame.to_csv('results/linear_statistics.csv', index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/linear_model.pth')
