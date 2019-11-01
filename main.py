'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from collections import OrderedDict

import os
import argparse

from models import *
from utils import progress_bar


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


# for epoch in range(start_epoch, start_epoch+200):
#     train(epoch)
#     test(epoch)

class FeatureShapeExtractor():

    def __call__(self,net,input_shape):
        self.hook_handles = []
        self.module_shape_dict = OrderedDict()

        net.apply(self.hook_adder)
        rand_input = torch.rand(input_shape)
        net(rand_input)

        for hook_handle in self.hook_handles:
            hook_handle.remove()

        return self.module_shape_dict

    def hook_adder(self,module):
        children_count = len(list(module.children()))
        if children_count == 0:
            self.hook_handles.append ( module.register_forward_pre_hook(self.forward_pre_hook) )

    def forward_pre_hook(self,module,input_tensor):
        self.module_shape_dict[module] = input_tensor[0].shape
        

    


def run_experiment(p):
    
    ### TRANSFORMS ###
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    ### DATA LOADERS ###
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    p["train_loader"] = trainloader
    p["test_loader"] = testloader

    ### CRITERION ###
    criterion = nn.CrossEntropyLoss()

    # if p["use_param_groups"]
    # optimizer = optim.SGD(p["net"].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


#create a list of all the model constructors and their arguments
net_list = [
    ( VGG               ,('VGG19',) ),
    ( ResNet18          ,()         ),
    ( PreActResNet18    ,()         ),
    ( GoogLeNet         ,()         ),
    ( DenseNet121       ,()         ),
    ( ResNeXt29_2x64d   ,()         ),
    ( MobileNet         ,()         ),
    ( MobileNetV2       ,()         ),
    ( DPN92             ,()         ),
    # ( ShuffleNetG2      ,()         ),
    ( SENet18           ,()         ),
    ( ShuffleNetV2      ,(1,)       ),
    # ( EfficientNetB0    ,()         ),
]

#create a parameter dictionary that holds the params for the experiments
p = {}
p["lr"]=0.01
p["momentum"] = 0.9
p["weight_decay"] = 5e-4
p["num_epochs"] = 200
p["results_dict"] = {}

#for all models
for net_class, net_args in net_list:

    
    p["net_name"] = net_class.__name__
    p["net"] = net_class(*net_args)
    
    print(p["net_name"])
    f = FeatureShapeExtractor()
    m = f(p["net"],(1,3,32,32))
    print(m)
    continue

    p1 = p.copy()
    p1["use_param_groups"] = False
    run_experiment(p1)

    p2 = p.copy()
    p2["use_param_groups"] = True
    run_experiment(p2)

