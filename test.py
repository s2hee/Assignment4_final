import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from resnet import ResNet18
import torchvision.transforms as transforms
import torchvision
import os
import util
from dataset import CIFAR10
import torchvision.transforms as tr
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001,
                        momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

jittering = util.ColorJitter(brightness=0.4, contrast=0.4,
                                saturation=0.4)
lighting = util.Lighting(alphastd=0.1,
                            eigval=[0.2175, 0.0188, 0.0045],
                            eigvec=[[-0.5675, 0.7192, 0.4009],
                                    [-0.5808, -0.0045, -0.8140],
                                    [-0.5836, -0.6948, 0.4203]])

transf_test = tr.Compose([tr.ToTensor(), tr.Normalize(*stats, inplace=True)])

testset = CIFAR10('C:/Users/JHP/Desktop/cifar10/test', transform=transf_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

best_acc = 0  

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

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
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