import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from resnet import ResNet18
import torchvision.transforms as transforms
import torchvision
import util
import torchvision.transforms as tr
from dataset import CIFAR10
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = ResNet18()
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

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

transf_train = tr.Compose([tr.ToTensor(), tr.RandomCrop(32, padding=4, padding_mode='reflect'),
                        tr.RandomHorizontalFlip(), jittering,
                        lighting, tr.Normalize(*stats, inplace=True)])

trainset = CIFAR10('C:/Users/JHP/Desktop/cifar10/train', transform=transf_train)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

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

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))