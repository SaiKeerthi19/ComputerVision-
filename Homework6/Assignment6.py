import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import models


# CIFAR-10 settings shared by Q2 and Q4
DATA_DIR = "./cifar10/"
CIFAR_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def _device():
    # use the Mac GPU (mps) or an NVIDIA GPU if there is one, otherwise the CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _loaders(batch_size=4):
    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=CIFAR_TRANSFORM)
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=CIFAR_TRANSFORM)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader


def _resnet18_pretrained():
    # newer torchvision uses weights=..., older versions use pretrained=True
    try:
        return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except AttributeError:
        return models.resnet18(pretrained=True)


def _accuracy(net, loader, device):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predicted = net(images).argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def _train(net, params, trainloader, device, epochs=10, frozen_backbone=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)
    for epoch in range(epochs):
        # frozen backbone: keep BatchNorm in eval mode so it uses its ImageNet statistics
        # (in train mode it would normalize over the 4 images of the batch and change its stored stats)
        net.eval() if frozen_backbone else net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                # print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0


# ---------------------------------------------------------------------------
# Q1
# ---------------------------------------------------------------------------
def compute_num_parameters(net:nn.Module):
    """compute the number of trainable parameters in *net* e.g., ResNet-34.
    Return the estimated number of parameters Q1.
    """
    # each parameter tensor has numel() numbers; count only the ones that get trained
    num_para = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return num_para


def CIFAR10_dataset_a():

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    return images, labels


# ---------------------------------------------------------------------------
# Q2
# ---------------------------------------------------------------------------
class GAPNet(nn.Module):
    """
    Insert your code here
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)                      # 32x32 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)                       # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(6, 10, 5)                     # 14x14 -> 10x10
        self.gap = nn.AvgPool2d(kernel_size=10, stride=10)   # 10x10 -> 1x1 (one average per channel)
        self.fc = nn.Linear(10, 10)                          # 10 averages -> 10 class scores

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (N, 6, 14, 14)
        x = F.relu(self.conv2(x))              # (N, 10, 10, 10)
        x = self.gap(x)                        # (N, 10, 1, 1)
        x = torch.flatten(x, 1)                # (N, 10)
        x = self.fc(x)                         # (N, 10)
        return x


def train_GAPNet():
    """
    Insert your code here
    """
    device = torch.device("cpu")   # small network: the CPU is fast enough
    trainloader, _ = _loaders()
    net = GAPNet().to(device)
    _train(net, net.parameters(), trainloader, device, epochs=10)
    torch.save(net.state_dict(), "./Gap_net_10epoch.pth")
    return net


def eval_GAPNet():
    """
    Insert your code here
    """
    device = torch.device("cpu")
    _, testloader = _loaders()
    net = GAPNet()
    net.load_state_dict(torch.load("./Gap_net_10epoch.pth", map_location="cpu"))
    net = net.to(device)
    accuracy = _accuracy(net, testloader, device)
    # print(f"GAPNet accuracy on the 10000 test images: {accuracy:.2f} %")
    return accuracy


# ---------------------------------------------------------------------------
# Q3
# ---------------------------------------------------------------------------
def backbone():
    """
    Insert your code here, Q3
    """
    model = _resnet18_pretrained()
    feature_extractor = nn.Sequential(*list(model.children())[:-1])   # everything except the last fc layer
    feature_extractor.eval()

    # standard ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open("cat_eye.jpg").convert("RGB")
    x = preprocess(img).unsqueeze(0)          # (1, 3, 224, 224)

    with torch.no_grad():
        features = feature_extractor(x)       # (1, 512, 1, 1)
    return features


# ---------------------------------------------------------------------------
# Q4
# ---------------------------------------------------------------------------
def transfer_learning():
    """
    Insert your code here, Q4
    """
    device = _device()
    trainloader, testloader = _loaders()

    model = _resnet18_pretrained()
    for param in model.parameters():          # freeze every pretrained weight
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 10)   # new last layer (trainable): 512 -> 10 classes
    model = model.to(device)

    _train(model, model.fc.parameters(), trainloader, device, epochs=10, frozen_backbone=True)   # only the last layer learns
    torch.save(model.state_dict(), "./Res_net_10epoch.pth")

    accuracy = _accuracy(model, testloader, device)
    # print(f"ResNet-18 transfer learning accuracy: {accuracy:.2f} %")
    return accuracy


# ---------------------------------------------------------------------------
# Q5
# ---------------------------------------------------------------------------
class MobileNetV1(nn.Module):
    """Define MobileNetV1 please keep the strucutre of the class Q5"""
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            # normal 3x3 convolution + BatchNorm + ReLU
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            # depthwise-separable block
            return nn.Sequential(
                # depthwise: one 3x3 filter per input channel (groups=inp)
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                # pointwise: 1x1 convolution mixes the channels
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)          # (N, 1024, 1, 1)
        x = x.view(-1, 1024)       # (N, 1024)
        x = self.fc(x)             # (N, n_classes)
        return x


if __name__ == '__main__':
    #Q1
    from torchvision import models
    resnet34 = models.resnet34(pretrained=True)
    num_para = compute_num_parameters(resnet34)
    # print("Q1 ResNet-34 parameters:", num_para)

    # Q2
    # train_GAPNet()
    # print("Q2 GAPNet accuracy:", eval_GAPNet())

    # Q3
    # print("Q3 features:", backbone().shape)

    # Q4
    # print("Q4 transfer learning accuracy:", transfer_learning())

    # Q5
    ch_in=3
    n_classes=1000
    model = MobileNetV1(ch_in=ch_in, n_classes=n_classes)
    # print(model(torch.randn(1, 3, 224, 224)).shape)
