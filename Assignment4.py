import torch
import torchvision
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------------
DATA_DIR = "./cifar10"
PATH = "./cifar_net_2epoch.pth"
CLASSES = ("plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

# ToTensor -> values 0..1, Normalize(mean 0.5, std 0.5) -> values -1..1
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def _trainset():
    return torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                        download=True, transform=TRANSFORM)


def _testset():
    return torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                        download=True, transform=TRANSFORM)


# ---------------------------------------------------------------------------
# Question 1a
# ---------------------------------------------------------------------------
def CIFAR10_dataset_a():
    """write the code to grab a single mini-batch of 4 images from the training set, at random.
   Return:
    1. A batch of images as a torch array with type torch.FloatTensor.
    The first dimension of the array should be batch dimension, the second channel dimension,
    followed by image height and image width.
    2. Labels of the images in a torch array

    """
    trainloader = torch.utils.data.DataLoader(_trainset(), batch_size=4,
                                              shuffle=True, num_workers=0)
    images, labels = next(iter(trainloader))      # images: (4, 3, 32, 32), labels: (4,)
    images = images.type(torch.FloatTensor)
    return images, labels


def show_batch(images, labels, save_path=None):
    """Display the 4 images side by side with their labels overlaid (for the report only)."""
    fig, axes = plt.subplots(1, len(images), figsize=(10, 3))
    for ax, img, lab in zip(axes, images, labels):
        img = img / 2 + 0.5                          # undo normalization: -1..1 -> 0..1
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))  # (C, H, W) -> (H, W, C)
        ax.text(1, 3, CLASSES[int(lab)], color="yellow", fontsize=12,
                fontweight="bold", backgroundcolor="black")
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Question 2 - network (same structure as the PyTorch CIFAR-10 tutorial)
# ---------------------------------------------------------------------------
class Net(nn.Module):
    # Use this function to define your network
    # Creates the network
    def __init__(self):
        super().__init__()
        # Inits the model layers
        self.conv1 = nn.Conv2d(3, 6, 5)       # 3 -> 6 channels, 5x5 kernels
        self.pool = nn.MaxPool2d(2, 2)        # halves height and width
        self.conv2 = nn.Conv2d(6, 16, 5)      # 6 -> 16 channels, 5x5 kernels
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)          # 10 classes

    def forward(self, x):
        # Defines forward apth
        x = self.pool(F.relu(self.conv1(x)))  # (N, 6, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (N, 16, 5, 5)
        x = torch.flatten(x, 1)               # (N, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                       # (N, 10): one score per class
        return x


def train_classifier():
    # Creates dataset
    trainloader = torch.utils.data.DataLoader(_trainset(), batch_size=4,
                                              shuffle=True, num_workers=0)

    # Creates Network
    net = Net()

    # Defines loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset for 2 iteration
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()               # reset gradients
            outputs = net(inputs)               # forward
            loss = criterion(outputs, labels)
            loss.backward()                     # backward
            optimizer.step()                    # update weights

            running_loss += loss.item()
            if i % 2000 == 1999:                # every 2000 mini-batches
                # print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    # Saves the model weights after training
    PATH = './cifar_net_2epoch.pth'
    torch.save(net.state_dict(), PATH)


def evalNetwork():
    # Initialized the network and load from the saved weights
    PATH = './cifar_net_2epoch.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH, map_location="cpu"))
    net.eval()
    # Loads dataset
    batch_size=4
    transform = TRANSFORM
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            # Evaluates samples
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)   # class with the highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    # print(f"Accuracy of the network on the 10000 test images: {accuracy:.2f} %")
    return accuracy


# ---------------------------------------------------------------------------
# Question 3 - weights
# ---------------------------------------------------------------------------
def get_first_layer_weights():
    net = Net()
    # TODO: load the trained weights
    net.load_state_dict(torch.load(PATH, map_location="cpu"))
    first_weight = net.conv1.weight.detach().clone()   # (6, 3, 5, 5), bias excluded
    return first_weight


def get_second_layer_weights():
    net = Net()
    # TODO: load the trained weights
    net.load_state_dict(torch.load(PATH, map_location="cpu"))
    second_weight = net.conv2.weight.detach().clone()  # (16, 6, 5, 5), bias excluded
    return second_weight


# ---------------------------------------------------------------------------
# Question 4 - hyperparameter sweep
# ---------------------------------------------------------------------------
def _error_on_subset(net, dataset, n=1000):
    """Error (%) on n randomly sampled images from the dataset."""
    idx = torch.randperm(len(dataset))[:n].tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, idx),
                                         batch_size=100, shuffle=False, num_workers=0)
    correct = 0
    net.eval()
    with torch.no_grad():
        for images, labels in loader:
            correct += (net(images).argmax(1) == labels).sum().item()
    net.train()
    return 100.0 * (1 - correct / n)


def hyperparameter_sweep():
    '''
    Reuse the CNN and training code from Question 2
    Train the network three times using different learning rates: 0.01, 0.001, and 0.0001
    During training, record the training loss every 2000 iterations
    compute and record the training and test errors every 2000 iterations by randomly sampling 1000 images from each dataset
    After training, plot three curves
    '''
    trainset, testset = _trainset(), _testset()
    learning_rates = [0.01, 0.001, 0.0001]
    results = {}

    for lr in learning_rates:
        torch.manual_seed(0)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                  shuffle=True, num_workers=0)
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

        iters, losses, train_err, test_err = [], [], [], []
        step, running_loss = 0, 0.0
        for epoch in range(2):
            for inputs, labels in trainloader:
                optimizer.zero_grad()
                loss = criterion(net(inputs), labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                step += 1
                if step % 2000 == 0:
                    iters.append(step)
                    losses.append(running_loss / 2000)
                    train_err.append(_error_on_subset(net, trainset))
                    test_err.append(_error_on_subset(net, testset))
                    running_loss = 0.0
                    # print(f"lr={lr} iter={step} loss={losses[-1]:.3f} "
                    #       f"train_err={train_err[-1]:.1f}% test_err={test_err[-1]:.1f}%")

        results[lr] = {"iters": iters, "loss": losses,
                       "train_err": train_err, "test_err": test_err}

    # Three plots, one curve per learning rate
    for key, ylabel, fname in [("loss", "Training loss", "sweep_train_loss.png"),
                               ("train_err", "Training error (%)", "sweep_train_error.png"),
                               ("test_err", "Test error (%)", "sweep_test_error.png")]:
        plt.figure(figsize=(7, 4.5))
        for lr in learning_rates:
            plt.plot(results[lr]["iters"], results[lr][key], marker="o", label=f"lr = {lr}")
        plt.xlabel("Iteration (mini-batches of 4)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs iteration")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname)
        # plt.show()
        plt.close()

    return results


if __name__ == "__main__":
    # your text code here
    # images, labels = CIFAR10_dataset_a()
    # show_batch(images, labels, save_path="q1a_batch.png")

    train_classifier()                     # Question 2: trains and saves cifar_net_2epoch.pth
    # print("Test accuracy:", evalNetwork())

    # w1 = get_first_layer_weights()       # Question 3
    # w2 = get_second_layer_weights()
    # print(w1.shape, w2.shape)

    # results = hyperparameter_sweep()     # Question 4 (slow: trains 3 networks)
