import torch
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from variables import saved_model_path, data_transform, net


def train(train_set_path, batch_size, epoch_count):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    trainset = datasets.ImageFolder(root=train_set_path, transform=data_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("Training ...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epoch_count):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print("Training complete.")
    torch.save(net.state_dict(), saved_model_path)
