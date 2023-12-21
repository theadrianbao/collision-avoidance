import torch

from torch.utils.data import DataLoader
from torchvision import datasets

from variables import classes, data_transform, net, saved_model_path


def test(test_set_path, batch_size):
    net.load_state_dict(torch.load(saved_model_path))

    testset = datasets.ImageFolder(root=test_set_path, transform=data_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("Testing ...")

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
