import pickle
import numpy as np
import os
import sys
import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465),
#                          (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465),
#                          (0.2023, 0.1994, 0.2010)),
# ])

# trainset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=100, shuffle=False, num_workers=2)


def get_data_loader():
    train_x, train_y, train_l = get_data_set("train")
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).long()
    train_dataset = Data.TensorDataset(train_x, train_y)
    trainloader = Data.DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=True)

    test_x, test_y, test_l = get_data_set("test")
    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y).long()
    test_dataset = Data.TensorDataset(test_x, test_y)
    testloader = Data.DataLoader(
        dataset=test_dataset, batch_size=100, shuffle=False)
    return trainloader, testloader


def get_data_set(name="train"):
    x = None
    y = None
    l = None

    folder_name = "cifar-10-batches-py"
    f = open('data/'+folder_name+'/batches.meta', 'rb')
    datadict = pickle.load(f, encoding='latin1')
    f.close()
    l = datadict['label_names']

    if name == "train":
        for i in range(5):
            f = open('data/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 1, 2, 3])
            # _X = transforms.ToTensor()(_X)
            # _X = _X.reshape([-1, 3, 32, 32])
            # _X = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(_X)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name == "test":
        f = open('data/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 1, 2, 3])
        # x = transforms.ToTensor()(x)
        # x = x.reshape([-1, 3, 32, 32])
        # x = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(x)

    return x, y, l
