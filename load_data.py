import torch
import numpy as np
import os
from torchvision import datasets, transforms


def get_samples(dataset, classes, samples_per_class):
    # import pdb;pdb.set_trace()
    sample_idx = np.empty(0)
    targets = np.array(dataset.targets, dtype=int)
    np.random.seed(420)
    for c in classes:
        idx_mask = targets == c
        idx = np.where(idx_mask)[0]
        if samples_per_class != None:
            sample_idx = np.concatenate((sample_idx,idx[:samples_per_class]), axis=0)
        else:
            sample_idx = np.concatenate((sample_idx,idx), axis=0)
    np.random.shuffle(sample_idx)
    return sample_idx.astype(int)

class DataImageLoaders():

    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.dataset_sizes = {}
        self.classes_names = []
        self.data_loaders = {}

    def FashionMNIST_dataloaders(self,
                                 batch_size = 4,
                                 num_classes = 2,
                                 num_samples_train = None,
                                 num_samples_val = None,
                                 transform = None):
        if transform == None:
            transform = transforms.Compose([transforms.ToTensor()])
        self.classes_names = list(range(num_classes))
        train_data = datasets.FashionMNIST(root='_data', train=True, download=True, transform=transform)
        val_data = datasets.FashionMNIST(root='_data', train=False, download=True, transform=transform)
        train_idx = get_samples(train_data, self.classes_names, num_samples_train)
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_idx = get_samples(val_data, self.classes_names, num_samples_val)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        image_datasets = {
            "train": train_data,
            "val": val_data
        }
        image_samplers = {
            "train": train_sampler,
            "val": val_sampler
        }
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
        self.classes_names = list(range(num_classes))
        self.dataloaders = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, sampler=image_samplers[x],num_workers=2)
            for x in ["train", "val"]
        }


    def KMNIST_dataloaders(self,
                           batch_size = 4,
                           num_classes = 2,
                           num_samples_train = None,
                           num_samples_val = None,
                           transform = None):
        if transform == None:
            transform = transforms.Compose([transforms.ToTensor()])
        self.classes_names = list(range(num_classes))
        train_data = datasets.KMNIST(root='_data', train=True, download=True, transform=transform)
        val_data = datasets.KMNIST(root='_data', train=False, download=True, transform=transform)
        train_idx = get_samples(train_data, self.classes_names, num_samples_train)
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_idx = get_samples(val_data, self.classes_names, num_samples_val)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        image_datasets = {
            "train": train_data,
            "val": val_data
        }
        image_samplers = {
            "train": train_sampler,
            "val": val_sampler
        }
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
        self.classes_names = list(range(num_classes))
        self.dataloaders = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, sampler=image_samplers[x],num_workers=2)
            for x in ["train", "val"]
        }


    def MNIST_dataloaders(self,
                          batch_size = 4,
                          num_classes = 2,
                          num_samples_train = None,
                          num_samples_val = None,
                          transform = None):
        if transform == None:
            transform = transforms.Compose([transforms.ToTensor()])
        self.classes_names = list(range(num_classes))
        train_data = datasets.MNIST(root='_data', train=True, download=True, transform=transform)
        val_data = datasets.MNIST(root='_data', train=False, download=True, transform=transform)
        train_idx = get_samples(train_data, self.classes_names, num_samples_train)
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_idx = get_samples(val_data, self.classes_names, num_samples_val)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        image_datasets = {
            "train": train_data,
            "val": val_data
        }
        image_samplers = {
            "train": train_sampler,
            "val": val_sampler
        }
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
        self.classes_names = list(range(num_classes))
        self.dataloaders = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, sampler=image_samplers[x],num_workers=2)
            for x in ["train", "val"]
        }

    def CIFAR10_dataloaders(self,
                            batch_size = 4,
                            num_classes = 2,
                            num_samples_train = None,
                            num_samples_val = None,
                            transform = None):
        if transform == None:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.classes_names = list(range(num_classes))
        train_data = datasets.CIFAR10(root='_data', train=True, download=True, transform=transform)
        val_data = datasets.CIFAR10(root='_data', train=False, download=True, transform=transform)
        train_idx = get_samples(train_data, self.classes_names, num_samples_train)
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_idx = get_samples(val_data, self.classes_names, num_samples_val)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

        # val_data.data, val_data.targets = get_samples_CIFAR(val_data, self.classes_names, num_samples, val_data.data.shape[1:])
        image_datasets = {
            "train": train_data,
            "val": val_data
        }
        image_samplers = {
            "train": train_sampler,
            "val": val_sampler
        }
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
        self.dataloaders = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, sampler=image_samplers[x], num_workers=2)
            for x in ["train", "val"]
        }

    def ImageNet_dataloaders(self, batch_size = 4, transform = None):
        if transform == None:
            data_transforms = {
                "train": transforms.Compose(
                    [
                        # transforms.RandomResizedCrop(224),     # uncomment for data augmentation
                        # transforms.RandomHorizontalFlip(),     # uncomment for data augmentation
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        # Normalize input channels using mean values and standard deviations of ImageNet.
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                ),
                "val": transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                ),
            }
        image_datasets = {
            x if x == "train" else "val": datasets.ImageFolder(
                os.path.join(self.data_dir, x), data_transforms[x]
            )
            for x in ["train", "val"]
        }
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
        self.classes_names = list(range(2))

        # Initialize dataloader
        self.dataloaders = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=8)
            for x in ["train", "val"]
        }
