import torch
from torch.utils.data import (
    Dataset, random_split, DataLoader
)
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from argparse import Namespace


# custom Dataset
class DogData(Dataset):

    def __init__(self, ds, transform = None) :
        self.ds = ds
        self.transform = transform

    def __len__(self) :
        return len(self.ds)

    def __getitem__(self, idx) :
        img, label = self.ds[idx]
        if self.transform :
            img = self.transform(img)
            return img, label


def get_loader(data, args, is_train=True):
    if is_train:
        collate_func = [transforms.RandomRotation(30)]
    else:
        collate_func = []
    collate_func.extend([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data_transforms = transforms.Compose(collate_func)
    dataset = DogData(data, data_transforms)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )


def dataloader(args):

    dataset = ImageFolder(args.imgfolder)

    test_size = int(len(dataset)*args.test_pct)
    dataset_size = len(dataset) - test_size

    val_size = int(dataset_size*args.val_pct)
    train_size = dataset_size - val_size

    train, val, test = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    trainLoader = get_loader(train, args)
    valLoader = get_loader(val, args, is_train=False)
    testLoader = get_loader(test, args, is_train=False)

    return trainLoader, valLoader, testLoader
