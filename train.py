import json
import time
import os
import copy

from torch import nn, optim
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

from torch.autograd import Variable
import numpy as np
from PIL import Image
import torch
import shutil, argparse
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

running_on = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_data_loaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    resize = 256
    crop_size = 224
    batch_size = 64

    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, std)
        ]),
        'validation': transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(means, std)
        ]),
        'testing': transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(means, std)
        ]),
    }

    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform = data_transforms['training']),
        'validation': datasets.ImageFolder(valid_dir, transform = data_transforms['validation']),
        'testing': datasets.ImageFolder(test_dir, transform = data_transforms['testing']),
    }

    dataloaders = {
        'training': DataLoader(image_datasets['training'], batch_size = batch_size, shuffle = True),
        'validation': DataLoader(image_datasets['validation'], batch_size = batch_size),
        'testing': DataLoader(image_datasets['testing'], batch_size = batch_size)
    }

    return dataloaders, image_datasets, data_transforms

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def validate(loaders, criterion, model):
    accuracy = 0
    validation_loss = 0
    for images, labels in loaders['validation']:
        images = Variable(images)
        labels = Variable(labels)
        images, labels = images.to(running_on), labels.to(running_on)
        
        output = model.forward(images)
        validation_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        accuracy = accuracy + equality.type_as(torch.FloatTensor()).mean()
        
    return accuracy, validation_loss

def retain_checkpoint(checkpoint, dir, filename = 'checkpoint.pth'):
    path = dir + filename
    torch.save(checkpoint, path)

def train_model(loaders, img_datasets, transform, architecture, hidden_units, learning_rate, isCpu, epochs, directory_to_save):
    start = time.time()

    #model = models.densenet121(pretrained = True)
    model = eval("models.{}(pretrained=True)".format(architecture))

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1024, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 250)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(250, 102)),
        ('output', nn.LogSoftmax(dim = 1))
    ]))


    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    whole = 40
    steps = 0
    running_loss = 0

    if isCpu:
        model.cpu()
    else:
        model.cuda()
    
    for e in range(epochs):
        model.train()
        model.to(running_on)
        for images, labels in loaders['training']:
            images, labels = Variable(images), Variable(labels)
            steps += 1
            
            images, labels = images.to(running_on), labels.to(running_on)
            
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % whole == 0:
                model.eval()
                model.to(running_on)
                
                with torch.no_grad():
                    accuracy, validation_loss = validate(loaders, criterion, model)
                time_elapsed = time.time() - start
                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                     "-> Training Loss: {:.4f}.. ".format(running_loss / whole),
                     "-> Test Loss: {:.4f}.. ".format(validation_loss / len(loaders['validation'])),
                     "-> Test Accuracy: {:.4f}".format(accuracy / len(loaders['validation'])),
                     "-> {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
                
                running_loss = 0
                model.train()
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    checkpoint = {'batch_size': 64,
                 'data_transforms': transform,
                 'model': eval("models.{}(pretrained=True)".format(architecture)),
                 'classifier': classifier,
                 'optimizer': optimizer.state_dict(),
                 'state_dict': model.state_dict(),
                 'class_to_idx': img_datasets['training'].class_to_idx}

    retain_checkpoint(checkpoint, directory_to_save)


def main():
    parser = argparse.ArgumentParser(description = 'Start training with parameters')

    parser.add_argument('data_dir', type = str, help = 'Image Dataset Path')
    parser.add_argument('--save_dir', type = str, help = 'Directory to retain checkpoint')
    parser.add_argument('--arch', type = str, help = 'Models architeture. Default is densenet121')
    parser.add_argument('--learning_rate', type = float, help = 'Learning rate. Default is 0.003')
    parser.add_argument('--hidden_units', type = int, help = 'Number of hidden units. Default is 500')
    parser.add_argument('--epochs', type = int, help = 'Number of epochs. Default is 8')
    parser.add_argument('--gpu', action = 'store_true', help = 'GPU if available')
    
    args, _ = parser.parse_known_args()

    data_dir = args.data_dir

    save_dir = './'
    if args.save_dir:
        save_dir = args.save_dir

    arch = 'densenet121'
    if args.arch:
        arch = args.arch

    isCpu = True
    if args.gpu:
        if torch.cuda.is_available():
            isCpu = False

    learning_rate = 0.003
    if args.learning_rate:
        learning_rate = args.learning_rate

    hidden_units = 500
    if args.hidden_units:
        hidden_units = args.hidden_units

    epochs = 8
    if args.epochs:
        epochs = args.epochs

    loaders, img_datasets, transform = init_data_loaders(data_dir)

    train_model(loaders, img_datasets, transform, architecture = arch, hidden_units = hidden_units, learning_rate = learning_rate, isCpu = isCpu, epochs = epochs, directory_to_save = save_dir)

if __name__ == "__main__":
    main()






