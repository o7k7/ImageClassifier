from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import shutil, argparse, json
from torch.autograd import Variable

from PIL import Image


means = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
resize = 256
crop_size = 224
batch_size = 64


def load_checkpoint(filepath):
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location = 'cpu')
    
    model = models.densenet121(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model


def process_image(image):
    w, h = image.size # width, height

    generated_size = [0, 0]
    
    if w < h:
        generated_size = [resize, h]
    else:
        generated_size = [w, resize]
        
    image.thumbnail(generated_size, Image.ANTIALIAS)

    top = (resize - crop_size) / 2
    bottom = (resize + crop_size) / 2
    start = (resize - crop_size) / 2
    end = (resize + crop_size) / 2
    
    image = image.crop((start, top, end, bottom))
    
    image = np.array(image)
    image = image / 255
    
    stdv = np.array(std)
    mean = np.array(means)
    image = (image - mean) / stdv
    
    image = np.transpose(image, (2, 0, 1))
    
    return torch.from_numpy(image)

def predict(image_path, model, topk = 5, category_names = None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.cpu()
    
    image = process_image(Image.open(image_path))
    image = image.unsqueeze(0)
    output = model.forward(Variable(image, volatile = True))
    top_prob, top_labels = torch.topk(output, topk)
    top_prob = top_prob.exp()
    top_prob_arr = top_prob.data.numpy()[0]
    
    inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    
    top_labels_data = top_labels.data.numpy()
    
    top_classes = [inv_class_to_idx[x] for x in top_labels_data[0].tolist()]
    
    class_names = "None"
    if category_names is not None:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)

    class_names = [cat_to_name[i] for i in top_classes]
    
    return top_prob_arr, top_classes, class_names


def main():
    parser = argparse.ArgumentParser(description = 'Predicts the probability of flower image')
    parser.add_argument('input', type = str, help = 'Image path')
    parser.add_argument('checkpoint', type = str, help = 'Checkpoint of your model')
    parser.add_argument('--top_k', type = int, help = 'Returns the top k most likely classes')
    parser.add_argument('--category_names', type = str, help = 'Use a mapping of categories to real names from a json file')
    parser.add_argument('--gpu', action = 'store_true', help = 'Use GPU for inference if available')

    args, _ = parser.parse_known_args()

    image_path = args.input
    checkpoint_path = args.checkpoint

    top_k = 1
    if args.top_k:
        top_k = args.top_k

    category_names = None
    if args.category_names:
        category_names = args.category_names

    isCpu = True
    if args.gpu:
        if torch.cuda.is_available():
            isCpu = False

    checkpoint_model = load_checkpoint(checkpoint_path)
    probs, classes, class_names = predict(image_path, checkpoint_model.double(), topk = top_k, category_names = category_names)
    print("=" * 50)
    print(" " * 15 + 'RESULTS')
    print("=" * 50)
    print("Label in category names = {}".format(classes))
    print("Result confidence = {}".format(probs))
    print("Flower name : {}".format(class_names[0]))
    print("=" * 50)
    

if __name__ == '__main__':
    main()