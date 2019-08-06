import argparse
import json
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from imgaug import augmenters as iaa
import imgaug as ia

import torchvision
from torchvision import datasets, models, transforms
import numpy as np


# sadly the library does have GPU support as of now
class AdvancedAugmentation:
  def __init__(self):
    self.aug = iaa.Sequential([
        iaa.Resize((224, 224)),
        # blur images with a sigma of 0 to 3.0
        iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
        # horizontally flip 50% of the images
        iaa.Fliplr(0.5),
        # rotate by -20 to +20 degrees. The default mode is 'constant' which displays a constant value where the
        # picture was 'rotated out'. A better mode is 'symmetric' which 
        #'Pads with the reflection of the vector mirrored along the edge of the array' (see docs) 
        iaa.Affine(rotate=(-20, 20), mode='symmetric'),
        # do edge detection on 25% of the pictures
        iaa.Sometimes(0.10,
                      iaa.EdgeDetect(alpha=(0.5, 1.0))),
    ])
      
  def __call__(self, img):
    img = np.array(img)

    '''
    # This is a experimental try to get even more shape
    # informations via imgaug heatmap feature.
    # While this code works, I could not find out how
    # to create custom heatmaps (not the quoakka_heatmap) for our images

    heatmap = ia.quokka_heatmap(size=0.25)
    print(f'img shape via heatmap.shape {heatmap.shape}')
    print(f'img shape {img.shape}')
    image_aug, heatmap_aug = self.aug(image=img, heatmaps=heatmap)
    return np.hstack([image_aug, heatmap_aug.draw(cmap="gray")[0]])
    
    '''

    # either return a PIL.Image here or use PyTorch Lambda transform later on
    # return PIL.Image.fromarray(self.aug.augment_image(img))    
    return self.aug.augment_image(img)


# imports the model (in our case it just loads a pretrained one)
def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the pretrained vgg16 model
    model = models.vgg16(pretrained=True)

    # freeze all parameters, that they will not be changed during training (core of transfer learning)
    for param in model.parameters():
        param.requires_grad = False

    # replace the last layer with a new linear one, that is not trained yet
    n_inputs = model.classifier[6].in_features
    last_layer = nn.Linear(n_inputs, 133)
    model.classifier[6] = last_layer

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")

    return model


# Define a PyTorch dataloader that can be used in our training script
def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    from PIL import ImageFile
    import PIL
    
    # Fix for cuda error resulting from truncated images
    # https://stackoverflow.com/a/23575424/7434289
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    data_transforms = {
        'train': torchvision.transforms.Compose([
            AdvancedAugmentation(),
            # this lambda expression is quite important as it transforms the output of imgaug library to a format
            # that our torchvision.transforms steps can pick up
            torchvision.transforms.Lambda(lambda x: PIL.Image.fromarray(x)),
            # grayscale 30% of our pictures
            torchvision.transforms.RandomGrayscale(p=0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
    }

    data_dir = training_dir

    # load the dataset according to their path structure on S3. Apply the data_transforms
    # specified above
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'valid', 'test']}
    
    #
    # ATTENTION: WE NEED TO SET num_workers to 0! Otherwise we will run into errors (because of the imgaug dependency)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'valid', 'test']}

    class_names = image_datasets['train'].classes
    n_classes = len(class_names)

    return dataloaders


# Provided training function
def train(model, train_loader, epochs, criterion, optimizer, device):
    """
    This method closely resembles the training methods that we used throughout the course.
    This is the training method that is called by the PyTorch training script.
    
    The parameters passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training.
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """

    for epoch in range(1, epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader['train']:
            # move to GPU
            data = data.to(device)
            target = target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()
        for data, target in train_loader['valid']:
            # move to GPU
            data = data.to(device)
            target = target.to(device)
            ## update the average validation loss
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss += loss.item() * data.size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader['train'].dataset)
        valid_loss = valid_loss / len(train_loader['valid'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))


if __name__ == '__main__':

    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job

    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # define vgg16 model
    model = models.vgg16(pretrained=True)  # models.resnet50(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

        
    # ResNet50 code    
    # to be able to switch out the last layer, we first need to know the shape,
    # so that the dimensions are the same
    # in_features = model.fc.in_features

    # our transfer learning algorithm will cut out the last layer
    # and retrain it on our data
    # model.fc = nn.Linear(in_features, 133)
    # model.to(device)

    # vgg16
    n_inputs = model.classifier[6].in_features
    last_layer = nn.Linear(n_inputs, 133)
    model.classifier[6] = last_layer
    model.to(device)

    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Trains the model (given line of code, which calls the above training function)
    train(model, train_loader, args.epochs, criterion, optimizer, device)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
