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
class ImgAugTransform:
  def __init__(self):
    #heatmaps = np.random.random(size=(16, 64, 64, 1)).astype(np.float32)
    #self.heatmap = ia.quokka_heatmap(size=0.25)
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
        
        #iaa.Sometimes(0.25,
        #              iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
        #                         iaa.CoarseDropout(0.1, size_percent=0.5)])),
        iaa.Sometimes(0.25,
                      iaa.EdgeDetect(alpha=(0.5, 1.0))),
        
        #iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
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


# imports the model in model.py by name
# from model import BinaryClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.vgg16(pretrained=True)  # models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # vgg16
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

    '''
    ####
    model = models.vgg16(pretrained=True) # models.resnet50(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False


    # vgg16
    n_inputs = model.classifier[6].in_features
    last_layer = nn.Linear(n_inputs, 133)
    model.classifier[6] = last_layer
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    ####

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.vgg16(pretrained=True) # models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # vgg16
    n_inputs = model.classifier[6].in_features
    last_layer = nn.Linear(n_inputs, 133)
    model.classifier[6] = last_layer
    '''
    return model


# Gets training data in batches from the train.csv file
def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    # train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    # train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()

    # train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    # return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

    # Fix for cuda error resulting from truncated images
    # https://stackoverflow.com/a/23575424/7434289

    from PIL import ImageFile
    #from PIL import Image
    import PIL

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    data_transforms = {
        'train': torchvision.transforms.Compose([
            ImgAugTransform(),
            torchvision.transforms.Lambda(lambda x: PIL.Image.fromarray(x)),
            #torchvision.transforms.Resize(224),
            #torchvision.transforms.CenterCrop((224,224)),
            #iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            #torchvision.transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(1.0, 1.1), shear=5, resample=False, fillcolor=0),
            #torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.5),
            torchvision.transforms.RandomGrayscale(p=0.3),
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

    data_dir = training_dir  # 'dogImages'

    # we create some dictionaries
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'valid', 'test']}
    # ATTENTION: WE NEED TO SET num_workers to 0! Otherwise we will run into errors
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'valid', 'test']}

    class_names = image_datasets['train'].classes
    n_classes = len(class_names)

    return dataloaders


# Provided training function
def train(model, train_loader, epochs, criterion, optimizer, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training.
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """

    '''
    # training loop is provided
    for epoch in range(1, epochs + 1):
        model.train() # Make sure that the model is in training mode.

        total_loss = 0

        for batch in train_loader:
            # get data
            batch_x, batch_y = batch

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # get predictions from model
            y_pred = model(batch_x)

            # perform backprop
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()

        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))
    '''

    # valid_loss_min = 5.0 #np.Inf  # 3.877533

    # if os.path.exists(save_path):
    #    model.load_state_dict(torch.load(save_path))

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

        # save model if validation loss has decreased
        # if valid_loss <= valid_loss_min:
        #    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        #        valid_loss_min,
        #        valid_loss))
        #    torch.save(model.state_dict(), save_path)
        #    valid_loss_min = valid_loss
    # return trained model
    # return model


## TODO: Complete the main code
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

    ## TODO: Add args for the three model parameters: input_features, hidden_dim, output_dim
    # Model Parameters
    # parser.add_argument('--input_features', type=str, default='c_1 c_2 c_5 lcs_word')
    # parser.add_argument('--input_features', type=int, default=4)
    # parser.add_argument('--hidden_dim', type=int, default=10)
    # parser.add_argument('--output_dim', type=int, default=1)

    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    ## --- Your code here --- ##

    ## TODO:  Build the model by passing in the input params
    # To get params from the parser, call args.argument_name, ex. args.epochs or ards.hidden_dim
    # Don't forget to move your model .to(device) to move to GPU , if appropriate

    # model = BinaryClassifier(args.input_features, args.hidden_dim, args.output_dim).to(device)
    # define ResNet50 model
    model = models.vgg16(pretrained=True)  # models.resnet50(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

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

    ## TODO: Define an optimizer and loss function for training
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # criterion = nn.BCELoss()

    # optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Trains the model (given line of code, which calls the above training function)
    train(model, train_loader, args.epochs, criterion, optimizer, device)

    ## TODO: complete in the model_info by adding three argument names, the first is given
    # Keep the keys of this dictionary as they are
    # model_info_path = os.path.join(args.model_dir, 'model_info.pth')

    # with open(model_info_path, 'wb') as f:
    #    model_info = {
    #        'input_features': args.input_features,
    #        'hidden_dim': args.hidden_dim,
    #        'output_dim': args.output_dim,
    #    }
    #    torch.save(model_info, f)

    ## --- End of your code  --- ##

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
