# import libraries
import os
import numpy as np
import torch
from six import BytesIO
from torchvision import datasets, models, transforms
import torch.nn as nn

# import model from model.py, by name
# from model import BinaryClassifier

# default content type is numpy array
NP_CONTENT_TYPE = 'application/x-npy'


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


# Provided input data loading
def input_fn(serialized_input_data, content_type):
    """A default input_fn that can handle JSON, CSV and NPZ formats.
    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type
    Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # np_array = encoders.decode(serialized_input_data, content_type)
    # WE ASSUME THAT THE INPUT IS NOT SERIALIZED AND COMES IN AS NP ARRAY
    # tensor = torch.from_numpy(serialized_input_data)
    # return tensor.to(device)

    # next try from https://sagemaker.readthedocs.io/en/stable/using_pytorch.html

    # return torch.load(BytesIO(serialized_input_data)) #.to(device)

    print('Deserializing the input data.')
    if content_type == NP_CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


# Provided output data handling
def output_fn(prediction_output, accept):
    print('Serializing the generated output.')

    # stream = BytesIO()
    # np.save(stream, prediction_output)
    # return stream.getvalue(), accept
    if accept == NP_CONTENT_TYPE:
        stream = BytesIO()
        np.save(stream, prediction_output)
        return stream.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


# Provided predict function
def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process input_data so that it is ready to be sent to our model.
    # data = torch.from_numpy(input_data.astype('float32'))
    # data = data.to(device)

    data = torch.from_numpy(input_data.astype('float32'))
    data = data.to(device)

    # Put the model into evaluation mode
    model.eval()

    # Compute the result of applying the model to the input data
    # The variable `out_label` should be a rounded value, either 1 or 0
    out = model(data)
    out_np = out.cpu().detach().numpy()
    out_label = out_np.round()

    return out_label