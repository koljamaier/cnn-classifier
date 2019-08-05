# import libraries
import os
import numpy as np
import torch
from six import BytesIO
from torchvision import datasets, models, transforms
import torch.nn as nn

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

    return model


# Provided input data loading
def input_fn(serialized_input_data, content_type):
    """
    We assume, that data will be passed in as numpy array. With this information we can deserialize the
    data straightforward
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Deserializing the input data.')
    if content_type == NP_CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


# Provided output data handling
def output_fn(prediction_output, accept):
    print('Serializing the generated output.')

    if accept == NP_CONTENT_TYPE:
        stream = BytesIO()
        np.save(stream, prediction_output)
        return stream.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)



# this function gets called after our model is deployed. It gives back the prediction label for the dog breed
def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.from_numpy(input_data.astype('float32'))
    data = data.to(device)

    # Put the model into evaluation mode
    model.eval()

    # Compute the result of applying the model to the input data
    # The variable `out_label` should be a rounded value between 0 and 133 (our dog breed classes)
    out = model(data)
    out_np = out.cpu().detach().numpy()
    out_label = out_np.round()

    return out_label