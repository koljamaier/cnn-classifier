# cnn-classifier
In this repository CNNs are used to classify dog breeds, leveraging PyTorch &amp; AWS SageMaker. 
The application mostly consists of notebooks that can be executed on AWS SageMaker (you can setup a notebook instance
there and refer to this git repository for instance).

## Notebooks
### 01-dog-breed-data-exploration-preprocessing.ipynb
This notebook gives an overview on the dataset and leads you through the preprocessing pipeline.

### 02-dog-breed-local-pytorch-training.ipynb
This notebook serves as a blueprint for the pytorch training that we want to execute on AWS SageMaker
(see `source_pytorch` and `source_pytorch_style_transfer`).

### 03-dog-breed-sagemaker-deployment-vgg16_imgaug.ipynb
This notebook executes our specified training and preprocessing via `imgaug` library within AWS SageMaker.
Furthermore this model is then deployed as an endpoint to make predictions and calculate the final accuracy.
All code that will be deployed to AWS SageMaker lives under `source_pytorch`.

### 04-dog-breed-sagemaker-deployment-vgg16_style-baseline.ipynb
This notebook is very similar to `03-dog-breed-sagemaker-deployment-vgg16_imgaug.ipynb`.
In fact, the same model is trained on style transferred images - except that this model does not leverage the `imgaug`
library. All code for this model (and deployment) can be found in `source_pytorch_style_transfer`.
Furthermore this notebook trains and validates the baseline model (that is the same model only trained on the original images).

### style_transfer
This package conatins the code, that is necessary to execute the AdaIN style transfer.
It leaverages the code base of https://github.com/naoto0804/pytorch-AdaIN.


### Libraries
This project makes heavy use of PyTorch and its ecosystem (like `torchvision`). Furthermore `imgaug` is used partly
for image augmentation. Also AWS SageMaker is used and its corresponding libraries.
