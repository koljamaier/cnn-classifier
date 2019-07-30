# cnn-classifier
In this repository CNNs are used to classify dog breeds, leveraging PyTorch &amp; AWS SageMaker


# setup
`conda create -n capstone_project python=3.6`


To ensure reproducibility we use an `environment.yaml`
`conda env export > environment.yml`

If we later install packages via `conda install ...` we can update the `environment` via:
`conda env update --file environment.yml`

We use pysacffold in this project to enforce structurized, modular and packaged code (ML Engineering aspect!):
`conda install -c conda-forge pyscaffold`

Now we create a directory structure for our futurue package code:
`putup style_transfer`
Next we install everything, that resides inside `style_transfer/src`

`python setup.py develop`


Modules can now easily be imported like `from style_transfer.stylize import transfer`


