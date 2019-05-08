import os

MODELS = "models"


def create_directory(directory=MODELS):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_path(directory=MODELS, file="example.h5"):
    create_directory(directory)
    return os.path.join(directory, file)
