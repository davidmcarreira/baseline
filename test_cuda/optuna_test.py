import os
#os.system("pip install optuna optuna-dashboard")
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import engine_v3
import utils
import time
import optuna
import torch.optim as optim

from utils.logger import OutputLogger
from optuna.trial import TrialState
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training, extract_face
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import distributed
from metrics import ArcMarginProduct
from torch.optim.lr_scheduler import StepLR, PolynomialLR


def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int
):
    """
    Creates training and testing DataLoaders.
    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.
    Args:
      train_dir: Path to training directory.
      test_dir: Path to testing directory.
      transform: torchvision transforms to perform on training and testing data.
      batch_size: Number of samples per batch in each of the DataLoaders.
      num_workers: An integer for number of workers per DataLoader.
    Returns:
      A tuple of (train_dataloader, test_dataloader, class_names).
      Where class_names is a list of the target classes.
      Example usage:
        train_dataloader, test_dataloader, class_names = \
          = create_dataloaders(train_dir=path/to/train_dir,
                               test_dir=path/to/test_dir,
                               transform=some_transform,
                               batch_size=32,
                               num_workers=4)
    """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    img,label = train_data[0][0], train_data[0][1]

    # Get class names
    class_names = train_data.classes
    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names

def model_summary(model): 
    
    try:
        import torchinfo
    except ModuleNotFoundError:
        os.system("pip install torchinfo")
        import torchinfo
        
    summary = torchinfo.summary(model=model, 
        input_size=(32, 3, 160, 160), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )
    
    return summary


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


epochs = 20
batch_size = 256
num_classes = 10000#5319
#sample_rate = 1.0
momentum = 0.9
weight_decay = 5e-6
num_workers = os.cpu_count()


def objective(trial):
    torch.cuda.empty_cache()
    model = InceptionResnetV1(classify=False, pretrained='casia-webface').to(device)
    for param in model.parameters():
        param.requires_grad = False # Freezes all the layers
        
    s = trial.suggest_int("scale", 6, 64, 2)
    m = trial.suggest_float("margin", 5e-1, 1, log=True)
        
    criterion = torch.nn.CrossEntropyLoss()
    metric_fc = ArcMarginProduct(512, num_classes, s=s, m=m, easy_margin=False).to(device)
    #
    #unfreeze_layers = [
    #                   facenet.mixed_7a,
    #                   facenet.repeat_3, 
                        #model.block8.conv2d, 
    #                   facenet.avgpool_1a, 
    #                   facenet.dropout, 
                        #model.last_linear, 
                        #model.last_bn]
                        #model.logits]
    
    #for layer in unfreeze_layers:
        #for param in layer.parameters():
            #param.requires_grad = True
        
    # Generate the optimizers.
    #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    #optimizer = getattr(optim, optimizer_name)([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=lr)
    #optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr = lr)
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}], momentum = momentum, lr = lr, weight_decay = weight_decay)
    #scheduler = PolynomialLR(optimizer, epochs, 1, verbose=False)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_dir = "/test_cuda/datasets/digiface_cropped/train_dir"
    test_dir = "/test_cuda/datasets/digiface_cropped/test_dir"
    #train_dir = "/test_cuda/datasets/QMUL_split/train_dir"
    #test_dir = "/test_cuda/datasets/QMUL_split/train_dir"
    data_transforms = transforms.Compose([
                                        transforms.Resize((160,160)),
                                        np.float32,
                                        transforms.ToTensor(),
                                        #transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
                                        fixed_image_standardization
                                        ])
    
    train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir, test_dir, data_transforms, batch_size, num_workers)
    
    print(lr, s, m)
    lfw, xqlfw, cplfw = engine_v3.train(model = model, 
             train_dataloader = train_dataloader, 
             test_dataloader = test_dataloader,
             metric_fc = metric_fc,
             scheduler = scheduler,
             criterion = criterion,
             optimizer = optimizer,
             epochs = epochs,
             device = device)

    return lfw, xqlfw, cplfw


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        # No arguments provided, treat the argument as None
        arg_v = None
    else:
        # Argument provided, store its value
        arg_v = sys.argv[1]
        
    study_name = "Study 24"
    storage = "sqlite:///optuna/study24.db"

    if arg_v == "True":
        study = optuna.load_study(study_name = study_name, storage = storage)
        trials = sorted(study.best_trials, key=lambda t: t.values)
        log_filename = "optuna.txt"
        with open(log_filename, "a") as log_file:
            ## Create an instance of the OutputLogger
            output_logger = OutputLogger(log_file)
            ## Set sys.stdout to the OutputLogger instance
            sys.stdout = output_logger
            print("\n========== {} {} ==========".format(study_name, storage))
            for trial in trials:
                print("  Trial#{}".format(trial.number))
                print("    Values: lfw={}, xqlfw={}, cplfw={}".format(trial.values[0], trial.values[1], trial.values[2]))
                print("    Params: {}".format(trial.params))
        sys.stdout = sys.__stdout__
        log_file.close()
    elif arg_v is None:
        study = optuna.create_study(directions=['maximize', 'maximize', 'maximize'], study_name = study_name, storage = storage)
        study.optimize(objective, show_progress_bar = True)
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])