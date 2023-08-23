import torch
import sys
import time

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model, 
               dataloader,
               metric_fc,
               scheduler,
               criterion,
               optimizer,
               device):
    """Trains a PyTorch model for a single epoch.
    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).
    Args:
      model: A PyTorch model to be trained.
      dataloader: A DataLoader instance for the model to be trained on.
      loss_fn: A PyTorch loss function to minimize.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      device: A target device to compute on (e.g. "cuda" or "cpu").
    Returns:
      A tuple of training loss and training accuracy metrics.
      In the form (train_loss, train_accuracy). For example:
      (0.1112, 0.8743)
    """
    
    metric_fc.to(device)
    
    model.train()
    train_loss, train_acc = 0, 0

    for batch, data in enumerate(dataloader):
        data_input, label = data
        # Sending data to target device
        data_input = data_input.to(device)
        label = label.to(device).long()
        # Making predictions
        feature = model(data_input)
        output = metric_fc(feature, label)
        loss = criterion(output, label)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        train_acc += (y_pred_class == label).sum().item()/len(output)
        
        i = len(dataloader)*batch
        if i % 100 == 0:
            print("Train loss: {} ---- Train acc: {}".format(loss, (y_pred_class == label).sum().item()/len(output)))
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    scheduler.step()
    return train_loss, train_acc
            
    
def test_step(model, 
              dataloader,
              metric_fc,
              criterion,
              device):
    """Tests a PyTorch model for a single epoch.
    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.
    Args:
      model: A PyTorch model to be tested.
      dataloader: A DataLoader instance for the model to be tested on.
      loss_fn: A PyTorch loss function to calculate loss on the test data.
      device: A target device to compute on (e.g. "cuda" or "cpu").
    Returns:
      A tuple of testing loss and testing accuracy metrics.
      In the form (test_loss, test_accuracy). For example:
      (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, data in enumerate(dataloader):
            data_input, label = data
            # Sending data to target device
            data_input = data_input.to(device)
            label = label.to(device).long()
            
            # 1. Forward pass
            feature = model(data_input)
            
            # 2. Calculate and accumulate loss
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = output.argmax(dim=1)
            test_acc += ((test_pred_labels == label).sum().item()/len(test_pred_labels))
            
            i = len(dataloader)*batch
            if i % 100 == 0:
                print("Test loss: {} ---- Test acc: {}".format(loss, ((test_pred_labels == label).sum().item()/len(test_pred_labels))))
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    

    return test_loss, test_acc

def train(model, 
          train_dataloader, 
          test_dataloader,
          metric_fc,
          scheduler,
          criterion,
          optimizer,
          epochs,
          device):
    """Trains and tests a PyTorch model.
    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.
    Calculates, prints and stores evaluation metrics throughout.
    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for 
      each epoch.
      In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]} 
      For example if training for epochs=2: 
                   {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

        # Loop through training and testing steps for a number of epochs

    for epoch in tqdm(range(epochs)):
        print("\n{} -------> EPOCH: {}/{} \n".format(time.asctime(time.localtime(time.time())), epoch+1, epochs))
        train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            metric_fc=metric_fc,
                                            scheduler=scheduler,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            device=device)        
            
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        metric_fc=metric_fc,
                                        criterion=criterion,
                                        device=device)
            
        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}")

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    # Return the filled results at the end of the epochs
    return results




