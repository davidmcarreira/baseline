import torch
import sys
import time
import math
import argparse
import torch.backends.cudnn as cudnn
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple   
from sklearn.model_selection import KFold
from scipy import interpolate
#from utils.logger import OutputLogger
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training, extract_face
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
#from torch.optim.lr_scheduler import StepLR, PolynomialLR

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist
def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    is_false_positive = []
    is_false_negative = []

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], _ ,_ = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _, _, _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx], is_fp, is_fn = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
        is_false_positive.extend(is_fp)
        is_false_negative.extend(is_fn)

    return tpr, fpr, accuracy, is_false_positive, is_false_negative

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
    is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc, is_fp, is_fn

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, fp, fn  = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far, fp, fn

def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def get_paths_lfw(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs, dtype=object)

# def read_pairs_cplfw(pairs_filename):
#    pairs = []
#    with open(pairs_filename, 'r') as f:
#        for line in f.readlines()[0:]:
#            pair = line.strip().split()
#            pairs.append(pair)
#    return np.array(pairs, dtype=object)

# def get_paths_cplfw(lfw_dir, pairs):
#    nrof_skipped_pairs = 0
#    path_list = []
#    issame_list = []
#
#    for index in range(0, len(pairs)-1):
#        if pairs[index][1] == "1" and pairs[index+1][1] == "1":
#            if index % 2 == 0:
#                identity0 = ''.join([char for char in pairs[index][0][:-6]])
#                identity1 = ''.join([char for char in pairs[index+1][0][:-6]])
#                
#                path0 = os.path.join(lfw_dir, identity0, pairs[index][0])
#                path1 = os.path.join(lfw_dir, identity1, pairs[index+1][0])
#                
#                #print("True --->", index, pairs[index][0], pairs[index+1][0])
#                issame = True
#            else:
#                continue          
#        elif pairs[index][1] == "0" or pairs[index+1][1] == "0":
#            if index % 2 == 0:
#                identity0 = ''.join([char for char in pairs[index][0][:-6]])
#                identity1 = ''.join([char for char in pairs[index+1][0][:-6]])
#
#                path0 = os.path.join(lfw_dir, identity0, pairs[index][0])
#                path1 = os.path.join(lfw_dir, identity1, pairs[index+1][0])
#                
#                #print("False --->", index, pairs[index][0], pairs[index+1][0])
#                issame = False
#            else:
#                continue
#        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
#            path_list += (path0,path1)
#            issame_list.append(issame)
#        else:
#            nrof_skipped_pairs += 1
#    if nrof_skipped_pairs>0:
#        print('Skipped %d image pairs' % nrof_skipped_pairs)
#
#    return path_list, issame_list


def get_paths_xqlfw(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


#-------------------------------------------------------------------------------
def cplfw_evaluate(model, device):
    def cplfw_distance(embeddings1, embeddings2, distance_metric=0):
        if distance_metric==0:
            # Euclidian distance
            diff = np.subtract(embeddings1, embeddings2)
            dist = np.sum(np.square(diff),1)
        elif distance_metric==1:
            # Distance based on cosine similarity
            dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
            norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
            similarity = dot / norm
            dist = np.arccos(similarity) / math.pi
        else:
            raise 'Undefined distance metric %d' % distance_metric
    
        return dist
    
    def cplfw_calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
        assert(embeddings1.shape[0] == embeddings2.shape[0])
        assert(embeddings1.shape[1] == embeddings2.shape[1])
        nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = KFold(n_splits=nrof_folds, shuffle=True)
    
        tprs = np.zeros((nrof_folds,nrof_thresholds))
        fprs = np.zeros((nrof_folds,nrof_thresholds))
        accuracy = np.zeros((nrof_folds))
    
        is_false_positive = []
        is_false_negative = []
    
        indices = np.arange(nrof_pairs)
    
        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            if subtract_mean:
                mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
            else:
                mean = 0.0
            dist = cplfw_distance(embeddings1-mean, embeddings2-mean, distance_metric)
    
            # Find the best threshold for the fold
            acc_train = np.zeros((nrof_thresholds))
            for threshold_idx, threshold in enumerate(thresholds):
                _, _, acc_train[threshold_idx], _ ,_ = cplfw_calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
            best_threshold_index = np.argmax(acc_train)
            for threshold_idx, threshold in enumerate(thresholds):
                tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _, _, _ = cplfw_calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
            _, _, accuracy[fold_idx], is_fp, is_fn = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
    
            tpr = np.mean(tprs,0)
            fpr = np.mean(fprs,0)
            is_false_positive.extend(is_fp)
            is_false_negative.extend(is_fn)
    
        return tpr, fpr, accuracy, is_false_positive, is_false_negative
    
    def cplfw_calculate_accuracy(threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    
        is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
        is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)
    
        tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
        fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
        acc = float(tp+tn)/dist.size
        return tpr, fpr, acc, is_fp, is_fn
    
    def cplfw_calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
        assert(embeddings1.shape[0] == embeddings2.shape[0])
        assert(embeddings1.shape[1] == embeddings2.shape[1])
        nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = KFold(n_splits=nrof_folds, shuffle=True)
    
        val = np.zeros(nrof_folds)
        far = np.zeros(nrof_folds)
    
        indices = np.arange(nrof_pairs)
    
        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            if subtract_mean:
                mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
            else:
                mean = 0.0
            dist = cplfw_distance(embeddings1-mean, embeddings2-mean, distance_metric)
    
            # Find the threshold that gives FAR = far_target
            far_train = np.zeros(nrof_thresholds)
            for threshold_idx, threshold in enumerate(thresholds):
                _, far_train[threshold_idx] = cplfw_calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
            if np.max(far_train)>=far_target:
                f = interpolate.interp1d(far_train, thresholds, kind='slinear')
                threshold = f(far_target)
            else:
                threshold = 0.0
    
            val[fold_idx], far[fold_idx] = cplfw_calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
    
        val_mean = np.mean(val)
        far_mean = np.mean(far)
        val_std = np.std(val)
        return val_mean, val_std, far_mean
    
    def cplfw_calculate_val_far(threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
        false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        
        n_same = np.sum(actual_issame)
        n_diff = np.sum(np.logical_not(actual_issame))
        val = float(true_accept) / float(n_same)
        far = float(false_accept) / float(n_diff)
            
        return val, far
    
    
    def cplfw_evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
        # Calculate evaluation metrics
        thresholds = np.arange(0, 4, 0.01)
        embeddings1 = embeddings[0::2]
        embeddings2 = embeddings[1::2]
        tpr, fpr, accuracy, fp, fn  = cplfw_calculate_roc(thresholds, embeddings1, embeddings2,
            np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
        thresholds = np.arange(0, 4, 0.001)
        val, val_std, far = cplfw_calculate_val(thresholds, embeddings1, embeddings2,
            np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
        return tpr, fpr, accuracy, val, val_std, far, fp, fn
    
    def cplfw_add_extension(path):
        if os.path.exists(path+'.jpg'):
            return path+'.jpg'
        elif os.path.exists(path+'.png'):
            return path+'.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)
    
    def cplfw_get_paths(lfw_dir, pairs):
        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []
    
        for index in range(0, len(pairs)-1):
            if pairs[index][1] == "1" and pairs[index+1][1] == "1":
                if index % 2 == 0:
                    identity0 = ''.join([char for char in pairs[index][0][:-6]])
                    identity1 = ''.join([char for char in pairs[index+1][0][:-6]])
                    
                    path0 = os.path.join(lfw_dir, identity0, pairs[index][0])
                    path1 = os.path.join(lfw_dir, identity1, pairs[index+1][0])
                    
                    #print("True --->", index, pairs[index][0], pairs[index+1][0])
                    issame = True
                else:
                    continue          
            elif pairs[index][1] == "0" or pairs[index+1][1] == "0":
                if index % 2 == 0:
                    identity0 = ''.join([char for char in pairs[index][0][:-6]])
                    identity1 = ''.join([char for char in pairs[index+1][0][:-6]])
    
                    path0 = os.path.join(lfw_dir, identity0, pairs[index][0])
                    path1 = os.path.join(lfw_dir, identity1, pairs[index+1][0])
                    
                    #print("False --->", index, pairs[index][0], pairs[index+1][0])
                    issame = False
                else:
                    continue
            if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
                path_list += (path0,path1)
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs>0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)
    
        return path_list, issame_list
    
    def cplfw_read_pairs(pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[0:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs, dtype=object)

    # Testing folder
    root = "/test_cuda/datasets/CPLFW/cplfw"
    dest_dir = "/test_cuda/datasets/CPLFW/cplfw_cropped/"
    pairs_dir = "/test_cuda/datasets/CPLFW/cplfw_pairs.txt"
    
    torch.cuda.empty_cache()
    
    workers = 16
    batch_size = 32
    
    transf = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization #Normalizes tensors to [-1, 1]
    ])
    
    dataset = datasets.ImageFolder(dest_dir, transform=transf)
    
    data_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset)
    )

    classes = []
    embeddings = []
    model.eval()
    with torch.no_grad():
        for xb, yb in data_loader:
            #print(xb, yb)
            xb = xb.to(device)
            b_embeddings = model(xb)
            b_embeddings = b_embeddings.to('cpu').numpy()
            classes.extend(yb.numpy())
            embeddings.extend(b_embeddings)
            
            
    crop_path = []
    for i in os.listdir("/test_cuda/datasets/CPLFW/cplfw_cropped/"):
        cropped = os.path.join("/test_cuda/datasets/CPLFW/cplfw_cropped/", i)
        for j in os.listdir(cropped):
            final = os.path.join(cropped, j)    
            crop_path.append(final)
    
    crop_path = sorted(crop_path)
    
    embeddings_dict = dict(zip(crop_path, embeddings))
    
    pairs = cplfw_read_pairs(pairs_dir)
    path_list, issame_list = cplfw_get_paths(dest_dir, pairs)

    embeddings = np.array([embeddings_dict[path] for path in path_list])
    tpr, fpr, accuracy, val, val_std, far, fp, fn = cplfw_evaluate(embeddings, issame_list)
    
    cplfw_accuracy = np.mean(accuracy)
    
    return cplfw_accuracy


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

        ## Update results dictionary
        #results["train_loss"].append(train_loss)
        #results["train_acc"].append(train_acc)
        #results["test_loss"].append(test_loss)
        #results["test_acc"].append(test_acc)
    
    # Return the filled results at the end of the epochs
    torch.save(model.state_dict(), "model.pth")
    print("state_dict saved")
    
    resnet = InceptionResnetV1(num_classes=10000).to(device)

    dic = torch.load("model.pth")

    keys_to_remove = ['logits.weight', 'logits.bias']
    for key in keys_to_remove:
        dic.pop(key, None)
    
    resnet.load_state_dict(dic)
    print("model loaded lfw")
    
    cplfw_accuracy = cplfw_evaluate(resnet, device)
    
    #------------------------------------------------------------- LFW validation -------------------------------------------------------------
    # Testing folder
    root = "/test_cuda/datasets/LFW/lfw"
    dest_dir = "/test_cuda/datasets/LFW/lfw_cropped"
    pairs_dir = "/test_cuda/datasets/LFW/pairs.txt"
    
    torch.cuda.empty_cache()

    workers = 16
    batch_size = 32

    transf = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization #Normalizes tensors to [-1, 1]
    ])

    dataset = datasets.ImageFolder(dest_dir, transform=transf)

    data_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset)
    )
    
    
    classes = []
    embeddings = []
    resnet.eval()
    with torch.no_grad():
        for xb, yb in data_loader:
            #print(xb, yb)
            xb = xb.to(device)
            b_embeddings = resnet(xb)
            b_embeddings = b_embeddings.to('cpu').numpy()
            classes.extend(yb.numpy())
            embeddings.extend(b_embeddings)
    

    
    crop_path = []
    for i in os.listdir("/test_cuda/datasets/LFW/lfw_cropped"):
        cropped = os.path.join("/test_cuda/datasets/LFW/lfw_cropped", i)
        for j in os.listdir(cropped):
            final = os.path.join(cropped, j)    
            crop_path.append(final)
            
    crop_path = sorted(crop_path)
    
    embeddings_dict = dict(zip(crop_path, embeddings))   
    

    pairs = read_pairs(pairs_dir)
    path_list, issame_list = get_paths_lfw(dest_dir, pairs)


    embeddings = np.array([embeddings_dict[path] for path in path_list])
    tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate(embeddings, issame_list)

    lfw_accuracy = np.mean(accuracy)
    
    ##------------------------------------------------------------- XQLFW validation -------------------------------------------------------------
    ## Testing folder
    root = "/test_cuda/datasets/XQLFW/xqlfw"
    dest_dir = "/test_cuda/datasets/XQLFW/xqlfw_cropped"
    pairs_dir = "/test_cuda/datasets/XQLFW/xqlfw_pairs.txt"   

    torch.cuda.empty_cache()

    workers = 16
    batch_size = 32
    
    transf = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization #Normalizes tensors to [-1, 1]
    ])
    
    dataset = datasets.ImageFolder(dest_dir, transform=transf)
    
    data_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset)
    )
    
    classes = []
    embeddings = []
    resnet.eval()
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            b_embeddings = resnet(xb)
            b_embeddings = b_embeddings.to('cpu').numpy()
            classes.extend(yb.numpy())
            embeddings.extend(b_embeddings)
            
    crop_path = []
    for i in os.listdir("/test_cuda/datasets/XQLFW/xqlfw_cropped"):
        cropped = os.path.join("/test_cuda/datasets/XQLFW/xqlfw_cropped", i)
        for j in os.listdir(cropped):
            final = os.path.join(cropped, j)    
            crop_path.append(final)
            
    crop_path = sorted(crop_path) #Alphabetic sorting that coincides with the sequential sampling
    
    embeddings_dict = dict(zip(crop_path, embeddings))
    
    pairs = read_pairs(pairs_dir)
    path_list, issame_list = get_paths_xqlfw(dest_dir, pairs)
    
    embeddings = np.array([embeddings_dict[path] for path in path_list])

    tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate(embeddings, issame_list)
    
    xqlfw_accuracy = np.mean(accuracy)
    
    #--------------------------------------------
    
    
    
    return lfw_accuracy, xqlfw_accuracy, cplfw_accuracy
