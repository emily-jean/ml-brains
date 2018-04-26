#!/usr/bin/env python3

import csv
import fileinput
import io
import pickle
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def timestamp(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    hours = int(hours)
    minutes = int(minutes)
    if hours:
        return f'{hours}:{minutes}:{seconds:0.3f}'
    elif minutes:
        return f'{minutes}:{seconds:0.3f}'
    else:
        return f'{seconds:0.3f}'


def load_data_dir(path):
    data = None
    for filename in os.listdir(path):
        new = np.genfromtxt(os.path.join(path, filename), delimiter=',')
        if data is None:
            data = new
        else:
            data = np.concatenate((data, new))
    return np.split(data, (-1,), axis=1)

def measure_accuracy(model, test_data, test_labels):
    predicted = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predicted)
    print(f'Mean validation accuracy: {accuracy*100:.3f}%')
    
#    target_names = ['background', 'foreground']
#    cm = pd.DataFrame(confusion_matrix(test_labels, predicted), columns=target_names,
#                      index=target_names)
#    cm = cm.apply(np.log10)
#    cm = cm / cm.values.sum()
#    ax = sns.heatmap(cm, annot=True)
#    ax.set_title('Log-Normalized Validation Accuracy')


def save_prediction(model, path, validation_data):
        predicted = model.predict(validation_data)
        pd.DataFrame(predicted).to_csv(path, header=False, index=False)


def partition_data(n, path):
    files = [os.path.join(path, p) for p in os.listdir(path)
             if os.path.isfile(os.path.join(path, p))]
    #total size of dataset
    size = 0
    for filename in files:
        size += os.path.getsize(filename)
    
    if os.path.isfile(os.path.join(path, 'part_0.pickle')):
        return [os.path.join(path, f'part_{i}.pickle') for i in range(n)]

    output_files = []
    with fileinput.input(files=files) as f:
        for i in range(n):
            data = ''
            
            if i < n - 1:
                while len(data) < size / n:
                    data += f.readline()
            else:
                for line in f:
                    data += line
            
            outfile = os.path.join(path, f'part_{i}.pickle')
            output_files.append(outfile)
            ndarr = pd.read_csv(io.BytesIO(data.encode())).values
            data, labels = np.split(ndarr, (-1,), axis=1)
        
            with open(outfile, 'wb') as fo:
                pickle.dump((data, labels), fo)
        
    return output_files
    

def create_trees(files, modelpath, *,
            criterion='gini', overwrite=False, max_depth=15,
            random_state=123456):
    trees = []
    for i, filename in enumerate(files):
        model_name = os.path.join(modelpath,
                f'decision_tree_{criterion}_{i}.pickle')
        if os.path.isfile(model_name):
            with open(model_name, 'rb') as f:
                trees.append(pickle.load(f))
        else:
            with open(filename, 'rb') as f:
                train_data, train_labels = pickle.load(f)
            dt = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                    random_state=random_state)
            dt.fit(train_data, train_labels)
            with open(model_name, 'wb') as f:
                pickle.dump(dt, f)
            trees.append(dt)
            print(f'Tree {i} trained at {time.strftime("%H:%M:%S", time.localtime())}')
    return trees

def train_large_forest(n_estimators, modelpath, datapath, *,
            criterion='gini', overwrite=False, max_depth=15,
            random_state=123456):
    files = partition_data(n_estimators, datapath)

    rf = RandomForestClassifier(n_estimators = 1, criterion=criterion,
            max_depth=max_depth, random_state=random_state)
    with open(files[0], 'rb') as f:
        train_data, train_labels = pickle.load(f)
        rf.fit(train_data, train_labels.ravel().astype(int))
    rf.estimators_ = create_trees(files, modelpath, criterion=criterion,
            max_depth=max_depth, random_state=random_state)
    rf.n_estimators_ = len(rf.estimators_)
    return rf

    
def create_large_model(path, datapath, *, criterion='gini', overwrite=False, n_estimators=50,
                 max_depth=15, random_state=123456):
    if os.path.isfile(path) and not overwrite:
        print('loading model...')
        with open(path, 'rb') as f:
            rf = pickle.load(f)
        print(f'Model loaded! Original training time {timestamp(rf.training_time)}')
    else:
        resp = input('type "train" to confirm you want to train a new model: ')
        if resp != 'train':
            print('training aborted')
            return
        else:
            now = time.time()
            print(f'training... {time.strftime("%H:%M:%S", time.localtime())}')
            
            rf = train_large_forest(n_estimators, path, datapath, criterion=criterion,
                    max_depth=max_depth, random_state=random_state)
            rf.training_time = time.time() - now
            
            print(f'trained! Elapsed time {timestamp(rf.training_time)}')
            with open(os.path.join(path, 'full_class.pickle'), 'wb') as f:
                pickle.dump(rf, f)
    return rf


entropy_full = create_large_model('../models/',
                                  '../brain-data/full-train')

def load_data_dir(path, *, size):
    data = ''
    for filename in os.listdir(path):
        with open(os.path.join(path, filename)) as f:
            for line in f:
                if len(data) > size:
                    break
                else:
                    data += line + '\n'
        if len(data) > size:
            break

    ret = pd.read_csv(io.BytesIO(data.encode())).values
    return np.split(ret, (-1,), axis=1)

test_data, test_labels = load_data_dir('../brain-data/full-test', size=1e6)

measure_accuracy(entropy_full, test_data, test_labels)



