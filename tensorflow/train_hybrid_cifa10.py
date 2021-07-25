from qarch_tensorflow import QuantumInput
from sklearn import preprocessing
import os
import numpy as np
import sys

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def pad(data):
    data_pad = np.zeros((data.shape[0], 2**(int(np.log2(data.shape[1]))+1)))
    data_pad[:,:data.shape[1]] = data
    data_pad = preprocessing.normalize(data_pad)

    return data_pad

use_pad = False
if len(sys.argv)==2:
    if sys.argv[1] == 'cifar10':
        data_train = np.load('../cifar10_mobilenetv2_features/train/data.npy')
        labels_train = np.load('../cifar10_mobilenetv2_features/train/labels.npy')
        data_val = np.load('../cifar10_mobilenetv2_features/val/data.npy')
        labels_val = np.load('../cifar10_mobilenetv2_features/val/labels.npy')
    else:
        data_train = np.load('../mnist_features/train/data.npy')
        labels_train = np.load('../mnist_features/train/labels.npy')
        data_val = np.load('../mnist_features/val/data.npy')
        labels_val = np.load('../mnist_features/val/labels.npy')
        use_pad = True
else:
    data_train = np.load('../mnist_features/train/data.npy')
    labels_train = np.load('../mnist_features/train/labels.npy')
    data_val = np.load('../mnist_features/val/data.npy')
    labels_val = np.load('../mnist_features/val/labels.npy')

if use_pad:
    data_train = pad(data_train)
    data_val = pad(data_val)
res_dir = 'results'
make_dirs(res_dir)
qmodel = QuantumInput((data_train, labels_train),(data_val, labels_val),list(range(10)),4)
qmodel.training(batch_size=8, epochs=50)
np.save(os.path.join(res_dir,'train_loss'),qmodel.train.history['loss'])
np.save(os.path.join(res_dir,'train_acc'),qmodel.train.history['accuracy'])
np.save(os.path.join(res_dir,'val_loss'),qmodel.train.history['val_loss'])
np.save(os.path.join(res_dir,'val_acc'),qmodel.train.history['val_accuracy'])