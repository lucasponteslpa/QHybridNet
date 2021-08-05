from qarch_tensorflow import QuantumInput
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
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
    elif sys.argv[1]=='digits':
        data, labels = load_digits(n_class=10, return_X_y=True)
        data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=0.3)
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
if sys.argv[1]!='digits':
    qmodel = QuantumInput((data_train[:5000,:512], labels_train[:5000]),
                          (data_val[:1500,:512], labels_val[:1500]),
                        list(range(2)),
                        6,
                        pca_dim=None,
                        num_measurements = None,
                        layer_type=0)
else:
    qmodel = QuantumInput((data_train, labels_train),
                          (data_val, labels_val),
                          list(range(10)),
                          30,
                          pca_dim=None,
                          num_measurements = None,
                          layer_type=0)
qmodel.training(batch_size=8, epochs=5, lr=1e-1, steps_decay=50)
np.save(os.path.join(res_dir,'train_loss'),qmodel.train.history['loss'])
np.save(os.path.join(res_dir,'train_acc'),qmodel.train.history['accuracy'])
np.save(os.path.join(res_dir,'val_loss'),qmodel.train.history['val_loss'])
np.save(os.path.join(res_dir,'val_acc'),qmodel.train.history['val_accuracy'])
