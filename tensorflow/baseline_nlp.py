import tensorflow as tf
from sklearn import preprocessing, decomposition
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import os
import numpy as np
import sys

from tensorflow.keras import layers
from tensorflow.python.keras.layers.core import Flatten

def load_hot_labels(labels, n_class):
    out_labels = []
    for y in labels:
        one_hot_labels = np.zeros(n_class)
        one_hot_labels[int(y)] = 1.0
        out_labels.append(one_hot_labels)

    return np.array(out_labels)

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
        data_train = np.load('../cifar10_mobilenetv2_features_2class/train/data.npy')
        labels_train = np.load('../cifar10_mobilenetv2_features_2class/train/labels.npy')
        data_val = np.load('../cifar10_mobilenetv2_features_2class/val/data.npy')
        labels_val = np.load('../cifar10_mobilenetv2_features_2class/val/labels.npy')
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
n_class = 2
pca = decomposition.PCA(n_components=128)
labels_train = load_hot_labels(labels_train, n_class)
labels_val = load_hot_labels(labels_val, n_class)
arch_type = 3
res_dir = 'results_baseline_cifar10_2classes_pca_arch3'
make_dirs(res_dir)
if arch_type == 0:
    model = tf.keras.Sequential([tf.keras.layers.Dense(n_class,activation="softmax")])
elif arch_type == 1:
    model = tf.keras.Sequential([tf.keras.layers.Dense(1,activation="relu"),
                                 tf.keras.layers.Dense(n_class,activation="softmax")])
elif arch_type == 2:
    model = tf.keras.Sequential([tf.keras.layers.Dense(2,activation="relu"),
                                 tf.keras.layers.Dense(n_class,activation="softmax")])
elif arch_type == 3:
    model = tf.keras.Sequential([tf.keras.layers.Conv1D(2, 8, strides=6, activation="relu"),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(n_class,activation="softmax")])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
if sys.argv[1]!='digits':
    if arch_type == 3:
        #pca_data = np.expand_dims(np.concatenate((data_train[:5000,:512], data_val[:1500,:512])),axis=2)
        pca_data = np.expand_dims(np.concatenate((data_train[:5000], data_val[:1500])),axis=2)
    else:
        pca_data = np.concatenate((data_train[:5000], data_val[:1500]))
else:
    if arch_type == 3:
        pca_data = np.expand_dims(np.concatenate((data_train, data_val)),axis=2)
    else:
        pca_data = np.concatenate((data_train, data_val))
pca_data = pca.fit_transform(pca_data)
#pca_data = np.expand_dims(pca_data, axis=2)
#scaler = preprocessing.StandardScaler().fit(pca_data)
#pca_data = scaler.transform(pca_data)
#pca_data = preprocessing.normalize(pca_data, norm='l2')
train = model.fit(x=pca_data[:5000] if sys.argv[1] != 'digits' else pca_data[:data_train.shape[0]],
                  y=labels_train[:5000] if sys.argv[1] != 'digits' else labels_train,
                  epochs=50,
                  batch_size=16,
                  verbose=1,
                  validation_data=(pca_data[5000:] if sys.argv[1] != 'digits' else pca_data[data_train.shape[0]:],
                                   labels_val[:1500] if sys.argv[1] != 'digits' else labels_val))
print(model.summary())
np.save(os.path.join(res_dir,'train_loss'),train.history['loss'])
np.save(os.path.join(res_dir,'train_acc'),train.history['accuracy'])
np.save(os.path.join(res_dir,'val_loss'),train.history['val_loss'])
np.save(os.path.join(res_dir,'val_acc'),train.history['val_accuracy'])
