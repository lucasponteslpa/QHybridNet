import tensorflow as tf
from sklearn import preprocessing, decomposition
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
n_class = 10
pca = decomposition.PCA(n_components=512)
labels_train = load_hot_labels(labels_train, n_class)
labels_val = load_hot_labels(labels_val, n_class)
res_dir = 'results_baseline_nlp'
make_dirs(res_dir)
arch_type = 3
if arch_type == 0:
    model = tf.keras.Sequential([tf.keras.layers.Dense(n_class,activation="sigmoid")])
elif arch_type == 1:
    model = tf.keras.Sequential([tf.keras.layers.Dense(1,activation="relu"),
                                 tf.keras.layers.Dense(n_class,activation="sigmoid")])
elif arch_type == 2:
    model = tf.keras.Sequential([tf.keras.layers.Dense(2,activation="relu"),
                                 tf.keras.layers.Dense(n_class,activation="sigmoid")])
elif arch_type == 3:
    model = tf.keras.Sequential([tf.keras.layers.Conv1D(1, 64, strides=64, activation="relu"),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(n_class,activation="sigmoid")])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#pca_data = pca.fit_transform(np.concatenate((data_train, data_val)))
if arch_type == 3:
    pca_data = np.expand_dims(np.concatenate((data_train[:5000,:512], data_val[:1500,:512])),axis=2)
else:
    pca_data = np.concatenate((data_train[:5000,:512], data_val[:1500,:512]))
#scaler = preprocessing.StandardScaler().fit(pca_data)
#pca_data = scaler.transform(pca_data)
#pca_data = preprocessing.normalize(pca_data, norm='l2')
train = model.fit(x=pca_data[:5000],
                  y=labels_train[:5000],
                  epochs=5,
                  batch_size=16,
                  verbose=1,
                  validation_data=(pca_data[5000:],labels_val[:1500]))
print(model.summary())
np.save(os.path.join(res_dir,'train_loss'),train.history['loss'])
np.save(os.path.join(res_dir,'train_acc'),train.history['accuracy'])
np.save(os.path.join(res_dir,'val_loss'),train.history['val_loss'])
np.save(os.path.join(res_dir,'val_acc'),train.history['val_accuracy'])
