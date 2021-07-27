import tensorflow as tf
from sklearn import preprocessing, decomposition
import os
import numpy as np
import sys

def load_hot_labels(labels):
    out_labels = []
    for y in labels:
        one_hot_labels = np.zeros(10)
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

pca = decomposition.PCA(n_components=512)
labels_train = load_hot_labels(labels_train)
labels_val = load_hot_labels(labels_val)
res_dir = 'results_baseline_nlp'
make_dirs(res_dir)
model = tf.keras.Sequential([tf.keras.layers.Dense(10,activation="sigmoid")])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
pca_data = pca.fit_transform(np.concatenate((data_train, data_val)))
#pca_data = np.concatenate((data_train, data_val))
scaler = preprocessing.StandardScaler().fit(pca_data)
pca_data = scaler.transform(pca_data)
pca_data = preprocessing.normalize(pca_data, norm='l2')
train = model.fit(x=pca_data[:data_train.shape[0]],
                  y=labels_train,
                  epochs=50,
                  batch_size=16,
                  verbose=1,
                  validation_data=(pca_data[data_train.shape[0]:],labels_val))
np.save(os.path.join(res_dir,'train_loss'),train.history['loss'])
np.save(os.path.join(res_dir,'train_acc'),train.history['accuracy'])
np.save(os.path.join(res_dir,'val_loss'),train.history['val_loss'])
np.save(os.path.join(res_dir,'val_acc'),train.history['val_accuracy'])
