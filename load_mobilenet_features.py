from PyTorch_CIFAR10.cifar10_models.mobilenetv2 import mobilenet_v2
from load_data import DataImageLoaders
from utils import make_dirs
import torch
from tqdm import tqdm
import os
import numpy as np

save_dir = 'cifar10_mobilenetv2_features'
train_dir = os.path.join(save_dir,'train')
val_dir = os.path.join(save_dir,'val')
make_dirs(train_dir)
make_dirs(val_dir)
dirs_dict = {"train":train_dir,"val":val_dir}
data_obj = DataImageLoaders(data_dir="_data/hymenoptera_data")
data_obj.CIFAR10_dataloaders(batch_size=64, num_classes=4)
img_batch, c = next(iter(data_obj.dataloaders["train"]))
model = mobilenet_v2(pretrained=True)
model.eval()

for phase in ["train","val"]:
    feat_out = model.features(img_batch)
    feat_out = feat_out.mean([2,3])
    all_data = np.empty((0,feat_out.shape[1]))
    all_labels = np.empty(0)
    n_batches = data_obj.dataset_sizes[phase] // 64
    with tqdm(total=n_batches) as t:
        for i, feat_l in enumerate(data_obj.dataloaders[phase]):
            feat, l = feat_l
            feat_out = model.features(feat)
            feat_out = feat_out.mean([2,3])
            out_norm = torch.norm(feat_out, dim=1)
            feat_out = feat_out/out_norm.view(-1,1)
            all_data = np.concatenate((all_data,feat_out.detach().numpy()))
            all_labels = np.concatenate((all_labels,l.detach().numpy()))
            t.update()
            del feat_out
    np.save(os.path.join(dirs_dict[phase],'data'), all_data)
    np.save(os.path.join(dirs_dict[phase], 'labels'), all_labels)
