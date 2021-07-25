import torch
import os

def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict

def save_ckpt(ckpt_name, models, optimizers, n_epochs):
    ckpt_dict = {'n_epochs': n_epochs}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)

def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name, map_location='cpu')
    for prefix, model in models:
        assert isinstance(model, torch.nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['n_epochs']


def checkpoint(model, optimizer, epoch, exp_dir):
    print('Saving checkpoint... \n')
    save_ckpt('{:s}/{:d}.pth'.format(exp_dir, epoch+1),
              [('model', model)], [('optimizer', optimizer)], epoch+1)

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

