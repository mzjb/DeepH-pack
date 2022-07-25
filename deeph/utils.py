import os
import shutil
import sys
from configparser import ConfigParser

import numpy as np
import scipy
import torch
from torch import nn, package
import h5py


def print_args(args):
    for k, v in args._get_kwargs():
        print('{} = {}'.format(k, v))
    print('')


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class MaskMSELoss(nn.Module):
    def __init__(self) -> None:
        super(MaskMSELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert input.shape == target.shape == mask.shape
        mse = torch.pow(input - target, 2)
        mse = torch.masked_select(mse, mask).mean()

        return mse


class MaskMAELoss(nn.Module):
    def __init__(self) -> None:
        super(MaskMAELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert input.shape == target.shape == mask.shape
        mae = torch.abs(input - target)
        mae = torch.masked_select(mae, mask).mean()

        return mae


class LossRecord:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        self.last_val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


class Transform:
    def __init__(self, tensor=None, mask=None, normalizer=False, boxcox=False):
        self.normalizer = normalizer
        self.boxcox = boxcox
        if normalizer:
            raise NotImplementedError
            self.mean = abs(tensor).sum(dim=0) / mask.sum(dim=0)
            self.std = None
            print(f'[normalizer] mean: {self.mean}, std: {self.std}')
        if boxcox:
            raise NotImplementedError
            _, self.opt_lambda = scipy.stats.boxcox(tensor.double())
            print('[boxcox] optimal lambda value:', self.opt_lambda)

    def tran(self, tensor):
        if self.boxcox:
            tensor = scipy.special.boxcox(tensor, self.opt_lambda)
        if self.normalizer:
            tensor = (tensor - self.mean) / self.std
        return tensor

    def inv_tran(self, tensor):
        if self.normalizer:
            tensor = tensor * self.std + self.mean
        if self.boxcox:
            tensor = scipy.special.inv_boxcox(tensor, self.opt_lambda)
        return tensor

    def state_dict(self):
        result = {'normalizer': self.normalizer,
                  'boxcox': self.boxcox}
        if self.normalizer:
            result['mean'] = self.mean
            result['std'] = self.std
        if self.boxcox:
            result['opt_lambda'] = self.opt_lambda
        return result

    def load_state_dict(self, state_dict):
        self.normalizer = state_dict['normalizer']
        self.boxcox = state_dict['boxcox']
        if self.normalizer:
            self.mean = state_dict['mean']
            self.std = state_dict['std']
            print(f'Load state dict, mean: {self.mean}, std: {self.std}')
        if self.boxcox:
            self.opt_lambda = state_dict['opt_lambda']
            print('Load state dict, optimal lambda value:', self.opt_lambda)


def save_model(state, model_dict, model_state_dict, path, is_best):
    model_dir = os.path.join(path, 'model.pt')
    with package.PackageExporter(model_dir, verbose=False) as exp:
        exp.intern('deeph.**')
        exp.extern([
            'scipy.**', 'numpy.**', 'torch_geometric.**', 'sklearn.**',
            'torch_scatter.**', 'torch_sparse.**', 'torch_sparse.**', 'torch_cluster.**', 'torch_spline_conv.**',
            'pyparsing', 'jinja2', 'sys', 'mkl', 'io', 'setuptools.**', 'rdkit.Chem', 'tqdm',
            '__future__', '_operator', '_ctypes', 'six.moves.urllib', 'ase', 'matplotlib.pyplot', 'sympy', 'networkx',
        ])
        exp.save_pickle('checkpoint', 'model.pkl', state | model_dict)
    torch.save(state | model_state_dict, os.path.join(path, 'state_dict.pkl'))
    if is_best:
        shutil.copyfile(os.path.join(path, 'model.pt'), os.path.join(path, 'best_model.pt'))
        shutil.copyfile(os.path.join(path, 'state_dict.pkl'), os.path.join(path, 'best_state_dict.pkl'))


def write_ham_h5(hoppings_dict, path):
    fid = h5py.File(path, "w")
    for k, v in hoppings_dict.items():
        fid[k] = v
    fid.close()


def write_ham_npz(hoppings_dict, path):
    np.savez(path, **hoppings_dict)


def write_ham(hoppings_dict, path):
    os.makedirs(path, exist_ok=True)
    for key_term, matrix in hoppings_dict.items():
        np.savetxt(os.path.join(path, f'{key_term}_real.dat'), matrix)


def get_config(args):
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'default.ini'))
    for config_file in args:
        assert os.path.exists(config_file)
        config.read(config_file)
    if config['basic']['target'] == 'O_ij':
        assert config['basic']['O_component'] in ['H_minimum', 'H_minimum_withNA', 'H', 'Rho']
    if config['basic']['target'] == 'E_ij':
        assert config['basic']['energy_component'] in ['xc', 'delta_ee', 'both', 'summation', 'E_ij']
    else:
        assert config['hyperparameter']['criterion'] in ['MaskMSELoss']
    assert config['basic']['target'] in ['hamiltonian']
    assert config['basic']['interface'] in ['h5', 'h5_rc_only', 'h5_Eij', 'npz', 'npz_rc_only']
    assert config['network']['aggr'] in ['add', 'mean', 'max']
    assert config['network']['distance_expansion'] in ['GaussianBasis', 'BesselBasis', 'ExpBernsteinBasis']
    assert config['network']['normalization'] in ['BatchNorm', 'LayerNorm', 'PairNorm', 'InstanceNorm', 'GraphNorm',
                                                  'DiffGroupNorm', 'None']
    assert config['network']['atom_update_net'] in ['CGConv', 'GAT', 'PAINN']
    assert config['hyperparameter']['optimizer'] in ['sgd', 'sgdm', 'adam', 'adamW', 'adagrad', 'RMSprop', 'lbfgs']
    assert config['hyperparameter']['lr_scheduler'] in ['', 'MultiStepLR', 'ReduceLROnPlateau', 'CyclicLR']

    return config


def get_inference_config(*args):
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'inference', 'inference_default.ini'))
    for config_file in args:
        config.read(config_file)

    return config


def get_preprocess_config(*args):
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'preprocess', 'preprocess_default.ini'))
    for config_file in args:
        config.read(config_file)
    assert config['basic']['target'] in ['hamiltonian', 'density_matrix', 'phiVdphi']
    assert config['basic']['interface'] in ['openmx', 'abacus', 'aims', 'siesta']
    assert config['basic']['multiprocessing'] in ['False'], 'multiprocessing is not yet implemented'

    return config
